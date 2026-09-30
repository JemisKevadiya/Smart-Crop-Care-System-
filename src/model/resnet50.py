"""ResNet50 transfer-learning classifier with two-stage training.

Architecture:
    Input (224x224x3, already resnet50.preprocess_input-ed by the pipeline)
    -> ResNet50 (ImageNet weights, include_top=False, BatchNorm in inference mode)
    -> GlobalAveragePooling2D -> Dropout -> Dense (ReLU) -> Dense softmax (num_classes)

Stage 1 trains only the new head with the backbone frozen. Stage 2 unfreezes
the upper part of the backbone (from `finetune_from` onward, BatchNorm layers
stay frozen) and continues with a much smaller learning rate.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from src.preprocessing import config as data_config
from src.preprocessing import pipeline, splits

MODEL_DIR = data_config.ROOT / "models" / "resnet50"
BEST_MODEL_PATH = MODEL_DIR / "best_resnet50.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.npy"
REPORTS_DIR = data_config.ROOT / "artifacts" / "reports"

TRAIN = {
    "dropout": 0.3,
    "dense_units": 256,
    "stage1_epochs": 5,
    "stage1_lr": 1e-3,
    "stage2_epochs": 5,
    "stage2_lr": 1e-5,
    # First layer of ResNet50's last residual stage; everything from here up is
    # fine-tuned in stage 2 (conv5_x, ~15M of the backbone's 23.6M parameters).
    "finetune_from": "conv5_block1_1_conv",
    "early_stopping_patience": 5,
    "reduce_lr_patience": 2,
    "reduce_lr_factor": 0.2,
    "min_lr": 1e-7,
}


def load_class_names():
    """Class names in label-index order, derived from the dataset split manifests."""
    return splits.load_class_names()


def save_class_names(class_names, path=CLASS_NAMES_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(class_names))


def build_model(num_classes, dropout=TRAIN["dropout"], dense_units=TRAIN["dense_units"],
                weights="imagenet"):
    """Return (model, backbone). The backbone starts frozen (stage 1)."""
    inputs = keras.Input(shape=(*data_config.IMAGE_SIZE, data_config.CHANNELS), name="image")
    backbone = keras.applications.ResNet50(
        include_top=False, weights=weights, input_shape=inputs.shape[1:])
    backbone.trainable = False
    # training=False keeps BatchNorm statistics fixed in both stages, which is
    # what keeps fine-tuning on a small learning rate stable.
    x = backbone(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="resnet50_plant_disease"), backbone


def unfreeze_upper_layers(backbone, from_layer=TRAIN["finetune_from"]):
    """Make layers from `from_layer` onward trainable, except BatchNorm."""
    names = [layer.name for layer in backbone.layers]
    start = names.index(from_layer)
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= start and not isinstance(layer, keras.layers.BatchNormalization)
    return start


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")],
    )


def count_params(model):
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    frozen = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    return {"trainable": trainable, "non_trainable": frozen}



def make_callbacks(checkpoint_path, log_path, backup_dir, best_so_far=None):
    """Checkpoint on best val_accuracy; stop and reduce LR on val_loss plateaus.

    `best_so_far` is the best val_accuracy already on disk (from stage 1, or
    from before an interruption), so the checkpoint is only overwritten by an
    actual improvement. BackupAndRestore lets an interrupted stage resume from
    its last completed epoch; the CSV log is appended to for the same reason.
    """
    return [
        keras.callbacks.BackupAndRestore(str(backup_dir)),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", mode="max", save_best_only=True,
            initial_value_threshold=best_so_far, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=TRAIN["early_stopping_patience"],
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=TRAIN["reduce_lr_factor"],
            patience=TRAIN["reduce_lr_patience"], min_lr=TRAIN["min_lr"], verbose=1),
        keras.callbacks.CSVLogger(str(log_path), append=True),
    ]


def read_log(path):
    """Per-epoch metrics of one stage as a dict of lists ({} if absent)."""
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return {}
    df = pd.read_csv(path)
    return {c: df[c].tolist() for c in df.columns}


def best_val_accuracy(*logs):
    values = [v for log in logs for v in log.get("val_accuracy", [])]
    return max(values) if values else None


def train(stage1_epochs=TRAIN["stage1_epochs"], stage2_epochs=TRAIN["stage2_epochs"],
          batch_size=data_config.BATCH_SIZE, steps_per_epoch=None, validation_steps=None,
          model_path=BEST_MODEL_PATH, reports_dir=REPORTS_DIR):
    """Run both stages (resuming if a previous run was interrupted).

    `steps_per_epoch` / `validation_steps` limit each epoch (used for smoke tests);
    leave them as None to train on the full splits.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    state_dir = model_path.parent / "training_state"
    stage1_weights = state_dir / "stage1_final.weights.h5"
    logs = {s: reports_dir / f"training_{s}.csv" for s in ("stage1", "stage2")}
    backups = {s: state_dir / f"backup_{s}" for s in ("stage1", "stage2")}

    class_names = load_class_names()
    save_class_names(class_names, model_path.parent / "class_names.npy")

    train_ds = pipeline.make_dataset("train", batch_size=batch_size)
    val_ds = pipeline.make_dataset("val", batch_size=batch_size)
    if steps_per_epoch:
        train_ds = train_ds.repeat()

    model, backbone = build_model(num_classes=len(class_names))
    fit_kwargs = dict(validation_data=val_ds, steps_per_epoch=steps_per_epoch,
                      validation_steps=validation_steps, verbose=2)

    def start_stage(stage):
        """Clear a previous run's log unless this stage is resuming from a backup."""
        # BackupAndRestore stores the resumed epoch on the model and fit() keeps
        # using it; reset it so a resumed stage 1 cannot shift stage 2's epochs.
        model._initial_epoch = None
        resuming = backups[stage].exists() and any(backups[stage].iterdir())
        if not resuming:
            logs[stage].unlink(missing_ok=True)
        print(f"{stage}: {'resuming from backup' if resuming else 'starting'}", flush=True)

    # Stage 1: frozen backbone, train the head.
    if stage1_weights.exists():
        print("stage1: already complete, loading its final weights", flush=True)
        model.load_weights(stage1_weights)
    else:
        start_stage("stage1")
        if not backups["stage1"].exists():
            model_path.unlink(missing_ok=True)
        compile_model(model, TRAIN["stage1_lr"])
        print("Stage 1 params:", count_params(model), flush=True)
        model.fit(train_ds, epochs=stage1_epochs,
                  callbacks=make_callbacks(model_path, logs["stage1"], backups["stage1"],
                                           best_so_far=best_val_accuracy(read_log(logs["stage1"]))),
                  **fit_kwargs)
        state_dir.mkdir(parents=True, exist_ok=True)
        model.save_weights(stage1_weights)

    # Stage 2: fine-tune conv5_x with a small learning rate.
    start_stage("stage2")
    unfreeze_upper_layers(backbone)
    compile_model(model, TRAIN["stage2_lr"])
    print("Stage 2 params:", count_params(model), flush=True)
    best_so_far = best_val_accuracy(read_log(logs["stage1"]), read_log(logs["stage2"]))
    model.fit(train_ds, epochs=stage2_epochs,
              callbacks=make_callbacks(model_path, logs["stage2"], backups["stage2"],
                                       best_so_far=best_so_far),
              **fit_kwargs)

    history = {"stage1": read_log(logs["stage1"]), "stage2": read_log(logs["stage2"]),
               "config": TRAIN, "batch_size": batch_size, "steps_per_epoch": steps_per_epoch,
               "model_path": str(model_path)}
    with open(reports_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)
    return model, history
