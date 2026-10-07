"""LeafLiteNet: a lightweight CNN trained from scratch for leaf disease classification.

Input is the same tensor the ResNet50 pipeline produces (224x224x3 after
resnet50.preprocess_input: BGR, ImageNet means subtracted, roughly [-124, 152]),
so the data pipeline, the app and the leaf scope check work unchanged. The
first layer rescales it to about [-1, 1].

    Rescaling(1/127.5)
    Stem       Conv 3x3, stride 2 -> BN -> SiLU                     112x112
    Stages     inverted residual blocks (STAGES below)              112 -> 7
    Head       Conv 1x1 -> BN -> SiLU -> GlobalAvgPool -> Dropout -> Dense softmax

Inverted residual block (as in MobileNetV2 / EfficientNet "MBConv"):
    1x1 expand (x expansion) -> BN -> SiLU
    3x3 depthwise (stride s) -> BN -> SiLU
    squeeze-and-excitation (optional; channel attention)
    1x1 project -> BN            (linear bottleneck, no activation)
    + input                       (when stride 1 and channels unchanged)

Depthwise convolutions cost roughly k*k*C operations per pixel instead of
k*k*C*C for a normal convolution, which is where most of the saving comes from.
`width` scales every channel count and `use_se` switches squeeze-and-excitation
off, for ablation studies.
"""

import json
from pathlib import Path

from tensorflow import keras

from src.preprocessing import config as data_config

MODEL_DIR = data_config.ROOT / "models" / "lite_cnn"
BEST_MODEL_PATH = MODEL_DIR / "best_lite_cnn.keras"
REPORTS_DIR = data_config.ROOT / "artifacts" / "reports" / "lite_cnn"

# (expansion, output channels, repeats, first stride, squeeze-and-excitation)
STAGES = [
    (1, 16, 1, 1, False),   # 112x112
    (4, 24, 2, 2, False),   # 56x56
    (4, 40, 2, 2, True),    # 28x28
    (4, 80, 3, 2, True),    # 14x14
    (6, 112, 2, 1, True),   # 14x14
    (6, 160, 2, 2, True),   # 7x7
]
STEM_CHANNELS = 24
HEAD_CHANNELS = 640
SE_RATIO = 0.25

TRAIN = {
    "epochs": 40,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "early_stopping_patience": 8,
    "reduce_lr_patience": 3,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-6,
}


def _channels(c, width):
    """Scale a channel count, rounded to a multiple of 8 (efficient on most hardware)."""
    return max(8, int(c * width + 4) // 8 * 8)


def _conv_bn(x, filters, kernel, stride, name, activation=True):
    x = keras.layers.Conv2D(filters, kernel, strides=stride, padding="same", use_bias=False,
                            name=f"{name}_conv")(x)
    x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return keras.layers.Activation("silu", name=f"{name}_act")(x) if activation else x


def squeeze_excite(x, in_channels, name, ratio=SE_RATIO):
    """Channel attention: reweight each channel by a learned 0-1 gate."""
    channels = x.shape[-1]
    s = keras.layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_se_pool")(x)
    s = keras.layers.Conv2D(max(1, int(in_channels * ratio)), 1, activation="silu",
                            name=f"{name}_se_reduce")(s)
    s = keras.layers.Conv2D(channels, 1, activation="sigmoid", name=f"{name}_se_expand")(s)
    return keras.layers.Multiply(name=f"{name}_se_scale")([x, s])


def inverted_residual(x, expansion, out_channels, stride, use_se, name):
    in_channels = x.shape[-1]
    h = x
    if expansion != 1:
        h = _conv_bn(h, in_channels * expansion, 1, 1, f"{name}_expand")
    h = keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False,
                                     name=f"{name}_dw")(h)
    h = keras.layers.BatchNormalization(name=f"{name}_dw_bn")(h)
    h = keras.layers.Activation("silu", name=f"{name}_dw_act")(h)
    if use_se:
        h = squeeze_excite(h, in_channels, name)
    h = _conv_bn(h, out_channels, 1, 1, f"{name}_project", activation=False)
    if stride == 1 and in_channels == out_channels:
        h = keras.layers.Add(name=f"{name}_add")([x, h])
    return h


def build_model(num_classes, width=1.0, use_se=True, dropout=TRAIN["dropout"],
                input_shape=(*data_config.IMAGE_SIZE, data_config.CHANNELS)):
    inputs = keras.Input(shape=input_shape, name="image")
    x = keras.layers.Rescaling(1 / 127.5, name="rescale")(inputs)
    x = _conv_bn(x, _channels(STEM_CHANNELS, width), 3, 2, "stem")
    for s, (expansion, channels, repeats, stride, se) in enumerate(STAGES, start=1):
        for r in range(repeats):
            x = inverted_residual(x, expansion, _channels(channels, width),
                                  stride if r == 0 else 1, se and use_se, f"stage{s}_block{r + 1}")
    x = _conv_bn(x, _channels(HEAD_CHANNELS, width), 1, 1, "head")
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    suffix = ("" if width == 1.0 else f"_w{width:g}") + ("" if use_se else "_noSE")
    return keras.Model(inputs, outputs, name=f"leaflitenet{suffix}")


def count_macs(model):
    """Multiply-accumulate operations for one image (convolutions and dense layers)."""
    total = 0
    for layer in model.layers:
        if isinstance(layer, keras.Model):  # nested model, e.g. the ResNet50 backbone
            total += count_macs(layer)
        elif isinstance(layer, keras.layers.DepthwiseConv2D):
            h, w, c = layer.output.shape[1:]
            k = layer.kernel_size
            total += h * w * c * k[0] * k[1]
        elif isinstance(layer, keras.layers.Conv2D):
            h, w, c_out = layer.output.shape[1:]
            k = layer.kernel_size
            total += h * w * c_out * k[0] * k[1] * layer.input.shape[-1]
        elif isinstance(layer, keras.layers.Dense):
            total += layer.input.shape[-1] * layer.units
    return int(total)


# --- Training -----------------------------------------------------------------------
# Training uses a wrapper model: images in [0, 255] -> augmentation -> the same
# preprocessing as resnet50.preprocess_input -> the core network. Augmenting inside
# the model runs it on the GPU (Colab has only 2 CPU cores for the data pipeline).
# Only the core network is saved, so the saved model takes the same input as ResNet50.

IMAGENET_BGR_MEAN = (103.939, 116.779, 123.68)


class CaffePreprocess(keras.layers.Layer):
    """Clip to [0, 255], RGB -> BGR, subtract ImageNet means (= resnet50.preprocess_input)."""

    def call(self, x):
        x = keras.ops.clip(x, 0.0, 255.0)
        return keras.ops.flip(x, axis=-1) - keras.ops.convert_to_tensor(IMAGENET_BGR_MEAN)


def build_training_model(core, seed=data_config.SEED):
    from src.preprocessing import pipeline

    inputs = keras.Input(shape=core.input_shape[1:], name="image_0_255")
    x = pipeline.build_augmenter(seed)(inputs)
    x = CaffePreprocess(name="preprocess")(x)
    return keras.Model(inputs, core(x), name=f"train_{core.name}")


class SaveBestCore(keras.callbacks.Callback):
    """Save the core network whenever val_accuracy improves."""

    def __init__(self, core, path, best=None):
        super().__init__()
        self.core, self.path, self.best = core, Path(path), best

    def on_epoch_end(self, epoch, logs=None):
        acc = (logs or {}).get("val_accuracy")
        if acc is not None and (self.best is None or acc > self.best):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.core.save(self.path)
            print(f"\nval_accuracy improved to {acc:.4f}; saved {self.path}", flush=True)
            self.best = acc


def run_paths(run, root=data_config.ROOT):
    root = Path(root)
    return {"model": root / "models" / "lite_cnn" / run / "best.keras",
            "reports": root / "artifacts" / "reports" / "lite_cnn" / run,
            "state": root / "models" / "lite_cnn" / run / "training_state"}


def train(run="leaflitenet", width=1.0, use_se=True, epochs=TRAIN["epochs"],
          batch_size=data_config.BATCH_SIZE, root=data_config.ROOT, cache_dir=None,
          steps_per_epoch=None, validation_steps=None):
    """Train from scratch; resumes from the last completed epoch if interrupted."""
    import pandas as pd

    from src.preprocessing import pipeline, splits

    paths = run_paths(run, root)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    log_path = paths["reports"] / "training_log.csv"
    resuming = paths["state"].exists() and any(paths["state"].iterdir())
    if not resuming:
        log_path.unlink(missing_ok=True)
        paths["model"].unlink(missing_ok=True)

    class_names = splits.load_class_names()
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def cache(split):
        return Path(cache_dir) / f"{split}.cache" if cache_dir else None

    train_ds = pipeline.make_dataset("train", batch_size, augment=False, preprocess=False,
                                     cache=cache("train"))
    val_ds = pipeline.make_dataset("val", batch_size, preprocess=False, cache=cache("val"))
    if steps_per_epoch:
        train_ds = train_ds.repeat()

    core = build_model(len(class_names), width=width, use_se=use_se)
    model = build_training_model(core)
    model.compile(
        optimizer=keras.optimizers.AdamW(TRAIN["learning_rate"],
                                         weight_decay=TRAIN["weight_decay"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")])

    best = None
    if log_path.exists() and log_path.stat().st_size:
        best = float(pd.read_csv(log_path)["val_accuracy"].max())
    callbacks = [
        keras.callbacks.BackupAndRestore(str(paths["state"])),
        SaveBestCore(core, paths["model"], best),
        keras.callbacks.EarlyStopping(monitor="val_loss",
                                      patience=TRAIN["early_stopping_patience"], verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=TRAIN["reduce_lr_factor"],
                                          patience=TRAIN["reduce_lr_patience"],
                                          min_lr=TRAIN["min_lr"], verbose=1),
        keras.callbacks.CSVLogger(str(log_path), append=True),
    ]
    state = "resuming" if resuming else "starting"
    print(f"{run}: {core.count_params():,} parameters, {count_macs(core) / 1e6:.0f}M MACs, "
          f"{state}", flush=True)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps, callbacks=callbacks, verbose=2)

    run_config = {"run": run, "width": width, "use_se": use_se, "batch_size": batch_size,
                  "epochs_max": epochs, "train": TRAIN, "params": core.count_params(),
                  "macs": count_macs(core)}
    (paths["reports"] / "run_config.json").write_text(json.dumps(run_config, indent=2))
    return paths


# --- Evaluation ---------------------------------------------------------------------

def evaluate(run="leaflitenet", root=data_config.ROOT, limit=None):
    """Same metrics as the ResNet50 evaluation, written to the run's reports folder.

    `limit` evaluates only the first N images per split (smoke tests).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    from src.model.evaluate import plot_confusion, split_metrics
    from src.preprocessing import pipeline, splits

    paths = run_paths(run, root)
    out = paths["reports"]
    out.mkdir(parents=True, exist_ok=True)
    model = keras.models.load_model(paths["model"])
    class_names = splits.load_class_names()
    labels = range(len(class_names))

    results, preds = {}, {}
    for split in ("train", "val", "test"):
        df = splits.load_split(split)
        if limit:
            df = df.head(limit)
        ds = pipeline.make_dataset(split, 64, training=False, manifest=df)
        probs = model.predict(ds, verbose=2)
        y = df["label_idx"].to_numpy()
        preds[split] = (y, probs)
        results[split] = split_metrics(y, probs, len(class_names))
        print(split, json.dumps(results[split]), flush=True)

    y, probs = preds["test"]
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y, y_pred, labels=labels)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        out / "confusion_matrix_test.csv", index_label="true \\ predicted")
    plot_confusion(cm, class_names,
                   f"{model.name} · test split ({len(y):,} images, accuracy "
                   f"{results['test']['accuracy']:.4f})", path=out / "confusion_matrix.png")
    report = classification_report(y, y_pred, labels=labels, target_names=class_names,
                                   digits=4, zero_division=0)
    (out / "classification_report.txt").write_text(
        f"{model.name} - test split ({len(y)} images)\n\n{report}", encoding="utf-8")
    per_class = classification_report(y, y_pred, labels=labels, target_names=class_names,
                                      output_dict=True, zero_division=0)

    log_path = out / "training_log.csv"
    log = pd.read_csv(log_path) if log_path.exists() and log_path.stat().st_size else None
    if log is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ("accuracy", "loss")):
            ax.plot(log["epoch"] + 1, log[metric], marker="o", label="train (augmented)")
            ax.plot(log["epoch"] + 1, log[f"val_{metric}"], marker="o", label="validation")
            ax.set_xlabel("Epoch")
            ax.set_title(f"{model.name} {metric}")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "training_curves.png", dpi=130)
        plt.close(fig)

    lowest = sorted(((c, per_class[c]["f1-score"]) for c in class_names), key=lambda t: t[1])
    metrics = {
        "model": model.name, "model_path": str(paths["model"]),
        "params": model.count_params(), "macs": count_macs(model),
        "file_size_mb": round(paths["model"].stat().st_size / 1e6, 2),
        "evaluation": results,
        "epochs_trained": None if log is None else int(len(log)),
        "best_val_accuracy_epoch": None if log is None else int(log["val_accuracy"].idxmax()) + 1,
        "test_lowest_f1_classes": dict(lowest[:5]),
        "limit_per_split": limit,
    }
    (out / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Wrote", out / "model_metrics.json")
    return metrics
