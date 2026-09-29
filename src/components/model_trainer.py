from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf

from src import logger
from src.components.data_transformation import DataTransformation
from src.utils.common import create_directories, read_yaml, save_json


class ModelTrainer:
    def __init__(self, config_path: Path = Path("config/config.yaml"), params_path: Path = Path("params.yaml")):
        config = read_yaml(config_path).model_trainer
        params = read_yaml(params_path)

        self.model_path = Path(config.model_path).resolve()
        self.metrics_path = Path(config.metrics_path).resolve()
        self.log_dir = Path(config.log_dir).resolve()

        self.image_size = params.image_size
        self.epochs = params.epochs
        self.learning_rate = params.learning_rate
        self.fine_tune_base = params.fine_tune_base

    def _build_model(self, num_classes: int) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
        x = tf.keras.applications.resnet.preprocess_input(inputs)

        base_model = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=(self.image_size, self.image_size, 3)
        )
        base_model.trainable = self.fine_tune_base

        x = base_model(x, training=self.fine_tune_base)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name="resnet_crop_disease_classifier")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        create_directories([self.model_path.parent, self.log_dir])
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_path), monitor="val_accuracy", save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
            tf.keras.callbacks.CSVLogger(str(self.log_dir / "training.csv"), append=False),
        ]

    def train(
        self,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
    ) -> dict:
        epochs = epochs or self.epochs

        transformation = DataTransformation()
        train_ds, valid_ds = transformation.get_train_valid_datasets()
        num_classes = int(np.load(transformation.class_names_path).shape[0])

        logger.info(f"Building ResNet50 model for {num_classes} classes (fine_tune_base={self.fine_tune_base})")
        model = self._build_model(num_classes)

        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=self._get_callbacks(),
        )

        metrics = {key: [float(v) for v in values] for key, values in history.history.items()}
        if "val_accuracy" in metrics:
            metrics["final_val_accuracy"] = metrics["val_accuracy"][-1]
        if "val_loss" in metrics:
            metrics["final_val_loss"] = metrics["val_loss"][-1]
        save_json(self.metrics_path, metrics)

        logger.info(f"Training complete. Best model saved at: {self.model_path}")
        return metrics


if __name__ == "__main__":
    ModelTrainer().train()
