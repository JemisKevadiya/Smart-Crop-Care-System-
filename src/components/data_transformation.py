from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from src import logger
from src.utils.common import read_yaml
from src.utils.exception import CustomException


class DataTransformation:
    def __init__(self, config_path: Path = Path("config/config.yaml"), params_path: Path = Path("params.yaml")):
        config = read_yaml(config_path).data_transformation
        params = read_yaml(params_path)

        self.processed_data_dir = Path(config.processed_data_dir).resolve()
        self.class_names_path = Path(config.class_names_path).resolve()
        self.image_size = (params.image_size, params.image_size)
        self.batch_size = params.batch_size

        self.train_dir = self.processed_data_dir / "train"
        self.valid_dir = self.processed_data_dir / "valid"

        self.augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ],
            name="augmentation",
        )

    def _verify_dirs(self) -> None:
        for split_dir in (self.train_dir, self.valid_dir):
            if not split_dir.exists():
                raise CustomException(f"Expected dataset directory not found: {split_dir}")

    def _load_dataset(self, directory: Path, shuffle: bool) -> tf.data.Dataset:
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=shuffle,
            seed=42,
        )

    def _save_class_names(self, class_names: List[str]) -> None:
        self.class_names_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.class_names_path, np.array(class_names))
        logger.info(f"Saved {len(class_names)} class names to: {self.class_names_path}")

    def get_train_valid_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        self._verify_dirs()

        train_ds = self._load_dataset(self.train_dir, shuffle=True)
        valid_ds = self._load_dataset(self.valid_dir, shuffle=False)

        class_names = train_ds.class_names
        self._save_class_names(class_names)

        train_ds = train_ds.map(
            lambda x, y: (self.augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

        valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

        logger.info(
            f"Prepared train dataset ({len(train_ds)} batches) and valid dataset ({len(valid_ds)} batches) "
            f"at image size {self.image_size}, batch size {self.batch_size}"
        )
        return train_ds, valid_ds


if __name__ == "__main__":
    DataTransformation().get_train_valid_datasets()
