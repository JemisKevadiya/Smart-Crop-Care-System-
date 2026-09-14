import os
import shutil
from pathlib import Path

import kagglehub

from src import logger
from src.utils.common import read_yaml, create_directories


class DataIngestion:
    def __init__(self, config_path: Path = Path("config/config.yaml")):
        self.config = read_yaml(config_path).data_ingestion

    def download_dataset(self) -> str:
        logger.info(f"Downloading dataset: {self.config.kaggle_dataset}")
        cache_path = kagglehub.dataset_download(self.config.kaggle_dataset)
        logger.info(f"Dataset downloaded to cache: {cache_path}")
        return cache_path

    def copy_to_raw(self, source_path: str) -> None:
        raw_dir = Path(self.config.raw_data_dir)
        create_directories([raw_dir])

        for item in os.listdir(source_path):
            src_item = os.path.join(source_path, item)
            dst_item = raw_dir / item
            if os.path.isdir(src_item):
                shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
            else:
                shutil.copy2(src_item, dst_item)

        logger.info(f"Dataset copied to: {raw_dir}")

    def run(self) -> None:
        cache_path = self.download_dataset()
        self.copy_to_raw(cache_path)


if __name__ == "__main__":
    DataIngestion().run()
