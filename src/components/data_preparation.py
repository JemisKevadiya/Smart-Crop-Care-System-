import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src import logger
from src.utils.common import create_directories, read_yaml, save_json
from src.utils.exception import CustomException

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DataPreparation:
    def __init__(self, config_path: Path = Path("config/config.yaml")):
        config = read_yaml(config_path).data_preparation
        self.raw_data_dir = Path(config.raw_data_dir).resolve()
        self.processed_data_dir = Path(config.processed_data_dir).resolve()
        self.train_sample_ratio = config.train_sample_ratio
        self.random_seed = config.random_seed

    def _verify_raw_dataset(self) -> None:
        if not self.raw_data_dir.exists():
            raise CustomException(f"Raw dataset not found at: {self.raw_data_dir}")
        logger.info(f"Raw dataset verified at: {self.raw_data_dir}")

    @staticmethod
    def _list_images(directory: Path) -> List[Path]:
        return [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    @staticmethod
    def _discover_classes(split_dir: Path) -> List[str]:
        return sorted(d.name for d in split_dir.iterdir() if d.is_dir())

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    @staticmethod
    def _total_bytes(files: List[Path]) -> int:
        return sum(f.stat().st_size for f in files)

    def _check_disk_space(self, required_bytes: int, label: str) -> None:
        free_bytes = shutil.disk_usage(self.processed_data_dir.anchor).free
        required_gb = required_bytes / (1024 ** 3)
        free_gb = free_bytes / (1024 ** 3)
        logger.info(f"[{label}] required space: ~{required_gb:.2f} GB | available: ~{free_gb:.2f} GB")
        if required_bytes > free_bytes:
            raise CustomException(
                f"Insufficient disk space to prepare '{label}': requires ~{required_gb:.2f} GB "
                f"but only ~{free_gb:.2f} GB is available on {self.processed_data_dir.anchor}. "
                f"Stopping instead of copying partially."
            )

    def _prepare_train_split(self, classes: List[str]) -> Dict[str, Dict[str, int]]:
        rng = random.Random(self.random_seed)
        selection_map: Dict[str, List[Path]] = {}
        report: Dict[str, Dict[str, int]] = {}

        for class_name in classes:
            images = self._list_images(self.raw_data_dir / "train" / class_name)
            sample_size = min(round(len(images) * self.train_sample_ratio), len(images))
            selected = rng.sample(images, sample_size)
            selection_map[class_name] = selected
            report[class_name] = {"original": len(images), "selected": len(selected)}

        all_selected = [f for files in selection_map.values() for f in files]
        self._check_disk_space(self._total_bytes(all_selected), "train (~50%)")

        for class_name, selected in selection_map.items():
            dst_dir = self.processed_data_dir / "train" / class_name
            create_directories([dst_dir])
            for src_file in selected:
                self._link_or_copy(src_file, dst_dir / src_file.name)

        total_selected = sum(v["selected"] for v in report.values())
        logger.info(f"Train split prepared: {total_selected} images selected across {len(classes)} classes")
        return report

    def _prepare_classwise_split(self, split_name: str, classes: List[str]) -> Dict[str, int]:
        file_map: Dict[str, List[Path]] = {}
        report: Dict[str, int] = {}

        for class_name in classes:
            images = self._list_images(self.raw_data_dir / split_name / class_name)
            file_map[class_name] = images
            report[class_name] = len(images)

        all_files = [f for files in file_map.values() for f in files]
        self._check_disk_space(self._total_bytes(all_files), f"{split_name} (100%)")

        for class_name, images in file_map.items():
            dst_dir = self.processed_data_dir / split_name / class_name
            create_directories([dst_dir])
            for src_file in images:
                self._link_or_copy(src_file, dst_dir / src_file.name)

        logger.info(f"{split_name} split prepared: {sum(report.values())} images across {len(classes)} classes")
        return report

    def _prepare_flat_split(self, split_name: str) -> int:
        images = self._list_images(self.raw_data_dir / split_name)
        self._check_disk_space(self._total_bytes(images), f"{split_name} (100%, unlabeled)")

        dst_dir = self.processed_data_dir / split_name
        create_directories([dst_dir])
        for src_file in images:
            self._link_or_copy(src_file, dst_dir / src_file.name)

        logger.info(f"{split_name} split prepared: {len(images)} unlabeled images")
        return len(images)

    def run(self) -> dict:
        self._verify_raw_dataset()

        train_dir = self.raw_data_dir / "train"
        classes = self._discover_classes(train_dir)
        logger.info(f"Discovered {len(classes)} classes in train split")

        train_report = self._prepare_train_split(classes)

        valid_report: Dict[str, int] = {}
        valid_dir = self.raw_data_dir / "valid"
        if valid_dir.exists():
            valid_classes = self._discover_classes(valid_dir)
            valid_report = self._prepare_classwise_split("valid", valid_classes)
        else:
            logger.info("No 'valid' split found in raw dataset, skipping")

        test_report: Optional[Dict[str, int]] = None
        test_flat_count = 0
        test_dir = self.raw_data_dir / "test"
        if test_dir.exists():
            test_classes = self._discover_classes(test_dir)
            if test_classes:
                test_report = self._prepare_classwise_split("test", test_classes)
            else:
                test_flat_count = self._prepare_flat_split("test")
        else:
            logger.info("No 'test' split found in raw dataset, skipping")

        summary = {
            "raw_dataset_path": str(self.raw_data_dir),
            "processed_dataset_path": str(self.processed_data_dir),
            "num_classes": len(classes),
            "classes": classes,
            "train_sample_ratio": self.train_sample_ratio,
            "random_seed": self.random_seed,
            "train": {
                "original_total": sum(v["original"] for v in train_report.values()),
                "selected_total": sum(v["selected"] for v in train_report.values()),
                "per_class": train_report,
            },
            "valid": {
                "total": sum(valid_report.values()),
                "per_class": valid_report,
            },
            "test": {
                "structure": "classwise" if test_report is not None else "flat_unlabeled",
                "total": sum(test_report.values()) if test_report is not None else test_flat_count,
                "per_class": test_report if test_report is not None else {},
            },
        }

        create_directories([self.processed_data_dir])
        save_json(self.processed_data_dir / "summary_report.json", summary)

        return summary


if __name__ == "__main__":
    DataPreparation().run()
