from pathlib import Path
from typing import Dict, List

from PIL import Image

from src import logger
from src.utils.common import read_yaml, save_json
from src.utils.exception import CustomException

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DataValidation:
    def __init__(self, config_path: Path = Path("config/config.yaml")):
        config = read_yaml(config_path).data_validation
        self.processed_data_dir = Path(config.processed_data_dir).resolve()
        self.expected_num_classes = config.expected_num_classes
        self.status_file = Path(config.status_file).resolve()

    @staticmethod
    def _list_images(directory: Path) -> List[Path]:
        return [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    @staticmethod
    def _discover_classes(split_dir: Path) -> List[str]:
        return sorted(d.name for d in split_dir.iterdir() if d.is_dir())

    def _validate_classwise_split(self, split_name: str, errors: List[str]) -> Dict[str, int]:
        split_dir = self.processed_data_dir / split_name
        counts: Dict[str, int] = {}

        if not split_dir.exists():
            errors.append(f"Missing split directory: {split_name}")
            return counts

        classes = self._discover_classes(split_dir)
        if len(classes) != self.expected_num_classes:
            errors.append(
                f"'{split_name}' has {len(classes)} classes, expected {self.expected_num_classes}"
            )

        for class_name in classes:
            images = self._list_images(split_dir / class_name)
            counts[class_name] = len(images)
            if len(images) == 0:
                errors.append(f"'{split_name}/{class_name}' has no images")

        return counts

    @staticmethod
    def _find_corrupt_images(files: List[Path]) -> List[str]:
        corrupt = []
        for path in files:
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as e:
                corrupt.append(f"{path}: {e}")
        return corrupt

    def run(self) -> dict:
        if not self.processed_data_dir.exists():
            raise CustomException(f"Processed dataset not found at: {self.processed_data_dir}")

        logger.info(f"Validating dataset at: {self.processed_data_dir}")
        errors: List[str] = []

        train_counts = self._validate_classwise_split("train", errors)
        valid_counts = self._validate_classwise_split("valid", errors)

        test_dir = self.processed_data_dir / "test"
        test_count = len(self._list_images(test_dir)) if test_dir.exists() else 0
        if test_dir.exists() and test_count == 0:
            errors.append("'test' directory exists but has no images")

        logger.info("Scanning all images for corruption (this may take a while)...")
        all_images = (
            self._list_images(self.processed_data_dir / "train")
            + self._list_images(self.processed_data_dir / "valid")
            + (self._list_images(test_dir) if test_dir.exists() else [])
        )
        corrupt_images = self._find_corrupt_images(all_images)
        if corrupt_images:
            errors.append(f"Found {len(corrupt_images)} corrupt/unreadable image(s)")

        is_validated = len(errors) == 0

        report = {
            "processed_data_dir": str(self.processed_data_dir),
            "validation_status": is_validated,
            "expected_num_classes": self.expected_num_classes,
            "train": {"num_classes": len(train_counts), "total_images": sum(train_counts.values()), "per_class": train_counts},
            "valid": {"num_classes": len(valid_counts), "total_images": sum(valid_counts.values()), "per_class": valid_counts},
            "test": {"total_images": test_count},
            "total_images_scanned": len(all_images),
            "corrupt_images": corrupt_images,
            "errors": errors,
        }

        save_json(self.status_file, report)

        if is_validated:
            logger.info(f"Data validation PASSED: {len(all_images)} images checked, 0 issues found")
        else:
            logger.info(f"Data validation FAILED: {len(errors)} issue(s) found, see {self.status_file}")

        return report


if __name__ == "__main__":
    DataValidation().run()
