"""Settings shared by the split builder and the tf.data pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "Dataset" / "raw"
INVENTORY_CSV = ROOT / "artifacts" / "reports" / "dataset_inventory.csv"
SPLITS_DIR = ROOT / "data" / "splits"
PLOTS_DIR = ROOT / "artifacts" / "plots"

IMAGE_SIZE = (224, 224)
CHANNELS = 3
BATCH_SIZE = 32
SEED = 42

# Fractions of each class's images assigned to each split. The original
# valid/ folder shares leaves with train/ and test/ is a copy of valid/, so
# both are pooled and re-split by leaf instead of being used as-is.
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

# Training-only augmentation strengths (fractions, as Keras layers expect).
AUGMENT = {
    "rotation": 0.10,     # +/- 36 degrees
    "zoom": 0.10,
    "translation": 0.10,
    "brightness": 0.10,
    "contrast": 0.10,
}
