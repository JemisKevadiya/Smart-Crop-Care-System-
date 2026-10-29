"""Which trained disease model the app uses.

    DISEASE_MODEL=efficientnetb0   (default)  EfficientNetB0, full-dataset leaf-grouped split
    DISEASE_MODEL=resnet50                    ResNet50, the original model (fallback)

Set DISEASE_MODEL in the environment or .env. If the chosen model's files are
missing, the app falls back to ResNet50. Each model has its own preprocessing,
leaf reference (non-leaf check) and reports; everything else is shared.
"""

import logging
from dataclasses import dataclass

from src.env import get_setting
from src.preprocessing import config

log = logging.getLogger(__name__)
SETTING = "DISEASE_MODEL"
DEFAULT_MODEL = "efficientnetb0"
FALLBACK_MODEL = "resnet50"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str                 # display name
    folder: str               # under models/
    model_file: str
    preprocessing: str        # "resnet50" (BGR, mean subtraction) or "efficientnet" (raw 0-255 pixels)
    metrics_file: str         # under ROOT
    split_report_file: str    # under ROOT
    finetuned: str            # which backbone layers were fine-tuned in stage 2
    reference_per_class: int  # leaf reference images per class (scripts/build_leaf_reference.py)
    scope_threshold: float

    @property
    def model_path(self):
        return config.ROOT / "models" / self.folder / self.model_file

    @property
    def class_names_path(self):
        return config.ROOT / "models" / self.folder / "class_names.npy"

    @property
    def reference_path(self):
        return config.ROOT / "models" / self.folder / "leaf_reference.npz"

    @property
    def metrics_path(self):
        return config.ROOT / self.metrics_file

    @property
    def split_report_path(self):
        return config.ROOT / self.split_report_file

    def available(self):
        return self.model_path.is_file() and self.class_names_path.is_file()


MODELS = {
    "efficientnetb0": ModelSpec(
        key="efficientnetb0", name="EfficientNetB0", folder="efficientnetb0",
        model_file="best_efficientnetb0.keras", preprocessing="efficientnet",
        metrics_file="artifacts/reports/efficientnetb0/model_metrics.json",
        split_report_file="artifacts/reports/efficientnetb0/split_report.json",
        finetuned="last EfficientNet blocks (block6 and block7)",
        reference_per_class=300, scope_threshold=0.44),
    "resnet50": ModelSpec(
        key="resnet50", name="ResNet50", folder="resnet50",
        model_file="best_resnet50.keras", preprocessing="resnet50",
        metrics_file="artifacts/reports/resnet50/model_metrics.json",
        split_report_file="artifacts/reports/resnet50/split_report.json",
        finetuned="last ResNet block (conv5)",
        reference_per_class=300, scope_threshold=0.60),
}


def select_model(requested=None):
    """The ModelSpec to use: the requested one if its files exist, else ResNet50."""
    key = (requested or get_setting(SETTING) or DEFAULT_MODEL).strip().lower()
    spec = MODELS.get(key)
    if spec is None:
        log.warning("Unknown %s=%r; using %s", SETTING, key, MODELS[DEFAULT_MODEL].name)
        spec = MODELS[DEFAULT_MODEL]
    if not spec.available() and spec.key != FALLBACK_MODEL:
        log.warning("%s files not found in models/%s; falling back to ResNet50", spec.name, spec.folder)
        spec = MODELS[FALLBACK_MODEL]
    return spec


ACTIVE = select_model()


def preprocess_input(images, spec=None):
    """Model-specific pixel preprocessing for float images in [0, 255]."""
    spec = spec or ACTIVE
    if spec.preprocessing == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

        return resnet_preprocess(images)
    # EfficientNet rescales and normalises inside the model, so it takes raw pixels.
    return images
