"""Load the trained ResNet50 classifier and its class names (cached)."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from src.preprocessing import config

MODEL_PATH = config.ROOT / "models" / "resnet50" / "best_resnet50.keras"
CLASS_NAMES_PATH = config.ROOT / "models" / "resnet50" / "class_names.npy"


class ModelLoadError(RuntimeError):
    """The model or class names could not be loaded or do not match."""


@lru_cache(maxsize=4)
def load_class_names(path=CLASS_NAMES_PATH):
    """Class names in model output order."""
    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(f"Class names file not found: {path}")
    try:
        names = [str(n) for n in np.load(path, allow_pickle=False)]
    except Exception as exc:  # noqa: BLE001 - any read failure is fatal here
        raise ModelLoadError(f"Could not read class names from {path}: {exc}") from exc
    if not names or len(set(names)) != len(names):
        raise ModelLoadError(f"Class names in {path} are empty or contain duplicates")
    return tuple(names)


@lru_cache(maxsize=4)
def load_model(path=MODEL_PATH):
    """Load the Keras model once per process (inference only, not compiled)."""
    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(
            f"Model file not found: {path}. Train it with scripts/train_resnet50.py.")
    from tensorflow import keras  # deferred: TensorFlow import is slow

    try:
        return keras.models.load_model(path, compile=False)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Could not load model from {path}: {exc}") from exc


def load_classifier(model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
    """Return (model, class_names) after checking they belong together."""
    class_names = load_class_names(Path(class_names_path))
    model = load_model(Path(model_path))
    outputs = model.output_shape[-1]
    if outputs != len(class_names):
        raise ModelLoadError(
            f"Model predicts {outputs} classes but {len(class_names)} class names were loaded")
    expected_input = (None, *config.IMAGE_SIZE, config.CHANNELS)
    if tuple(model.input_shape) != expected_input:
        raise ModelLoadError(f"Model input {model.input_shape} != expected {expected_input}")
    return model, class_names
