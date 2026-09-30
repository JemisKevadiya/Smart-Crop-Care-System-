"""Predict the plant disease in a leaf image.

    predictor = DiseasePredictor()
    result = predictor.predict(uploaded_file)   # path, bytes, file-like or PIL image
    result.class_name, result.confidence, result.is_confident, result.top_k

A prediction whose top probability is below `confidence_threshold` is still
returned, but flagged `is_confident=False` with a message so the UI can tell
the user the result is uncertain instead of presenting it as a diagnosis.
Limitation: confidence does not detect non-leaf images. The model only knows
these 38 classes and is often confidently wrong on unrelated pictures (e.g.
random noise scores 0.999). Blank single-colour images are rejected during
validation; other out-of-scope images are not.
"""

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .model_loader import CLASS_NAMES_PATH, MODEL_PATH, ModelLoadError, load_classifier
from .preprocessing import ImageValidationError, preprocess_image

# Chosen on the validation split (scripts/test_prediction.py): at 0.80, 3.2% of
# images are flagged, 65% of the model's errors are among them, and accuracy on
# the unflagged rest is 99.2%.
DEFAULT_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_TOP_K = 3


class PredictionError(RuntimeError):
    """The model failed to produce a usable prediction."""


def parse_class_name(class_name):
    """'Corn_(maize)___Common_rust_' -> ('Corn (maize)', 'Common rust', False)."""
    crop, _, condition = class_name.partition("___")
    crop = crop.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()
    healthy = condition.lower() == "healthy"
    return crop, ("Healthy" if healthy else condition), healthy


@dataclass
class Candidate:
    class_name: str
    crop: str
    disease: str
    probability: float


@dataclass
class Prediction:
    class_name: str
    crop: str
    disease: str
    is_healthy: bool
    confidence: float
    is_confident: bool
    message: str
    top_k: list = field(default_factory=list)
    inference_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


class DiseasePredictor:
    """Loads the model once and predicts on individual images."""

    def __init__(self, model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH,
                 confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, top_k=DEFAULT_TOP_K):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.model, self.class_names = load_classifier(model_path, class_names_path)
        self.confidence_threshold = confidence_threshold
        self.top_k = max(1, min(top_k, len(self.class_names)))

    def predict_probabilities(self, batch):
        """Run the model on a preprocessed (N, 224, 224, 3) batch."""
        try:
            probs = np.asarray(self.model(batch, training=False))
        except Exception as exc:  # noqa: BLE001
            raise PredictionError(f"Model inference failed: {exc}") from exc
        if probs.shape != (len(batch), len(self.class_names)) or not np.isfinite(probs).all():
            raise PredictionError(f"Model returned invalid output with shape {probs.shape}")
        return probs

    def predict(self, source):
        """Validate, preprocess and classify one image.

        Raises ImageValidationError for bad input and PredictionError if the
        model fails; both carry a message suitable for showing to the user.
        """
        batch = preprocess_image(source)
        start = time.perf_counter()
        probs = self.predict_probabilities(batch)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        return self._build_prediction(probs, elapsed_ms)

    def _build_prediction(self, probs, elapsed_ms=0.0):
        order = np.argsort(probs)[::-1][: self.top_k]
        candidates = []
        for i in order:
            crop, disease, _ = parse_class_name(self.class_names[i])
            candidates.append(Candidate(self.class_names[i], crop, disease, float(probs[i])))

        best = candidates[0]
        crop, disease, healthy = parse_class_name(best.class_name)
        confident = best.probability >= self.confidence_threshold
        if confident:
            message = (f"{crop} leaf looks healthy." if healthy
                       else f"{crop}: {disease} detected.")
        else:
            alternatives = ", ".join(f"{c.crop} - {c.disease} ({c.probability:.0%})"
                                     for c in candidates)
            message = (f"Low confidence ({best.probability:.0%} < "
                       f"{self.confidence_threshold:.0%}). The image may be unclear, not a "
                       f"leaf, or a crop/disease the model does not know. Most likely: "
                       f"{alternatives}. Try a closer, well-lit photo of a single leaf.")
        return Prediction(
            class_name=best.class_name, crop=crop, disease=disease, is_healthy=healthy,
            confidence=best.probability, is_confident=confident, message=message,
            top_k=candidates, inference_ms=round(elapsed_ms, 1))


__all__ = ["DiseasePredictor", "Prediction", "Candidate", "PredictionError",
           "ImageValidationError", "ModelLoadError", "parse_class_name"]
