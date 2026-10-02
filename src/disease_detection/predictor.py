"""Predict the plant disease in a leaf image.

    predictor = DiseasePredictor()
    result = predictor.predict(uploaded_file)   # path, bytes, file-like or PIL image
    result.class_name, result.confidence, result.is_confident, result.top_k

A prediction whose top probability is below `confidence_threshold` is still
returned, but flagged `is_confident=False` with a message so the UI can tell
the user the result is uncertain instead of presenting it as a diagnosis.
Confidence does not detect non-leaf images: the model only knows these 38
classes and is often ~100% confident on unrelated pictures (faces, noise).
So every prediction is also scored by scope_check.LeafScopeChecker, which
compares the image's ResNet50 features with real training leaves. Images that
do not resemble them are flagged `in_scope=False` (and `is_confident=False`).
Blank single-colour images are already rejected during validation.
"""

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .model_loader import CLASS_NAMES_PATH, MODEL_PATH, ModelLoadError, load_classifier
from .preprocessing import ImageValidationError, preprocess_image
from .scope_check import FEATURE_LAYER, REFERENCE_PATH, load_scope_checker

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
    in_scope: bool = True
    scope_score: float | None = None

    def to_dict(self):
        return asdict(self)


class DiseasePredictor:
    """Loads the model once and predicts on individual images."""

    def __init__(self, model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH,
                 confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, top_k=DEFAULT_TOP_K,
                 scope_reference_path=REFERENCE_PATH, scope_checker=None):
        """scope_checker: a LeafScopeChecker, or False to disable the check. By
        default it is loaded from scope_reference_path (skipped if missing)."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.model, self.class_names = load_classifier(model_path, class_names_path)
        self.confidence_threshold = confidence_threshold
        self.top_k = max(1, min(top_k, len(self.class_names)))
        self.scope_checker = (load_scope_checker(scope_reference_path)
                              if scope_checker is None else scope_checker or None)
        from tensorflow import keras  # deferred: TensorFlow import is slow

        self._feature_model = keras.Model(
            self.model.input, [self.model.get_layer(FEATURE_LAYER).output, self.model.output])

    def extract(self, batch):
        """(features, probabilities) for a preprocessed (N, 224, 224, 3) batch."""
        try:
            features, probs = self._feature_model(batch, training=False)
        except Exception as exc:  # noqa: BLE001
            raise PredictionError(f"Model inference failed: {exc}") from exc
        return np.asarray(features), self._check_probs(np.asarray(probs), len(batch))

    def predict_probabilities(self, batch):
        """Run the model on a preprocessed (N, 224, 224, 3) batch."""
        try:
            probs = np.asarray(self.model(batch, training=False))
        except Exception as exc:  # noqa: BLE001
            raise PredictionError(f"Model inference failed: {exc}") from exc
        return self._check_probs(probs, len(batch))

    def _check_probs(self, probs, n):
        if probs.shape != (n, len(self.class_names)) or not np.isfinite(probs).all():
            raise PredictionError(f"Model returned invalid output with shape {probs.shape}")
        return probs

    def predict(self, source):
        """Validate, preprocess and classify one image.

        Raises ImageValidationError for bad input and PredictionError if the
        model fails; both carry a message suitable for showing to the user.
        """
        batch = preprocess_image(source)
        start = time.perf_counter()
        features, probs = self.extract(batch)
        scope_score = (float(self.scope_checker.scores(features)[0])
                       if self.scope_checker else None)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return self._build_prediction(probs[0], elapsed_ms, scope_score)

    def _build_prediction(self, probs, elapsed_ms=0.0, scope_score=None):
        order = np.argsort(probs)[::-1][: self.top_k]
        candidates = []
        for i in order:
            crop, disease, _ = parse_class_name(self.class_names[i])
            candidates.append(Candidate(self.class_names[i], crop, disease, float(probs[i])))

        best = candidates[0]
        crop, disease, healthy = parse_class_name(best.class_name)
        in_scope = scope_score is None or self.scope_checker.is_in_scope(scope_score)
        confident = in_scope and best.probability >= self.confidence_threshold
        if not in_scope:
            message = ("This image does not look like a leaf from the crops this model "
                       "knows (similarity to training leaves "
                       f"{scope_score:.2f} < {self.scope_checker.threshold:.2f}). Upload a "
                       "clear photo of a single leaf of a supported crop.")
        elif confident:
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
            top_k=candidates, inference_ms=round(elapsed_ms, 1), in_scope=in_scope,
            scope_score=None if scope_score is None else round(scope_score, 4))


__all__ = ["DiseasePredictor", "Prediction", "Candidate", "PredictionError",
           "ImageValidationError", "ModelLoadError", "parse_class_name"]
