"""Leaf image -> disease prediction -> fertilizer/treatment advice.

    analyzer = CropCareAnalyzer()
    result = analyzer.analyze(uploaded_file)
    result.status   # see STATUSES
    result.message  # one sentence for the user
    result.prediction, result.recommendation

Low-confidence policy: when the model is not confident, the result carries
the prediction and its top candidates but no treatment advice, because
spraying for the wrong disease wastes money and can harm the crop. Pass
advise_when_uncertain=True to attach advice for the top candidate anyway
(it stays flagged as low confidence).

Out-of-scope policy: images that do not resemble the training leaves (faces,
scenes, other objects) get status "out_of_scope" and never receive advice.
"""

import time
from dataclasses import asdict, dataclass, field

from src.disease_detection import (
    DiseasePredictor,
    ImageValidationError,
    Prediction,
    PredictionError,
)
from src.fertilizer import Recommendation, get_recommendation
from src.fertilizer.fertilizer_data import DATA_PATH

STATUSES = {
    "ok": "Disease identified and advice found.",
    "low_confidence": "Prediction is uncertain; advice withheld.",
    "out_of_scope": "The image does not look like a leaf of a supported crop.",
    "invalid_image": "The input is not a usable leaf image.",
    "model_error": "The model failed to run.",
    "recommendation_unavailable": "Disease identified, but no advice could be loaded.",
}


@dataclass
class AnalysisResult:
    status: str
    message: str
    prediction: Prediction | None = None
    recommendation: Recommendation | None = None
    timings_ms: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.status == "ok"

    def to_dict(self):
        return asdict(self)


class CropCareAnalyzer:
    """Holds the loaded model; call analyze() once per image.

    Raises ModelLoadError on construction if the model or class names are
    missing, because nothing can be analysed without them.
    """

    def __init__(self, predictor=None, recommendations_path=DATA_PATH,
                 advise_when_uncertain=False):
        self.predictor = predictor or DiseasePredictor()
        self.recommendations_path = recommendations_path
        self.advise_when_uncertain = advise_when_uncertain

    def analyze(self, image):
        """Run the full pipeline on one image (path, bytes, file-like or PIL image)."""
        t0 = time.perf_counter()
        try:
            prediction = self.predictor.predict(image)
        except ImageValidationError as exc:
            return AnalysisResult("invalid_image", str(exc))
        except PredictionError as exc:
            return AnalysisResult("model_error", f"Analysis failed: {exc}")
        t1 = time.perf_counter()
        timings = {"prediction": round((t1 - t0) * 1000, 1)}

        if not prediction.in_scope:
            return AnalysisResult("out_of_scope", prediction.message, prediction,
                                  timings_ms=timings)
        if not prediction.is_confident and not self.advise_when_uncertain:
            return AnalysisResult("low_confidence", prediction.message, prediction,
                                  timings_ms=timings)

        recommendation = get_recommendation(prediction, data_path=self.recommendations_path)
        timings["recommendation"] = round((time.perf_counter() - t1) * 1000, 1)

        if not recommendation.found:
            message = (f"{prediction.message} However, advice is unavailable: "
                       f"{recommendation.message}")
            return AnalysisResult("recommendation_unavailable", message, prediction,
                                  recommendation, timings)
        if not prediction.is_confident:
            return AnalysisResult("low_confidence", prediction.message, prediction,
                                  recommendation, timings)
        return AnalysisResult("ok", prediction.message, prediction, recommendation, timings)


def format_report(result):
    """Plain-text summary of an AnalysisResult (for CLI output and logs)."""
    lines = [f"Status: {result.status}", f"Message: {result.message}"]
    p, r = result.prediction, result.recommendation
    if p:
        lines.append(f"Prediction: {p.crop} - {p.disease} ({p.confidence:.1%} confidence)")
        lines.append("Top candidates: " + "; ".join(
            f"{c.crop} - {c.disease} {c.probability:.1%}" for c in p.top_k))
    if r and r.found:
        lines += [
            f"Category: {r.category}",
            f"Fertilizer: {r.fertilizer}",
            f"Treatment: {r.treatment}",
            f"Eco-friendly treatment: {r.eco_friendly_treatment}",
            f"Chemical treatment: {r.chemical_treatment}",
            f"Notes: {r.notes}",
            f"Disclaimer: {r.disclaimer}",
        ]
    return "\n".join(lines)
