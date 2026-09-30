import pytest

from src.preprocessing import config, splits

MODEL = config.ROOT / "models" / "resnet50" / "best_resnet50.keras"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="trained model not available")


@pytest.fixture(scope="module")
def analyzer():
    from src.integration import CropCareAnalyzer

    return CropCareAnalyzer()


@pytest.fixture(scope="module")
def leaf():
    row = splits.load_split("test").iloc[0]
    return config.ROOT / row["path"], row["label"]


def test_confident_prediction_gets_matching_advice(analyzer, leaf):
    path, label = leaf
    result = analyzer.analyze(path)
    assert result.ok
    assert result.prediction.class_name == label
    assert result.recommendation.disease == result.prediction.class_name
    assert result.recommendation.fertilizer and result.recommendation.treatment


def test_low_confidence_withholds_advice(analyzer, leaf):
    from src.disease_detection import DiseasePredictor
    from src.integration import CropCareAnalyzer

    unsure = DiseasePredictor(confidence_threshold=1.0)
    result = CropCareAnalyzer(predictor=unsure).analyze(leaf[0])
    assert result.status == "low_confidence" and result.recommendation is None
    opt_in = CropCareAnalyzer(predictor=unsure, advise_when_uncertain=True).analyze(leaf[0])
    assert opt_in.status == "low_confidence" and opt_in.recommendation.found


def test_invalid_image_and_missing_advice(analyzer, leaf, tmp_path):
    from src.integration import CropCareAnalyzer

    assert analyzer.analyze(b"not an image").status == "invalid_image"
    no_csv = CropCareAnalyzer(predictor=analyzer.predictor,
                              recommendations_path=tmp_path / "missing.csv")
    result = no_csv.analyze(leaf[0])
    assert result.status == "recommendation_unavailable" and result.prediction is not None


def test_format_report_contains_all_sections(analyzer, leaf):
    from src.integration import format_report

    text = format_report(analyzer.analyze(leaf[0]))
    for heading in ("Prediction:", "Fertilizer:", "Treatment:", "Eco-friendly treatment:",
                    "Chemical treatment:", "Disclaimer:"):
        assert heading in text
