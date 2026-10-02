"""Out-of-scope (non-leaf) image detection."""

import numpy as np
import pytest
from PIL import Image

from src.disease_detection.scope_check import REFERENCE_PATH, LeafScopeChecker, knn_similarity
from src.preprocessing import config, splits

MODEL = config.ROOT / "models" / "resnet50" / "best_resnet50.keras"
needs_model = pytest.mark.skipif(not (MODEL.exists() and REFERENCE_PATH.exists()),
                                 reason="trained model or leaf reference not available")


def test_knn_similarity_scores_near_and_far_vectors():
    reference = np.eye(4, dtype=np.float32)
    scores = knn_similarity(np.array([[2.0, 0, 0, 0], [1.0, 1, 1, 1]]), reference, k=1)
    assert scores[0] == pytest.approx(1.0) and scores[1] == pytest.approx(0.5)
    checker = LeafScopeChecker(reference, threshold=0.8, k=1)
    assert checker.is_in_scope(0.9) and not checker.is_in_scope(0.5)


@pytest.mark.skipif(not REFERENCE_PATH.exists(), reason="leaf reference not built")
def test_saved_reference_is_consistent():
    checker = LeafScopeChecker.from_file()
    assert checker.reference.shape[1] == 2048 and len(checker.reference) >= 38
    assert 0 < checker.threshold < 1


@pytest.fixture(scope="module")
def analyzer():
    from src.integration import CropCareAnalyzer

    return CropCareAnalyzer()


@needs_model
def test_real_leaf_is_in_scope(analyzer):
    leaf = config.ROOT / splits.load_split("test")["path"].iloc[0]
    result = analyzer.analyze(leaf)
    assert result.prediction.in_scope and result.prediction.scope_score >= 0.6
    assert result.status == "ok"


@needs_model
@pytest.mark.parametrize("kind", ["face", "noise"])
def test_non_leaf_images_are_out_of_scope(analyzer, kind):
    if kind == "face":
        from matplotlib import cbook

        image = Image.open(cbook.get_sample_data("grace_hopper.jpg"))
    else:
        rng = np.random.default_rng(1)
        image = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    result = analyzer.analyze(image)
    assert result.status == "out_of_scope" and result.recommendation is None
    assert not result.prediction.in_scope and not result.prediction.is_confident


@needs_model
def test_out_of_scope_never_gets_advice_even_when_opted_in(analyzer):
    from src.integration import CropCareAnalyzer

    noise = Image.fromarray(np.random.default_rng(2).integers(0, 256, (224, 224, 3), dtype=np.uint8))
    opt_in = CropCareAnalyzer(predictor=analyzer.predictor, advise_when_uncertain=True)
    result = opt_in.analyze(noise)
    assert result.status == "out_of_scope" and result.recommendation is None


@needs_model
def test_check_is_skipped_without_reference(tmp_path):
    from src.disease_detection import DiseasePredictor

    predictor = DiseasePredictor(scope_reference_path=tmp_path / "missing.npz")
    leaf = config.ROOT / splits.load_split("test")["path"].iloc[0]
    prediction = predictor.predict(leaf)
    assert predictor.scope_checker is None and prediction.scope_score is None
    assert prediction.in_scope
