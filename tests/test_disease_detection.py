import io

import numpy as np
import pytest
from PIL import Image

from src.disease_detection.predictor import parse_class_name
from src.disease_detection.preprocessing import (
    ImageValidationError,
    center_square,
    preprocess_image,
    validate_image,
)
from src.preprocessing import config, splits


def _jpeg_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture(scope="module")
def leaf_path():
    return config.ROOT / splits.load_split("test")["path"].iloc[0]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Corn_(maize)___Common_rust_", ("Corn (maize)", "Common rust", False)),
        ("Pepper,_bell___healthy", ("Pepper, bell", "Healthy", True)),
        ("Tomato___Spider_mites Two-spotted_spider_mite",
         ("Tomato", "Spider mites Two-spotted spider mite", False)),
    ],
)
def test_parse_class_name(name, expected):
    assert parse_class_name(name) == expected


@pytest.mark.parametrize(
    "source, message",
    [
        (b"", "empty"),
        (b"not an image", "not a valid image"),
        (12345, "Unsupported input type"),
    ],
)
def test_invalid_inputs_are_rejected(source, message):
    with pytest.raises(ImageValidationError, match=message):
        validate_image(source)


def test_truncated_and_blank_images_rejected(leaf_path):
    data = leaf_path.read_bytes()
    with pytest.raises(ImageValidationError, match="corrupted"):
        validate_image(data[: len(data) // 3])
    with pytest.raises(ImageValidationError, match="blank"):
        validate_image(Image.new("RGB", (256, 256), (90, 140, 60)))
    with pytest.raises(ImageValidationError, match="too small"):
        validate_image(Image.open(leaf_path).resize((20, 20)))


def test_rgba_transparency_becomes_white(leaf_path):
    rgba = Image.open(leaf_path).convert("RGBA")
    rgba.putalpha(0)
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    with pytest.raises(ImageValidationError, match="blank"):  # fully transparent -> white
        validate_image(buf.getvalue())


def test_center_square_crops_to_shorter_side():
    img = Image.new("RGB", (400, 300))
    assert center_square(img).size == (300, 300)
    assert center_square(Image.new("RGB", (256, 256))).size == (256, 256)


def test_preprocess_matches_training_pipeline(leaf_path):
    tf = pytest.importorskip("tensorflow")
    from src.preprocessing import pipeline

    ours = preprocess_image(leaf_path)
    ref = pipeline.preprocess_image_file(leaf_path).numpy()
    assert ours.shape == (1, 224, 224, 3) and ours.dtype == np.float32
    assert np.abs(ours - ref).max() < 1e-4
    # same result whether given a path, bytes, file-like object or PIL image
    data = leaf_path.read_bytes()
    for source in (data, io.BytesIO(data), Image.open(leaf_path)):
        assert np.abs(preprocess_image(source) - ours).max() < 1e-4
    assert tf is not None


@pytest.mark.skipif(not (config.ROOT / "models" / "resnet50" / "best_resnet50.keras").exists(),
                    reason="trained model not available")
def test_predictor_end_to_end(leaf_path):
    from src.disease_detection import DiseasePredictor

    predictor = DiseasePredictor()
    label = splits.load_split("test")["label"].iloc[0]
    result = predictor.predict(leaf_path)
    assert result.class_name == label
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_k) == 3 and result.top_k[0].class_name == result.class_name
    assert result.top_k[0].probability >= result.top_k[1].probability

    strict = DiseasePredictor(confidence_threshold=1.0)
    low = strict.predict(leaf_path)
    assert not low.is_confident and "Low confidence" in low.message
