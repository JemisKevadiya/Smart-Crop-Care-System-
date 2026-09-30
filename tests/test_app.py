"""Headless UI tests for app.py using Streamlit's AppTest (real uploads, real model)."""

import io

import pytest
from PIL import Image

from src.preprocessing import config, splits

APP = str(config.ROOT / "app.py")
MODEL = config.ROOT / "models" / "resnet50" / "best_resnet50.keras"
# A held-out test image the model is unsure about (confidence ~0.40; see
# artifacts/reports/integration_test.json).
LOW_CONFIDENCE_IMAGE = ("Dataset/raw/train/Raspberry___healthy/"
                        "7f771790-da66-43db-a051-5900affecc6e___Mary_HL 9261.JPG")

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="trained model not available")


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=120)


def _test_image(label):
    test = splits.load_split("test")
    path = config.ROOT / test[test["label"] == label]["path"].iloc[0]
    return path.name, path.read_bytes()


def _texts(at):
    return " ".join(m.value for m in at.markdown)


def _subheaders(at):
    return [s.value for s in at.subheader]


def test_initial_page():
    at = _app().run()
    assert not at.exception
    hero = at.get("html")[0].value
    assert "Smart Crop Care" in hero and "data:image/jpeg;base64," in hero
    assert at.file_uploader[0].label == "Upload leaf image"
    assert not at.error and "Predicted disease" not in _subheaders(at)


def test_diseased_leaf_shows_prediction_and_advice():
    at = _app().run()
    name, data = _test_image("Tomato___Late_blight")
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    assert not at.exception and not at.error
    heads = _subheaders(at)
    for section in ("Uploaded image", "Predicted disease", "Confidence",
                    "Fertilizer recommendation", "Treatment recommendation"):
        assert section in heads
    text = _texts(at)
    assert "Late blight" in text and "Tomato" in text
    assert "mancozeb" in text.lower()          # chemical option from the CSV
    confidence = next(m for m in at.metric if m.label == "Model confidence")
    assert confidence.value.endswith("%")
    assert at.info and "Late blight detected" in at.info[0].value


def test_healthy_leaf():
    at = _app().run()
    name, data = _test_image("Potato___healthy")
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    assert not at.exception
    assert at.success and "looks healthy" in at.success[0].value
    assert "Fertilizer recommendation" in _subheaders(at)


def test_low_confidence_hides_advice_until_opted_in():
    path = config.ROOT / LOW_CONFIDENCE_IMAGE
    at = _app().run()
    at.file_uploader[0].upload(path.name, path.read_bytes(), "image/jpeg").run()
    assert not at.exception
    assert at.warning and "Low confidence" in at.warning[0].value
    assert "Fertilizer recommendation" not in _subheaders(at)

    at.toggle(key="advise_uncertain").set_value(True).run()
    assert not at.exception
    assert "Fertilizer recommendation" in _subheaders(at)


def test_corrupt_file_shows_error():
    at = _app().run()
    at.file_uploader[0].upload("leaf.jpg", b"definitely not a jpeg", "image/jpeg").run()
    assert not at.exception
    assert at.error and "not a valid image" in at.error[0].value
    assert "Predicted disease" not in _subheaders(at)


def test_blank_image_shows_error():
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), "white").save(buf, format="PNG")
    at = _app().run()
    at.file_uploader[0].upload("blank.png", buf.getvalue(), "image/png").run()
    assert not at.exception
    assert at.error and "blank" in at.error[0].value
