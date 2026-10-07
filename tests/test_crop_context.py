"""Shared CropContext: setters, clearing, and filling it from the real modules."""

import json

import numpy as np
import pytest
import requests
from PIL import Image

from src.context import CropContext
from src.preprocessing import config, splits

MODEL = config.ROOT / "models" / "resnet50" / "best_resnet50.keras"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="trained model not available")


def test_new_context_is_empty():
    ctx = CropContext().get_context()
    assert ctx["crop"] is None and ctx["disease"] is None and ctx["confidence"] is None
    assert ctx["recommendation"] == {} and ctx["weather"] == {}
    assert CropContext().is_empty


# 1. Set disease  2. Set confidence --------------------------------------------------

def test_set_disease_and_confidence():
    ctx = CropContext()
    ctx.set_disease("Tomato", "Late blight", 0.93, is_healthy=False, status="ok")
    assert (ctx.crop, ctx.disease, ctx.confidence) == ("Tomato", "Late blight", 0.93)
    ctx.set_confidence(0.71)
    assert ctx.confidence == 0.71 and "disease" in ctx.updated
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            ctx.set_confidence(bad)


# 3. Set recommendation -------------------------------------------------------------

def test_set_recommendation():
    ctx = CropContext()
    ctx.set_recommendation(fertilizer="Soil-test based.", treatment="Remove leaves.",
                           chemical_treatment="", notes=None)
    assert ctx.recommendation == {"fertilizer": "Soil-test based.", "treatment": "Remove leaves."}
    with pytest.raises(ValueError, match="unknown"):
        ctx.set_recommendation(dose="10 kg")


# 4. Set weather --------------------------------------------------------------------

def test_set_weather():
    ctx = CropContext()
    ctx.set_weather({"name": "Surat", "latitude": 21.2, "longitude": 72.8},
                    {"temperature_c": 30.0}, [{"date": "2026-10-07"}], "Asia/Kolkata")
    assert ctx.weather["location"]["name"] == "Surat" and ctx.has_weather
    assert ctx.weather["forecast"] == [{"date": "2026-10-07"}]


# 5. Retrieve complete context  6. Clear context --------------------------------------

def test_get_context_is_a_plain_copy_and_clear_resets_everything():
    ctx = CropContext()
    ctx.set_disease("Potato", "Early blight", 0.95)
    ctx.set_recommendation(fertilizer="Soil-test based.")
    ctx.set_weather({"name": "Surat", "latitude": 21.2, "longitude": 72.8})
    data = ctx.get_context()
    assert set(data) == {"crop", "disease", "confidence", "is_healthy", "prediction_status",
                         "recommendation", "weather", "news", "updated"}
    json.dumps(data)                                   # plain data only
    data["recommendation"]["fertilizer"] = "changed"   # a copy: the context is unaffected
    assert ctx.recommendation["fertilizer"] == "Soil-test based."

    ctx.clear_disease()
    assert ctx.disease is None and ctx.recommendation == {} and ctx.has_weather
    ctx.clear_context()
    assert ctx.is_empty and ctx.get_context()["updated"] == {}


# Filling the context from the existing modules -------------------------------------

@pytest.fixture(scope="module")
def analyzer():
    from src.integration import CropCareAnalyzer

    return CropCareAnalyzer()


@needs_model
def test_update_from_real_analysis(analyzer):
    test = splits.load_split("test")
    path = config.ROOT / test[test["label"] == "Potato___Early_blight"]["path"].iloc[0]
    result = analyzer.analyze(path)
    ctx = CropContext()
    ctx.update_from_analysis(result)
    assert ctx.crop == result.prediction.crop == "Potato"
    assert ctx.disease == result.prediction.disease
    assert ctx.confidence == result.prediction.confidence
    assert ctx.prediction_status == result.status
    if result.recommendation is not None and result.recommendation.found:
        assert ctx.recommendation["fertilizer"] == result.recommendation.fertilizer
        assert ctx.recommendation["treatment"] == result.recommendation.treatment


@needs_model
def test_rejected_images_clear_the_disease_part(analyzer):
    ctx = CropContext()
    ctx.set_disease("Potato", "Early blight", 0.95)
    ctx.set_weather({"name": "Surat", "latitude": 21.2, "longitude": 72.8})
    noise = Image.fromarray(np.random.default_rng(3).integers(0, 256, (224, 224, 3),
                                                              dtype=np.uint8))
    ctx.update_from_analysis(analyzer.analyze(noise))          # out of scope
    assert ctx.disease is None and ctx.has_weather
    ctx.set_disease("Potato", "Early blight", 0.95)
    ctx.update_from_analysis(analyzer.analyze(b"not an image"))  # invalid
    assert ctx.disease is None


@needs_model
def test_low_confidence_keeps_prediction_but_no_advice(analyzer):
    from src.disease_detection import DiseasePredictor
    from src.integration import CropCareAnalyzer

    unsure = CropCareAnalyzer(predictor=DiseasePredictor(confidence_threshold=1.0))
    path = config.ROOT / splits.load_split("test")["path"].iloc[0]
    ctx = CropContext()
    ctx.set_recommendation(fertilizer="old advice")
    ctx.update_from_analysis(unsure.analyze(path))
    assert ctx.prediction_status == "low_confidence" and ctx.disease is not None
    assert ctx.recommendation == {}


def test_update_from_weather_report(monkeypatch):
    from src.weather import from_coordinates, get_weather_report
    from tests.test_weather import fake_api

    monkeypatch.setattr(requests, "get", fake_api())
    report = get_weather_report(from_coordinates(21.19594, 72.83023, "Surat"), api_key=None)
    ctx = CropContext()
    ctx.update_from_weather(report)
    w = ctx.weather
    assert w["location"]["name"] == "Surat" and w["timezone"] == "Asia/Kolkata"
    assert w["current"]["temperature_c"] == report.current.temperature_c
    assert w["current"]["source"] == "Open-Meteo"
    assert [d["date"] for d in w["forecast"]] == ["2026-10-07", "2026-10-08"]
    assert w["forecast"][0]["precipitation_mm"] == 3.5
    json.dumps(ctx.get_context())


def test_set_location_keeps_data_for_the_same_place():
    ctx = CropContext()
    ctx.set_location("Surat", 21.2, 72.8, "search")
    assert ctx.weather["location"]["name"] == "Surat" and ctx.weather["current"] == {}
    ctx.set_weather({"name": "Surat", "latitude": 21.2, "longitude": 72.8},
                    {"temperature_c": 30.0})
    ctx.set_location("Surat", 21.2, 72.8, "search")                 # same place: data kept
    assert ctx.weather["current"] == {"temperature_c": 30.0}
    ctx.set_location("Pune", 18.5, 73.9, "ip")                      # new place: data reset
    assert ctx.weather["location"]["name"] == "Pune" and ctx.weather["current"] == {}
