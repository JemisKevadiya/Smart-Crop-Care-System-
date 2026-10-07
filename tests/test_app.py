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


@pytest.fixture(autouse=True)
def news_feeds(monkeypatch):
    """App tests never call the real news feeds: they read a recorded-style test feed.

    Set news_feeds["mode"] to "ok", "empty" or "down"; news_feeds["calls"] counts fetches.
    """
    from src.news import news_service
    from src.ui import news_panel
    from tests.test_news import AGRI, AGRI_FEED, NOW, rss

    state = {"mode": "ok", "calls": 0}

    def fetch_feed(feed, timeout=None):
        state["calls"] += 1
        if state["mode"] == "down":
            raise news_service.NewsError("Could not connect to Agri Times.")
        return news_service.parse_feed(AGRI_FEED if state["mode"] == "ok" else rss(), feed)

    real = news_service.get_farmer_news
    monkeypatch.setattr(news_service, "fetch_feed", fetch_feed)
    monkeypatch.setattr(news_panel, "get_farmer_news", lambda: real(feeds=(AGRI,), now=NOW))
    news_panel.fetch_news.clear()
    yield state
    news_panel.fetch_news.clear()


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


def _dashboard():
    """The app opened on the Disease detection page (Home is the default page)."""
    at = _app().run()
    at.switch_page("app_pages/disease_detection.py").run()
    return at


def _has_advice(at):
    return "**Fertilizer:**" in _texts(at)


DASHBOARD_SECTIONS = ("Disease detection", "Recommended crop care", "Weather context",
                      "Live Farmer News", "Smart Crop Care summary",
                      "Smart Agriculture Assistant")


def test_home_page_and_try_now():
    at = _app().run()
    assert not at.exception and not at.error
    hero = at.get("html")[0].value
    assert "Smart Crop Care" in hero and "data:image/jpeg;base64," in hero
    assert "How it works" in _subheaders(at) and not at.file_uploader
    at.button(key="home_try_now").click().run()
    assert not at.exception
    assert at.file_uploader[0].label == "Upload leaf image"
    assert "Disease detection" not in _subheaders(at)          # nothing uploaded yet



def test_home_location_is_shared_with_the_other_pages(monkeypatch):
    import requests
    import streamlit as st
    from tests.test_weather import fake_api

    st.cache_data.clear()
    monkeypatch.setattr(requests, "get", fake_api())
    at = _app().run()
    assert at.expander[0].label == "Set your location"
    at.text_input(key="home_query").input("Surat")
    next(b for b in at.button if b.label == "Search").click().run()
    assert not at.exception and not at.error
    assert "Location: **Surat, Gujarat, India**" in _texts(at)
    assert at.session_state.crop_context.weather["location"]["name"] == "Surat, Gujarat, India"

    at.switch_page("app_pages/weather.py").run()               # no search needed there
    assert not at.exception and "Surat, Gujarat, India" in _subheaders(at)

    at.segmented_control(key="weather_mode").set_value("Coordinates").run()
    at.number_input(key="weather_lat").set_value(18.52)
    at.number_input(key="weather_lon").set_value(73.86)
    next(b for b in at.button if b.label == "Show weather").click().run()
    assert not at.exception and "18.5200, 73.8600" in _subheaders(at)

    at.switch_page("app_pages/home.py").run()                  # Home keeps the newer choice
    assert not at.exception and "Location: **18.5200, 73.8600**" in _texts(at)

def test_diseased_leaf_shows_prediction_and_advice():
    at = _dashboard()
    name, data = _test_image("Tomato___Late_blight")
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    assert not at.exception and not at.error
    heads = _subheaders(at)
    for section in DASHBOARD_SECTIONS:
        assert section in heads
    assert _has_advice(at) and "**Treatment:**" in _texts(at)
    text = _texts(at)
    assert "Late blight" in text and "Tomato" in text
    assert "mancozeb" in text.lower()          # chemical option from the CSV
    confidence = next(m for m in at.metric if m.label == "Confidence")
    assert confidence.value.endswith("%")
    assert at.info and "Late blight detected" in at.info[0].value


def test_healthy_leaf():
    at = _dashboard()
    name, data = _test_image("Potato___healthy")
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    assert not at.exception
    assert at.success and "looks healthy" in at.success[0].value
    assert _has_advice(at)


def test_low_confidence_hides_advice_until_opted_in():
    path = config.ROOT / LOW_CONFIDENCE_IMAGE
    at = _dashboard()
    at.file_uploader[0].upload(path.name, path.read_bytes(), "image/jpeg").run()
    assert not at.exception
    assert at.warning and "Low confidence" in at.warning[0].value
    assert not _has_advice(at)

    at.toggle(key="advise_uncertain").set_value(True).run()
    assert not at.exception
    assert _has_advice(at)


def test_corrupt_file_shows_error():
    at = _dashboard()
    at.file_uploader[0].upload("leaf.jpg", b"definitely not a jpeg", "image/jpeg").run()
    assert not at.exception
    assert at.error and "not a valid image" in at.error[0].value
    assert "Disease detection" not in _subheaders(at)


def test_blank_image_shows_error():
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), "white").save(buf, format="PNG")
    at = _dashboard()
    at.file_uploader[0].upload("blank.png", buf.getvalue(), "image/png").run()
    assert not at.exception
    assert at.error and "blank" in at.error[0].value


def test_face_photo_is_rejected_as_not_a_leaf():
    from matplotlib import cbook

    face = cbook.get_sample_data("grace_hopper.jpg").read()
    at = _dashboard()
    at.file_uploader[0].upload("face.jpg", face, "image/jpeg").run()
    assert not at.exception
    heads = _subheaders(at)
    assert "Not a supported leaf" in heads
    assert "Disease detection" not in heads and not _has_advice(at)
    assert "does not look like a leaf" in at.warning[0].value


# --- Weather page -------------------------------------------------------------------

def _weather_page(monkeypatch, fake_get):
    import requests
    import streamlit as st

    st.cache_data.clear()
    monkeypatch.setattr(requests, "get", fake_get)
    at = _app().run()
    at.switch_page("app_pages/weather.py").run()
    return at


def _search(at, query):
    at.text_input(key="weather_query").input(query)
    next(b for b in at.button if b.label == "Search").click().run()
    return at


def test_weather_page_shows_current_weather_and_forecast(monkeypatch):
    from tests.test_weather import fake_api

    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-key")
    at = _weather_page(monkeypatch, fake_api())
    assert not at.exception
    assert "Search for your city" in at.info[0].value
    _search(at, "Surat")
    assert not at.exception and not at.error
    assert at.selectbox(key="weather_match").value.label == "Surat, Gujarat, India"
    heads = _subheaders(at)
    assert "Surat, Gujarat, India" in heads and "Current weather" in heads
    assert "7-day forecast" in heads
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Temperature"] == "30.2 °C" and metrics["Humidity"] == "80%"
    assert metrics["Wind"] == "9.0 km/h" and metrics["Condition"] == "Light rain"
    assert len(at.dataframe[0].value) == 2


def test_weather_invalid_location_shows_error(monkeypatch):
    from tests.test_weather import fake_api

    at = _search(_weather_page(monkeypatch, fake_api(geocode={})), "Xyzzyqwv")
    assert not at.exception
    assert "No place called" in at.error[0].value
    assert "Current weather" not in _subheaders(at)


def test_disease_detection_still_works_after_weather_failure(monkeypatch):
    import requests

    def offline(*args, **kwargs):
        raise requests.ConnectionError("network down")

    at = _search(_weather_page(monkeypatch, offline), "Surat")
    assert not at.exception and "Could not connect" in at.error[0].value

    at.switch_page("app_pages/disease_detection.py").run()
    name, data = _test_image("Tomato___Late_blight")
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    assert not at.exception and not at.error
    assert "Disease detection" in _subheaders(at)
    assert _has_advice(at)


# --- Agriculture Assistant page -----------------------------------------------------

def test_assistant_answers_with_context_from_disease_page(monkeypatch):
    from src.chatbot import llm
    from tests.test_chatbot import RecordingModel

    model = RecordingModel(answer="Remove the infected leaves first.")
    monkeypatch.setattr(llm, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(llm, "create_chat_model", lambda api_key, **kw: model)

    at = _dashboard()
    name, data = _test_image("Tomato___Late_blight")
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    at.switch_page("app_pages/assistant.py").run()
    assert not at.exception
    assert "Late blight" in at.get("text")[0].value        # context panel
    at.chat_input[0].set_value("What should I do next?").run()
    assert not at.exception and not at.error
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert "Remove the infected leaves first." in at.chat_message[1].markdown[0].value
    system = model.calls[0][0].content
    assert "Crop: Tomato" in system and "Disease: Late blight" in system
    stored = str(at.session_state.to_dict())        # the API key is never kept in state
    assert "test-key" not in stored and "api_key" not in stored.lower()


def test_assistant_answers_in_the_chosen_language(monkeypatch):
    from src.chatbot import llm
    from tests.test_chatbot import RecordingModel

    model = RecordingModel(answer="पत्तियाँ हटाएँ।")
    monkeypatch.setattr(llm, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(llm, "create_chat_model", lambda api_key, **kw: model)
    at = _app().run()
    at.switch_page("app_pages/assistant.py").run()
    assert at.segmented_control(key="chat_language").value == "English"
    at.segmented_control(key="chat_language").set_value("Hindi").run()
    at.chat_input[0].set_value("What should I do next?").run()
    assert not at.exception and not at.error
    assert "पत्तियाँ हटाएँ।" in at.chat_message[1].markdown[0].value
    assert "write your whole answer in Hindi" in model.calls[0][0].content


def test_assistant_without_key_shows_error_and_disease_detection_still_works(monkeypatch):
    from src.chatbot import llm

    monkeypatch.setattr(llm, "get_api_key", lambda: None)
    at = _app().run()
    at.switch_page("app_pages/assistant.py").run()
    at.chat_input[0].set_value("How do I fertilize potatoes?").run()
    assert not at.exception
    assert "GROQ_API_KEY is not set" in at.error[0].value
    assert at.session_state.chat_messages == []

    at.switch_page("app_pages/disease_detection.py").run()
    name, data = _test_image("Potato___Early_blight")
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    assert not at.exception and "Disease detection" in _subheaders(at)


# --- Shared crop context ------------------------------------------------------------

def test_crop_context_persists_across_reruns_and_pages(monkeypatch):
    import requests
    import streamlit as st

    from tests.test_weather import fake_api

    st.cache_data.clear()
    monkeypatch.setattr(requests, "get", fake_api())
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    from src.weather import weather_service
    monkeypatch.setattr(weather_service, "get_setting", lambda name, env_file=None: None)

    at = _dashboard()
    name, data = _test_image("Potato___Early_blight")
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    disease = at.session_state["crop_context"].get_context()
    assert disease["crop"] == "Potato" and disease["disease"] is not None
    assert 0 < disease["confidence"] <= 1

    at.switch_page("app_pages/weather.py").run()
    _search(at, "Surat")
    for _ in range(3):                                    # plain reruns keep everything
        at.run()
    at.switch_page("app_pages/assistant.py").run()
    ctx = at.session_state["crop_context"].get_context()
    assert (ctx["crop"], ctx["disease"], ctx["confidence"]) == \
        (disease["crop"], disease["disease"], disease["confidence"])
    assert ctx["recommendation"] == disease["recommendation"]
    assert ctx["weather"]["location"]["name"] == "Surat, Gujarat, India"
    assert len(ctx["weather"]["forecast"]) == 2
    panel = at.get("text")[0].value
    assert "Crop: Potato" in panel and "Location: Surat, Gujarat, India" in panel

    at.button(key="context_clear").click().run()
    cleared = at.session_state["crop_context"]   # news is reloaded at once from the cache
    assert not cleared.has_disease and not cleared.recommendation and not cleared.has_weather
    panel = at.get("text")[0].value                   # only the (reloaded) news is left
    assert "Crop: Potato" not in panel and "Location:" not in panel
    assert "Recent agriculture news" in panel


# --- Combined dashboard: disease + fertilizer + weather -----------------------------

def _with_location(monkeypatch, fake_get):
    """App session whose crop context already has a location (set on the Weather page)."""
    at = _weather_page(monkeypatch, fake_get)
    _search(at, "Surat")
    at.switch_page("app_pages/disease_detection.py").run()
    return at


def _upload(at, label="Potato___Early_blight"):
    name, data = _test_image(label)
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    return at


def test_combined_dashboard_shows_disease_advice_and_weather(monkeypatch):
    from tests.test_weather import fake_api

    at = _upload(_with_location(monkeypatch, fake_api()))
    assert not at.exception and not at.error
    heads = _subheaders(at)
    assert [h for h in heads if h in DASHBOARD_SECTIONS] == list(DASHBOARD_SECTIONS)
    assert _has_advice(at)
    metrics = {m.label: m.value for m in at.metric}
    assert {"Temperature", "Humidity", "Wind", "Rain today"} <= set(metrics)
    assert metrics["Rain today"] == "3.5 mm"
    assert any("does not change the disease prediction" in c.value for c in at.caption)

    ctx = at.session_state["crop_context"].get_context()
    assert ctx["crop"] == "Potato" and ctx["confidence"] > 0
    assert ctx["recommendation"]["fertilizer"] and ctx["recommendation"]["treatment"]
    assert ctx["weather"]["location"]["name"] == "Surat, Gujarat, India"
    assert ctx["weather"]["forecast"][0]["precipitation_mm"] == 3.5


def test_weather_failure_keeps_disease_and_advice(monkeypatch):
    import requests
    import streamlit as st

    from tests.test_weather import fake_api

    at = _with_location(monkeypatch, fake_api())

    def offline(*args, **kwargs):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(requests, "get", offline)
    st.cache_data.clear()
    _upload(at)
    assert not at.exception and not at.error
    heads = _subheaders(at)
    assert "Disease detection" in heads and _has_advice(at)
    assert any("Weather information is currently unavailable" in w.value for w in at.warning)


def test_dashboard_without_location_points_to_weather_page():
    at = _upload(_dashboard())
    assert not at.exception and not at.error
    assert "Weather context" in _subheaders(at)
    assert any("Set your location on the Weather page" in i.value for i in at.info)


def test_fertilizer_failure_still_shows_prediction_and_weather(monkeypatch):
    import streamlit as st

    from src.fertilizer import Recommendation
    from src.integration import analyzer as analyzer_module
    from tests.test_weather import fake_api

    at = _with_location(monkeypatch, fake_api())
    monkeypatch.setattr(analyzer_module, "get_recommendation", lambda *a, **k: Recommendation(
        status="data_error", query="x", message="The recommendation file could not be read."))
    st.cache_data.clear()
    _upload(at)
    assert not at.exception
    heads = _subheaders(at)
    assert "Disease detection" in heads and "Weather context" in heads
    assert not _has_advice(at)
    assert any("Crop-care advice is unavailable" in w.value and "could not be read" in w.value
               for w in at.warning)
    assert at.session_state["crop_context"].get_context()["recommendation"] == {}


# --- Final integrated flow ----------------------------------------------------------

def test_full_flow_home_to_assistant(monkeypatch, news_feeds):
    """Home -> Try now -> upload -> disease, advice, weather, news, summary -> dashboard chat."""
    from src.chatbot import llm
    from tests.test_chatbot import RecordingModel
    from tests.test_weather import fake_api

    model = RecordingModel(answer="Based on the current image analysis ...")
    monkeypatch.setattr(llm, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(llm, "create_chat_model", lambda api_key, **kw: model)

    at = _weather_page(monkeypatch, fake_api())               # location first
    _search(at, "Surat")
    at.switch_page("app_pages/home.py").run()
    at.button(key="home_try_now").click().run()
    assert at.file_uploader[0].label == "Upload leaf image"     # Try now opened the page
    # AppTest keeps rendering its tracked page on later runs, so follow st.switch_page.
    at.switch_page("app_pages/disease_detection.py").run()
    _upload(at)
    assert not at.exception and not at.error
    heads = _subheaders(at)
    assert [h for h in heads if h in DASHBOARD_SECTIONS] == list(DASHBOARD_SECTIONS)

    summary = _texts(at)
    assert "**Detected disease:** Potato - Early blight (" in summary
    assert "**Fertilizer:**" in summary and "**Treatment:**" in summary
    assert ("**Weather in Surat, Gujarat, India:** 30.2 °C, 80% humidity, light rain, "
            "rain today 3.5 mm") in summary
    assert "**Latest agriculture update:** Centre hikes MSP for rabi crops" in summary

    # 5. The assistant is part of the dashboard and gets the whole crop context.
    at.chat_input(key="dashboard_input").set_value("What should I do now?").run()
    assert not at.exception and not at.error
    system = model.calls[0][0].content
    for part in ("Disease detection result: available", "Crop: Potato",
                 "Fertilizer and treatment recommendation: available",
                 "Current weather: available", "Live agriculture news: available"):
        assert part in system
    assert [m.name for m in at.chat_message] == ["user", "assistant"]

    at.run()                                                   # a rerun keeps everything
    ctx = at.session_state["crop_context"].get_context()
    assert ctx["crop"] == "Potato" and ctx["weather"]["location"]["name"].startswith("Surat")
    assert ctx["recommendation"]["treatment"] and ctx["news"]["items"]
    assert len(at.session_state.chat_messages) == 2

    # The Assistant page continues the same conversation with the same context.
    at.switch_page("app_pages/assistant.py").run()
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    panel = at.get("text")[0].value
    assert "Crop: Potato" in panel and "Location: Surat, Gujarat, India" in panel

    # Back on the dashboard, the last leaf and its result are shown again.
    at.switch_page("app_pages/disease_detection.py").run()
    assert not at.exception and "Disease detection" in _subheaders(at)
    assert any("Showing your last analysed leaf" in c.value for c in at.caption)
    at.switch_page("app_pages/home.py").run()
    assert "Latest result: **Potato" in _texts(at)


def test_start_over_clears_the_kept_leaf():
    at = _upload(_dashboard())
    at.switch_page("app_pages/home.py").run()
    at.switch_page("app_pages/disease_detection.py").run()
    assert "Disease detection" in _subheaders(at)
    at.button(key="leaf_reset").click().run()
    assert not at.exception and "Disease detection" not in _subheaders(at)
    assert at.session_state["crop_context"].disease is None


def test_chatbot_failure_leaves_results_intact(monkeypatch):
    import groq
    import httpx

    from src.chatbot import llm
    from tests.test_chatbot import RecordingModel

    request = httpx.Request("POST", "https://api.groq.com")
    model = RecordingModel(error=groq.APIConnectionError(request=request))
    monkeypatch.setattr(llm, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(llm, "create_chat_model", lambda api_key, **kw: model)

    at = _upload(_dashboard())
    assert _has_advice(at)
    at.chat_input(key="dashboard_input").set_value("What should I do next?").run()
    assert not at.exception
    assert at.error[0].value.startswith("AI assistant is temporarily unavailable.")
    assert "Disease detection" in _subheaders(at) and _has_advice(at)
    assert at.session_state["crop_context"].get_context()["disease"] is not None


def test_weather_news_and_chatbot_all_failing_together(monkeypatch, news_feeds):
    """Every external service down at once: disease and advice still work."""
    import requests

    from src.chatbot import llm
    from tests.test_weather import fake_api

    offline = requests.ConnectionError("offline")
    at = _weather_page(monkeypatch, fake_api())
    _search(at, "Surat")                               # location saved while online
    news_feeds["mode"] = "down"
    monkeypatch.setattr(requests, "get", fake_api(daily=offline, om_current=offline,
                                                  owm=offline, geocode=offline))
    import streamlit as st
    st.cache_data.clear()
    monkeypatch.setattr(llm, "get_api_key", lambda: None)

    at.switch_page("app_pages/disease_detection.py").run()
    _upload(at)
    at.chat_input(key="dashboard_input").set_value("What should I do now?").run()
    assert not at.exception
    assert "Disease detection" in _subheaders(at) and _has_advice(at)
    warnings = " ".join(w.value for w in at.warning)
    assert "Weather information is currently unavailable." in warnings
    assert "Live farmer news is temporarily unavailable." in warnings
    assert "GROQ_API_KEY is not set" in at.error[0].value
    summary = _texts(at)
    assert "**Weather:** currently unavailable." in summary
    assert "live farmer news is temporarily unavailable" in summary


# --- Live farmer news -------------------------------------------------------------------

def _news_page():
    at = _app().run()
    at.switch_page("app_pages/news.py").run()
    return at


def _news_titles(at):
    return [m.value.strip("*") for m in at.main.markdown if m.value.startswith("**")]


def test_news_page_shows_titles_sources_dates_and_links(news_feeds):
    at = _news_page()
    assert not at.exception and not at.error and not at.warning
    assert "Live Farmer News" in _subheaders(at)
    assert _news_titles(at) == ["Centre hikes MSP for rabi crops ahead of sowing",
                                "IMD forecasts heavy rain over Gujarat",
                                "Late blight alert for tomato growers in Nashik",
                                "Drone spraying start-ups raise funds"]
    captions = " ".join(c.value for c in at.caption)
    assert "Source: Agri Times · Published: 07 Oct 2026, 10:00 IST" in captions
    links = [b.proto.url for b in at.get("link_button")]
    assert links == ["https://agri.example/msp", "https://agri.example/imd",
                     "https://agri.example/blight", "https://agri.example/drones"]
    assert all(b.proto.label == "Read full article" for b in at.get("link_button"))
    ctx = at.session_state.crop_context.news      # headlines shared with the assistant
    assert ctx["status"] == "ok" and len(ctx["items"]) == 4


def test_news_category_filter_and_search(news_feeds):
    at = _news_page()
    assert at.pills(key="news_page_topic").options == [
        "All", "Market & MSP", "Crop & Disease", "Weather & Agriculture", "Farming Technology"]
    at.pills(key="news_page_topic").set_value("Crop & Disease").run()
    assert _news_titles(at) == ["Late blight alert for tomato growers in Nashik"]
    at.pills(key="news_page_topic").set_value("All").run()
    at.text_input(key="news_page_query").input("monsoon").run()
    assert _news_titles(at) == [] and "No news matches" in at.info[0].value
    at.text_input(key="news_page_query").input("spraying").run()
    assert _news_titles(at) == ["IMD forecasts heavy rain over Gujarat",
                                "Drone spraying start-ups raise funds"]


def test_news_is_cached_until_refresh(news_feeds):
    at = _news_page()
    at.run()
    at.pills(key="news_page_topic").set_value("Market & MSP").run()
    assert news_feeds["calls"] == 1                 # reruns use the cache
    at.button(key="news_page_refresh").click().run()
    assert news_feeds["calls"] == 2 and not at.exception


def test_news_unavailable_and_empty_messages(news_feeds):
    news_feeds["mode"] = "down"
    at = _news_page()
    assert not at.exception
    assert at.warning[0].value == "Live farmer news is temporarily unavailable. Please try again later."
    assert at.session_state.crop_context.news["status"] == "unavailable"
    news_feeds["mode"] = "empty"
    at.button(key="news_page_refresh").click().run()
    assert at.info[0].value == "No recent agriculture news available."
    assert not at.get("link_button")                # no made-up fallback news


def test_dashboard_order_and_crop_news_first(news_feeds):
    at = _dashboard()
    name, data = _test_image("Tomato___Late_blight")
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    assert not at.exception and not at.error
    heads = _subheaders(at)
    order = ["Disease detection", "Recommended crop care", "Weather context",
             "Live Farmer News", "Smart Crop Care summary", "Smart Agriculture Assistant"]
    assert [h for h in heads if h in order] == order
    # The detected crop (Tomato) comes from the model's class name; its news is first.
    assert _news_titles(at)[0] == "Late blight alert for tomato growers in Nashik"
    assert len(at.get("link_button")) == 3          # dashboard shows 3, then "Show more"
    assert any("News mentioning **Tomato**" in c.value for c in at.caption)
    assert "**Latest agriculture update:** Late blight alert" in _texts(at)        # summary line
    assert _has_advice(at)                         # news never replaces the advice


def test_news_failure_keeps_disease_advice_and_weather(news_feeds):
    news_feeds["mode"] = "down"
    at = _dashboard()
    name, data = _test_image("Potato___Early_blight")
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    assert not at.exception and not at.error
    assert "Disease detection" in _subheaders(at) and _has_advice(at)
    assert any("Live farmer news is temporarily unavailable" in w.value for w in at.warning)
    assert at.session_state.crop_context.disease == "Early blight"


def test_assistant_uses_only_the_loaded_news(news_feeds, monkeypatch):
    from src.chatbot import llm
    from tests.test_chatbot import RecordingModel

    model = RecordingModel(answer="The latest update is about MSP (Agri Times).")
    monkeypatch.setattr(llm, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(llm, "create_chat_model", lambda api_key, **kw: model)
    at = _app().run()
    at.switch_page("app_pages/assistant.py").run()     # news is loaded for the assistant
    at.chat_input[0].set_value("What are the latest agriculture updates?").run()
    assert not at.exception and not at.error
    system = model.calls[0][0].content
    assert "Live agriculture news: available (4 headlines)" in system
    assert "Centre hikes MSP for rabi crops ahead of sowing (Agri Times, 2026-10-07)" in system

    news_feeds["mode"] = "down"
    from src.ui import news_panel
    news_panel.fetch_news.clear()
    at.chat_input[0].set_value("Any crop disease news?").run()
    system = model.calls[1][0].content
    assert "live news is currently unavailable" in system and "Centre hikes MSP" not in system
