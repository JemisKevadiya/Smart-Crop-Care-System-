"""Agriculture Assistant tests with a stand-in chat model (no network, no API key)."""

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.chatbot import (
    AgricultureAssistant,
    ChatbotError,
    ChatServiceError,
    MissingApiKeyError,
    build_system_prompt,
)
from src.chatbot import chatbot, llm
from src.context import CropContext

def _potato_context():
    ctx = CropContext()
    ctx.set_disease("Potato", "Early blight", 0.95, is_healthy=False, status="ok")
    ctx.set_recommendation(category="Fungal", fertilizer="Fertilize according to a soil test.",
                           treatment="Remove infected leaves; rotate crops.",
                           eco_friendly_treatment="Copper-based fungicide.",
                           chemical_treatment="Mancozeb or chlorothalonil.")
    return ctx


def _weather_context():
    ctx = CropContext()
    ctx.set_weather(
        {"name": "Surat, Gujarat, India", "latitude": 21.1959, "longitude": 72.8302,
         "source": "search"},
        {"temperature_c": 37.4, "humidity_pct": 29.0, "wind_speed_kmh": 3.1,
         "condition": "Clear sky", "source": "Open-Meteo"},
        [{"date": "2026-10-07", "condition": "Light rain showers", "temp_min_c": 24.8,
          "temp_max_c": 37.3, "precipitation_mm": 3.5, "precipitation_probability_pct": 55.0,
          "humidity_mean_pct": 71.0, "wind_max_kmh": 13.5, "et0_mm": 4.7}],
        "Asia/Kolkata")
    return ctx


POTATO = _potato_context().get_context()
WEATHER = _weather_context().get_context()


class RecordingModel:
    """Stands in for ChatGroq: records the messages and returns a fixed answer."""

    def __init__(self, answer="Here is some general guidance.", error=None):
        self.answer, self.error, self.calls = answer, error, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return AIMessage(self.answer)


def ask(question, **kwargs):
    model = RecordingModel()
    answer = AgricultureAssistant(chat_model=model).reply(question, **kwargs)
    return answer, model.calls[0]


# --- 1-4: questions are sent with the right instructions and context ---------------

def test_disease_question_uses_detection_context():
    answer, messages = ask("What should I do next?", context=POTATO)
    assert answer == "Here is some general guidance."
    system, question = messages[0], messages[-1]
    assert isinstance(system, SystemMessage) and isinstance(question, HumanMessage)
    for text in ("Crop: Potato", "Disease: Early blight", "Model confidence: 95%",
                 "Fertilizer recommendation shown: Fertilize according to a soil test.",
                 "Chemical option shown: Mancozeb or chlorothalonil."):
        assert text in system.content
    assert question.content == "What should I do next?"


def test_fertilizer_question_keeps_conversation_history():
    history = [{"role": "user", "content": "My potato leaves have brown rings."},
               {"role": "assistant", "content": "That can be early blight."}]
    _, messages = ask("Which fertilizer should I use?", history=history, context=POTATO)
    assert [type(m) for m in messages] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert messages[1].content == "My potato leaves have brown rings."
    assert "product label" in messages[0].content


def test_crop_care_question_without_context():
    _, messages = ask("How often should I water tomatoes?")
    assert "No crop context is available" in messages[0].content
    assert "Disease detection result: NOT available" in messages[0].content
    assert "Latest disease detection result" not in messages[0].content


def test_weather_question_uses_weather_context():
    _, messages = ask("Can I spray fungicide tomorrow?", context=WEATHER)
    system = messages[0].content
    assert "Location: Surat, Gujarat, India (21.1959, 72.8302)" in system
    assert "Current weather: 37.4 °C" in system and "rain 3.5 mm (55% chance)" in system
    assert "2026-10-07: Light rain showers, 25-37 °C" in system and "ET0 4.7 mm" in system


# --- 5: unrelated questions --------------------------------------------------------

def test_system_prompt_rules():
    prompt = build_system_prompt()
    assert "If a question is not about agriculture" in prompt
    assert "Do not answer the unrelated question" in prompt
    assert "follow the product label" in prompt
    assert "not professional agricultural advice" in prompt
    assert "Do not pretend to be certain" in prompt
    _, messages = ask("Who won the cricket world cup?")
    assert "If a question is not about agriculture" in messages[0].content


# --- 6: API failures and other errors ----------------------------------------------

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _status(cls, code):
    return cls("error", response=httpx.Response(code, request=REQUEST), body=None)


@pytest.mark.parametrize("error, message", [
    (groq.APITimeoutError(request=REQUEST), "took too long"),
    (groq.APIConnectionError(request=REQUEST), "Could not connect"),
    (_status(groq.AuthenticationError, 401), "rejected the API key"),
    (_status(groq.RateLimitError, 429), "request limit"),
    (_status(groq.NotFoundError, 404), "GROQ_MODEL"),
    (_status(groq.InternalServerError, 500), "HTTP 500"),
    (RuntimeError("boom"), "RuntimeError"),
])
def test_api_failures_become_clear_errors(error, message):
    assistant = AgricultureAssistant(chat_model=RecordingModel(error=error))
    with pytest.raises(ChatServiceError, match=message) as info:
        assistant.reply("How do I treat leaf mold?")
    assert str(info.value).startswith("AI assistant is temporarily unavailable.")


def test_empty_answer_and_empty_question():
    with pytest.raises(ChatServiceError, match="empty answer"):
        AgricultureAssistant(chat_model=RecordingModel(answer="  ")).reply("Hello?")
    with pytest.raises(ChatbotError, match="type a question"):
        AgricultureAssistant(chat_model=RecordingModel()).reply("   ")


def test_missing_api_key(monkeypatch):
    monkeypatch.setattr(llm, "get_api_key", lambda: None)
    with pytest.raises(MissingApiKeyError, match="GROQ_API_KEY is not set"):
        AgricultureAssistant()


def test_long_history_is_trimmed():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
               for i in range(30)]
    _, messages = ask("Next?", history=history)
    assert len(messages) == 1 + chatbot.MAX_HISTORY_MESSAGES + 1
    assert messages[1].content == f"m{30 - chatbot.MAX_HISTORY_MESSAGES}"


# --- Context availability, dates and secrets ----------------------------------------

def test_availability_list_reflects_the_context():
    from datetime import date

    full = dict(POTATO, weather=WEATHER["weather"])
    prompt = build_system_prompt(full, today=date(2026, 10, 7))
    assert "Today's date: Wednesday 07 October 2026" in prompt
    for part in ("Disease detection result", "Fertilizer and treatment recommendation",
                 "Current weather", "Weather forecast"):
        assert f"{part}: available" in prompt
    assert "Model confidence: 95%" in prompt and "Location: Surat" in prompt

    weather_only = build_system_prompt(WEATHER)
    assert "Disease detection result: NOT available" in weather_only
    assert "Current weather: available" in weather_only
    assert "Crop: " not in weather_only                     # no disease is invented

    withheld = dict(POTATO, recommendation={}, prediction_status="low_confidence")
    prompt = build_system_prompt(withheld)
    assert "Fertilizer and treatment recommendation: NOT available" in prompt
    assert "withheld" in prompt


def test_prompt_rules_for_context():
    prompt = build_system_prompt()
    assert "Do not invent missing context" in prompt
    assert "Never say the plant definitely has the disease" in prompt
    assert "Based on the current image analysis" in prompt
    assert "not a replacement for a qualified agricultural professional" in prompt
    assert "local agricultural guidance" in prompt


def test_api_keys_never_reach_the_prompt(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_groq_value")
    monkeypatch.setenv("OPENWEATHER_API_KEY", "secret_openweather_value")
    full = dict(POTATO, weather=WEATHER["weather"])
    _, messages = ask("What should I do next?", context=full)
    text = " ".join(m.content for m in messages)
    assert "gsk_secret_groq_value" not in text and "secret_openweather_value" not in text
    assert "API_KEY" not in text


def test_long_earlier_messages_are_shortened():
    history = [{"role": "assistant", "content": "x" * 10_000}]
    _, messages = ask("Next?", history=history)
    assert len(messages[1].content) == chatbot.MAX_HISTORY_CHARS


# --- Answer language ------------------------------------------------------------------

@pytest.mark.parametrize("language, script", [("Hindi", "Devanagari"), ("Gujarati", "Gujarati")])
def test_answer_language_is_in_the_prompt(language, script):
    _, messages = ask("What should I do next?", context=POTATO, language=language)
    system = messages[0].content
    assert f"write your whole answer in {language}" in system and f"{script} script" in system
    assert "Disease: Early blight" in system          # the context itself stays in English


def test_english_is_the_default_and_unknown_languages_are_rejected():
    _, messages = ask("How do I water tomatoes?")
    assert "write your whole answer in English" in messages[0].content
    with pytest.raises(ValueError, match="unsupported language"):
        build_system_prompt(language="French")
