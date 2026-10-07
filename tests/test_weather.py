"""Weather module tests with fake HTTP responses (no network needed)."""

import pytest
import requests

from src.weather import (
    LocationError,
    MissingApiKeyError,
    WeatherServiceError,
    from_coordinates,
    get_api_key,
    get_weather_report,
    search_locations,
)
from src.weather import openmeteo, openweather
from src.weather.weather_service import read_env_file

GEOCODE = {"results": [
    {"name": "Surat", "latitude": 21.19594, "longitude": 72.83023, "admin1": "Gujarat",
     "country": "India"},
    {"name": "Surat", "latitude": 45.93741, "longitude": 3.25451, "admin1": "Rhône-Alpes",
     "country": "France"}]}
DAILY = {"timezone": "Asia/Kolkata", "daily": {
    "time": ["2026-10-07", "2026-10-08"], "weather_code": [80, 0],
    "temperature_2m_max": [37.3, 36.0], "temperature_2m_min": [24.8, 24.1],
    "precipitation_sum": [3.5, 0.0], "precipitation_probability_max": [55, 5],
    "wind_speed_10m_max": [13.5, 11.2], "relative_humidity_2m_mean": [71, 60],
    "et0_fao_evapotranspiration": [4.7, 5.1]}}
OM_CURRENT = {"timezone": "Asia/Kolkata", "current": {
    "time": "2026-10-07T13:45", "interval": 900, "temperature_2m": 37.4,
    "apparent_temperature": 40.5, "relative_humidity_2m": 29, "wind_speed_10m": 3.1,
    "weather_code": 0, "precipitation": 0.0, "cloud_cover": 16}}
OWM_CURRENT = {"dt": 1791360000, "timezone": 19800, "weather": [{"main": "Rain",
               "description": "light rain"}], "main": {"temp": 30.2, "feels_like": 34.0,
               "humidity": 80}, "wind": {"speed": 2.5}, "rain": {"1h": 0.6},
               "clouds": {"all": 75}}
SURAT = from_coordinates(21.19594, 72.83023, "Surat")


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.ok = status < 400

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def fake_api(geocode=GEOCODE, daily=DAILY, om_current=OM_CURRENT, owm=OWM_CURRENT):
    """requests.get replacement; an Exception value is raised, a tuple is (payload, status)."""
    def get(url, params=None, timeout=None):
        if "geocoding" in url:
            answer = geocode
        elif "openweathermap" in url:
            answer = owm
        elif "current" in (params or {}):
            answer = om_current
        else:
            answer = daily
        if isinstance(answer, BaseException):
            raise answer
        payload, status = answer if isinstance(answer, tuple) else (answer, 200)
        return FakeResponse(payload, status)
    return get


# --- 1. Valid location --------------------------------------------------------------

def test_valid_city_search_and_full_report(monkeypatch):
    monkeypatch.setattr(requests, "get", fake_api())
    matches = search_locations("Surat")
    assert [m.label for m in matches] == ["Surat, Gujarat, India", "Surat, Rhône-Alpes, France"]
    report = get_weather_report(matches[0], api_key="key")
    assert report.ok and report.current.source == "OpenWeather"
    assert report.current.condition == "Light rain" and report.current.rain_mm == 0.6
    assert report.current.wind_speed_kmh == pytest.approx(9.0)        # 2.5 m/s
    assert report.timezone == "Asia/Kolkata" and len(report.forecast) == 2
    day = report.forecast[0]
    assert (day.condition, day.temp_max_c, day.precipitation_mm, day.et0_mm) == \
        ("Light rain showers", 37.3, 3.5, 4.7)


# --- 2. Invalid location ------------------------------------------------------------

def test_unknown_place_and_bad_input(monkeypatch):
    monkeypatch.setattr(requests, "get", fake_api(geocode={"generationtime_ms": 0.1}))
    with pytest.raises(LocationError, match="No place called"):
        search_locations("Xyzzyqwv")
    with pytest.raises(LocationError, match="at least 2 letters"):
        search_locations(" ")
    with pytest.raises(LocationError, match="Latitude must be between"):
        from_coordinates(200, 10)
    with pytest.raises(LocationError, match="must be numbers"):
        from_coordinates(None, 10)


# --- 3. Missing API key -------------------------------------------------------------

def test_missing_key_falls_back_to_open_meteo(monkeypatch):
    monkeypatch.setattr(requests, "get", fake_api())
    with pytest.raises(MissingApiKeyError, match="OPENWEATHER_API_KEY"):
        openweather.get_current_weather(1, 2, None)
    report = get_weather_report(SURAT, api_key=None)
    assert report.current.source == "Open-Meteo" and report.current.temperature_c == 37.4
    assert report.current.rain_period == "last 15 minutes"
    assert "OPENWEATHER_API_KEY is not set" in report.current_note
    assert report.forecast and report.forecast_error is None


def test_api_key_from_env_or_file(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nOTHER=1\nOPENWEATHER_API_KEY = \"abc123\"\n")
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    assert read_env_file(env)["OPENWEATHER_API_KEY"] == "abc123"
    assert get_api_key(env) == "abc123"
    assert get_api_key(tmp_path / "missing.env") is None
    monkeypatch.setenv("OPENWEATHER_API_KEY", "from-env")
    assert get_api_key(env) == "from-env"


# --- 4. API failures ----------------------------------------------------------------

@pytest.mark.parametrize("failure, message", [
    (requests.Timeout(), "did not respond"),
    (requests.ConnectionError(), "Could not connect"),
    (({"reason": "Server error"}, 500), "HTTP 500"),
    ((ValueError("not json"), 200), "could not be read"),
])
def test_service_failures_become_clear_errors(monkeypatch, failure, message):
    monkeypatch.setattr(requests, "get", fake_api(daily=failure, geocode=failure))
    with pytest.raises(WeatherServiceError, match=message):
        openmeteo.get_forecast(1, 2)
    with pytest.raises(WeatherServiceError, match=message):
        search_locations("Surat")


def test_malformed_and_rejected_responses(monkeypatch):
    monkeypatch.setattr(requests, "get", fake_api(daily={"daily": {"time": ["x"]}},
                                                  owm=({"cod": 401}, 401)))
    with pytest.raises(WeatherServiceError, match="unexpected response"):
        openmeteo.get_forecast(1, 2)
    with pytest.raises(WeatherServiceError, match="rejected the API key"):
        openweather.get_current_weather(1, 2, "bad")


def test_report_keeps_working_parts_when_one_service_fails(monkeypatch):
    monkeypatch.setattr(requests, "get", fake_api(daily=requests.Timeout(),
                                                  owm=requests.ConnectionError()))
    report = get_weather_report(SURAT, api_key="key")
    assert report.current.source == "Open-Meteo" and "OpenWeather is unavailable" in \
        report.current_note
    assert report.forecast == [] and "did not respond" in report.forecast_error

    monkeypatch.setattr(requests, "get", fake_api(daily=requests.Timeout(),
                                                  om_current=requests.Timeout()))
    report = get_weather_report(SURAT, api_key=None)
    assert report.current is None and "unavailable" in report.current_error
    assert not report.ok


# --- Security: the API key never reaches a message ------------------------------------

SECRET = "0123456789abcdef0123456789abcdef"


@pytest.mark.parametrize("error", [
    requests.TooManyRedirects(f"Exceeded redirects for https://api.openweathermap.org/data/2.5/"
                              f"weather?lat=21.2&appid={SECRET}"),
    requests.ConnectionError(f"Max retries exceeded with url: /data/2.5/weather?appid={SECRET}"),
    requests.Timeout(f"Read timed out: /weather?appid={SECRET}"),
])
def test_request_errors_never_show_the_api_key(monkeypatch, error):
    from src.weather import get_weather_report

    monkeypatch.setattr(requests, "get", fake_api(owm=error))
    report = get_weather_report(from_coordinates(21.2, 72.8), api_key=SECRET)
    shown = f"{report.current_note} {report.current_error} {report.forecast_error}"
    assert SECRET not in shown and "OpenWeather is unavailable" in report.current_note
    assert report.current is not None                        # Open-Meteo still answers


def test_redact_removes_the_key():
    from src.weather.weather_service import redact

    assert redact(f"bad url ?appid={SECRET}", SECRET) == "bad url ?appid=***"
    assert redact(None, SECRET) is None and redact("text", None) == "text"
