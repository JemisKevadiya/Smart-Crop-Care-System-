"""Combine current weather (OpenWeather) and the daily forecast (Open-Meteo).

    report = get_weather_report(location)
    report.current, report.current_note, report.current_error
    report.forecast, report.forecast_error

Each part is fetched independently, so one failing service does not hide the
other. If OpenWeather cannot be used (no key, rejected key, outage), current
conditions come from Open-Meteo instead and `current_note` says why. All values
are real API data; nothing is estimated or invented.
"""

from dataclasses import dataclass, field

from src.env import ENV_FILE, get_setting, read_env_file  # noqa: F401 (re-exported)

from . import openmeteo, openweather
from .common import MissingApiKeyError, WeatherError

API_KEY_NAME = "OPENWEATHER_API_KEY"


def get_api_key(env_file=ENV_FILE):
    """OPENWEATHER_API_KEY from the environment, else from .env; None if not set."""
    return get_setting(API_KEY_NAME, env_file)


@dataclass
class WeatherReport:
    location: object
    current: object = None
    current_note: str | None = None     # why a fallback source was used
    current_error: str | None = None
    timezone: str | None = None
    forecast: list = field(default_factory=list)
    forecast_error: str | None = None

    @property
    def ok(self):
        return self.current is not None and bool(self.forecast)


def redact(text, secret):
    """Remove a secret (the API key) from a message before it is shown anywhere."""
    if not text or not secret:
        return text
    return text.replace(secret, "***")


def get_current(location, api_key):
    """(CurrentWeather or None, note, error) - OpenWeather first, Open-Meteo as fallback."""
    note = None
    try:
        return openweather.get_current_weather(location.latitude, location.longitude,
                                               api_key), None, None
    except MissingApiKeyError:
        note = ("OPENWEATHER_API_KEY is not set, so current conditions come from Open-Meteo. "
                "Add the key to .env to use OpenWeather.")
    except WeatherError as exc:
        note = f"OpenWeather is unavailable ({exc}) Showing Open-Meteo's current conditions."
    try:
        return openmeteo.get_current_weather(location.latitude, location.longitude), note, None
    except WeatherError as exc:
        return None, None, f"Current weather is unavailable: {exc}"


def get_weather_report(location, api_key=None, days=7):
    report = WeatherReport(location)
    current, note, error = get_current(location, api_key)
    report.current, report.current_note, report.current_error = (
        current, redact(note, api_key), redact(error, api_key))
    try:
        report.timezone, report.forecast = openmeteo.get_forecast(
            location.latitude, location.longitude, days)
    except WeatherError as exc:
        report.forecast_error = f"The forecast is unavailable: {exc}"
    return report
