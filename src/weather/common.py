"""Errors, the HTTP helper and data types shared by the weather modules."""

from dataclasses import dataclass
from datetime import datetime

import requests

TIMEOUT_SECONDS = 10


class WeatherError(Exception):
    """Base class; str(error) is a message suitable for showing to the user."""


class MissingApiKeyError(WeatherError):
    """OPENWEATHER_API_KEY is not configured."""


class LocationError(WeatherError):
    """The location could not be found or is invalid."""


class WeatherServiceError(WeatherError):
    """A weather/location service failed: network, timeout, HTTP error or bad response."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def get_json(url, params, service, timeout=TIMEOUT_SECONDS):
    """GET a JSON document, turning every failure into a WeatherServiceError."""
    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.Timeout as exc:
        raise WeatherServiceError(
            f"{service} did not respond within {timeout} seconds. Please try again later."
        ) from None
    except requests.ConnectionError as exc:
        raise WeatherServiceError(
            f"Could not connect to {service}. Check your internet connection.") from None
    except requests.RequestException as exc:
        # Not str(exc): requests errors can contain the full URL, including ?appid=<key>.
        raise WeatherServiceError(
            f"Request to {service} failed ({type(exc).__name__}). Please try again later."
        ) from None

    try:
        data = response.json()
    except ValueError:
        data = None
    if not response.ok:
        detail = ""
        if isinstance(data, dict):
            detail = data.get("reason") or data.get("message") or ""
        raise WeatherServiceError(
            f"{service} returned an error (HTTP {response.status_code})"
            + (f": {detail}" if detail else "."), status=response.status_code)
    if not isinstance(data, dict):
        raise WeatherServiceError(f"{service} sent a response that could not be read.")
    return data


def malformed(service, exc):
    return WeatherServiceError(f"{service} sent an unexpected response ({exc!r}).")


@dataclass
class CurrentWeather:
    temperature_c: float
    feels_like_c: float | None
    humidity_pct: float
    wind_speed_kmh: float
    condition: str
    rain_mm: float | None      # recent rain, over `rain_period`
    rain_period: str | None
    cloud_pct: float | None
    observed_at: datetime | None
    source: str


@dataclass
class DailyForecast:
    date: datetime
    condition: str
    temp_max_c: float | None
    temp_min_c: float | None
    precipitation_mm: float | None
    precipitation_probability_pct: float | None
    humidity_mean_pct: float | None
    wind_max_kmh: float | None
    et0_mm: float | None       # FAO reference evapotranspiration (crop water demand)
