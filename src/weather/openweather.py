"""Current weather from OpenWeather (needs OPENWEATHER_API_KEY).

https://openweathermap.org/current - metric units; wind is converted from m/s to km/h.
"""

from datetime import datetime, timedelta, timezone

from .common import (
    CurrentWeather,
    MissingApiKeyError,
    WeatherServiceError,
    get_json,
    malformed,
)

CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
SERVICE = "OpenWeather"


def get_current_weather(latitude, longitude, api_key):
    if not api_key:
        raise MissingApiKeyError(
            "OPENWEATHER_API_KEY is not set. Add it to the .env file in the project folder "
            "(OPENWEATHER_API_KEY=your_key) and restart the app.")
    try:
        data = get_json(CURRENT_URL, {"lat": latitude, "lon": longitude, "appid": api_key,
                                      "units": "metric"}, SERVICE)
    except WeatherServiceError as exc:
        if exc.status == 401:
            raise WeatherServiceError(
                "OpenWeather rejected the API key. Check OPENWEATHER_API_KEY in .env; a new "
                "key can take a few hours to become active.", status=401) from exc
        if exc.status == 429:
            raise WeatherServiceError(
                "OpenWeather's request limit was reached. Try again in a few minutes.",
                status=429) from exc
        raise
    try:
        main, weather = data["main"], data["weather"][0]
        rain = data.get("rain") or {}
        offset = timedelta(seconds=int(data.get("timezone", 0)))
        observed = (datetime.fromtimestamp(int(data["dt"]), timezone(offset))
                    if "dt" in data else None)
        return CurrentWeather(
            temperature_c=float(main["temp"]),
            feels_like_c=float(main["feels_like"]) if "feels_like" in main else None,
            humidity_pct=float(main["humidity"]),
            wind_speed_kmh=float(data["wind"]["speed"]) * 3.6,
            condition=str(weather.get("description") or weather["main"]).capitalize(),
            # OpenWeather only includes "rain" while it is raining.
            rain_mm=float(rain.get("1h", 0.0)),
            rain_period="last hour",
            cloud_pct=float(data["clouds"]["all"]) if "clouds" in data else None,
            observed_at=observed,
            source=SERVICE)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise malformed(SERVICE, exc) from exc
