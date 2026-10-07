"""Daily forecast (and current conditions) from Open-Meteo - free, no API key.

https://open-meteo.com/en/docs - metric units, times in the location's own time zone.
"""

from datetime import datetime

from .common import CurrentWeather, DailyForecast, get_json, malformed

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SERVICE = "Open-Meteo"
MAX_DAYS = 16

DAILY_FIELDS = ("weather_code", "temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "precipitation_probability_max", "wind_speed_10m_max",
                "relative_humidity_2m_mean", "et0_fao_evapotranspiration")
CURRENT_FIELDS = ("temperature_2m", "apparent_temperature", "relative_humidity_2m",
                  "wind_speed_10m", "weather_code", "precipitation", "cloud_cover")

# WMO weather interpretation codes used by Open-Meteo.
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    56: "Light freezing drizzle", 57: "Freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light rain showers", 81: "Rain showers", 82: "Violent rain showers",
    85: "Light snow showers", 86: "Snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with light hail", 99: "Thunderstorm with hail",
}


def describe_code(code):
    if code is None:
        return "Unknown"
    return WMO_CODES.get(int(code), f"Weather code {int(code)}")


def _num(value):
    return None if value is None else float(value)


def get_forecast(latitude, longitude, days=7):
    """(time zone name, list of DailyForecast) for the next `days` days, today first."""
    days = max(1, min(int(days), MAX_DAYS))
    data = get_json(FORECAST_URL, {"latitude": latitude, "longitude": longitude,
                                   "timezone": "auto", "forecast_days": days,
                                   "daily": ",".join(DAILY_FIELDS)}, SERVICE)
    try:
        daily = data["daily"]
        rows = []
        for i, day in enumerate(daily["time"]):
            value = {f: daily.get(f, [None] * len(daily["time"]))[i] for f in DAILY_FIELDS}
            rows.append(DailyForecast(
                date=datetime.fromisoformat(day),
                condition=describe_code(value["weather_code"]),
                temp_max_c=_num(value["temperature_2m_max"]),
                temp_min_c=_num(value["temperature_2m_min"]),
                precipitation_mm=_num(value["precipitation_sum"]),
                precipitation_probability_pct=_num(value["precipitation_probability_max"]),
                humidity_mean_pct=_num(value["relative_humidity_2m_mean"]),
                wind_max_kmh=_num(value["wind_speed_10m_max"]),
                et0_mm=_num(value["et0_fao_evapotranspiration"])))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise malformed(SERVICE, exc) from exc
    if not rows:
        raise malformed(SERVICE, ValueError("no forecast days"))
    return data.get("timezone"), rows


def get_current_weather(latitude, longitude):
    """Current conditions from Open-Meteo's model (used when OpenWeather is unavailable)."""
    data = get_json(FORECAST_URL, {"latitude": latitude, "longitude": longitude,
                                   "timezone": "auto", "current": ",".join(CURRENT_FIELDS)},
                    SERVICE)
    try:
        c = data["current"]
        return CurrentWeather(
            temperature_c=float(c["temperature_2m"]),
            feels_like_c=_num(c.get("apparent_temperature")),
            humidity_pct=float(c["relative_humidity_2m"]),
            wind_speed_kmh=float(c["wind_speed_10m"]),
            condition=describe_code(c.get("weather_code")),
            rain_mm=_num(c.get("precipitation")),
            rain_period=f"last {int(c.get('interval', 900)) // 60} minutes",
            cloud_pct=_num(c.get("cloud_cover")),
            observed_at=datetime.fromisoformat(c["time"]) if c.get("time") else None,
            source=SERVICE)
    except (KeyError, TypeError, ValueError) as exc:
        raise malformed(SERVICE, exc) from exc
