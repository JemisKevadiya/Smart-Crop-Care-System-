"""Shared crop context: what the app currently knows about the farmer's crop.

    Disease detection -> CropContext <- Weather, Live news
                             |
                  fertilizer/treatment advice, Agriculture Assistant

Modules exchange plain, structured data through this object instead of importing
each other. Values only ever come from the existing modules; anything unknown
stays None (or an empty dict). The context holds no images, models or API keys.

    ctx = CropContext()
    ctx.update_from_analysis(result)      # CropCareAnalyzer result
    ctx.update_from_weather(report)       # weather_service.WeatherReport
    ctx.update_from_news(report)          # news_service.NewsReport (headlines only)
    ctx.get_context()                     # plain dict, safe to show or send to the chatbot
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime

RECOMMENDATION_FIELDS = ("category", "fertilizer", "treatment", "eco_friendly_treatment",
                         "chemical_treatment", "notes")


MAX_NEWS_ITEMS = 10            # headlines kept for the assistant; full articles never are
NEWS_SUMMARY_CHARS = 200


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _confidence(value):
    if value is None:
        return None
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be between 0 and 1 (got {value})")
    return value


@dataclass
class CropContext:
    crop: str | None = None
    disease: str | None = None
    confidence: float | None = None          # model probability, 0-1
    is_healthy: bool | None = None
    prediction_status: str | None = None     # CropCareAnalyzer status, e.g. "ok"
    recommendation: dict = field(default_factory=dict)
    weather: dict = field(default_factory=dict)
    news: dict = field(default_factory=dict)     # status, fetched_at, items (headlines)
    updated: dict = field(default_factory=dict)  # part -> time it was last set

    # --- setters ------------------------------------------------------------------
    def set_disease(self, crop, disease, confidence=None, is_healthy=None, status=None):
        self.crop, self.disease = crop or None, disease or None
        self.confidence = _confidence(confidence)
        self.is_healthy, self.prediction_status = is_healthy, status
        self.updated["disease"] = _now()

    def set_confidence(self, confidence):
        self.confidence = _confidence(confidence)
        self.updated["disease"] = _now()

    def set_recommendation(self, **fields):
        """Fertilizer/treatment advice; unknown keys are rejected, empty values dropped."""
        unknown = set(fields) - set(RECOMMENDATION_FIELDS)
        if unknown:
            raise ValueError(f"unknown recommendation fields: {sorted(unknown)}")
        self.recommendation = {k: v for k, v in fields.items() if v not in (None, "")}
        self.updated["recommendation"] = _now()

    def set_location(self, name, latitude, longitude, source=None):
        """Remember the chosen location right away; weather data is added when it arrives.

        Choosing the same place again keeps the weather already stored for it.
        """
        location = {"name": name, "latitude": latitude, "longitude": longitude,
                    "source": source}
        current = self.weather.get("location") or {}
        if (current.get("latitude"), current.get("longitude")) != (latitude, longitude):
            self.set_weather(location)

    def set_weather(self, location, current=None, forecast=None, timezone=None):
        """location: dict (name, latitude, longitude...); current: dict; forecast: list of dicts."""
        self.weather = {"location": dict(location), "current": dict(current or {}),
                        "forecast": [dict(day) for day in forecast or []],
                        "timezone": timezone}
        self.updated["weather"] = _now()

    def set_news(self, status, items=(), fetched_at=None, crop=None):
        """status: "ok", "empty" or "unavailable"; items: dicts from NewsItem.to_dict().

        News is information only: it never changes the prediction or the advice.
        """
        if status not in ("ok", "empty", "unavailable"):
            raise ValueError(f"unknown news status {status!r}")
        self.news = {"status": status, "fetched_at": fetched_at, "prioritized_crop": crop,
                     "items": [dict(i) for i in list(items)[:MAX_NEWS_ITEMS]]}
        self.updated["news"] = _now()

    # --- clearing -----------------------------------------------------------------
    def clear_disease(self):
        self.crop = self.disease = self.confidence = None
        self.is_healthy = self.prediction_status = None
        self.recommendation = {}
        self.updated.pop("disease", None)
        self.updated.pop("recommendation", None)

    def clear_weather(self):
        self.weather = {}
        self.updated.pop("weather", None)

    def clear_news(self):
        self.news = {}
        self.updated.pop("news", None)

    def clear_context(self):
        self.clear_disease()
        self.clear_weather()
        self.clear_news()

    # --- reading ------------------------------------------------------------------
    def get_context(self):
        """A copy of the context as plain data (dicts, lists, str, float, bool, None)."""
        return copy.deepcopy({
            "crop": self.crop, "disease": self.disease, "confidence": self.confidence,
            "is_healthy": self.is_healthy, "prediction_status": self.prediction_status,
            "recommendation": self.recommendation, "weather": self.weather,
            "news": self.news, "updated": self.updated,
        })

    @property
    def has_disease(self):
        return self.disease is not None

    @property
    def has_weather(self):
        return bool(self.weather)

    @property
    def has_news(self):
        return bool(self.news.get("items"))

    @property
    def is_empty(self):
        return not (self.has_disease or self.recommendation or self.has_weather or self.news)

    # --- adapters from the existing modules (read-only use of their results) --------
    def update_from_analysis(self, result):
        """Copy a CropCareAnalyzer AnalysisResult into the context.

        Rejected or failed analyses clear the disease part, so the context never
        describes an image the app did not accept.
        """
        p = getattr(result, "prediction", None)
        if p is None or result.status in ("invalid_image", "model_error", "out_of_scope"):
            self.clear_disease()
            return
        self.set_disease(p.crop, p.disease, p.confidence, p.is_healthy, result.status)
        rec = getattr(result, "recommendation", None)
        if rec is not None and rec.found:
            self.set_recommendation(**{f: getattr(rec, f) for f in RECOMMENDATION_FIELDS})
        else:
            self.recommendation = {}       # advice withheld (low confidence) or unavailable
            self.updated.pop("recommendation", None)

    def update_from_weather(self, report):
        """Copy a weather_service.WeatherReport into the context."""
        loc = report.location
        location = {"name": loc.label, "latitude": loc.latitude, "longitude": loc.longitude,
                    "source": loc.source}
        current = {}
        if report.current is not None:
            c = report.current
            current = {"temperature_c": c.temperature_c, "feels_like_c": c.feels_like_c,
                       "humidity_pct": c.humidity_pct, "wind_speed_kmh": round(c.wind_speed_kmh, 1),
                       "condition": c.condition, "rain_mm": c.rain_mm,
                       "rain_period": c.rain_period, "source": c.source}
        forecast = [{"date": d.date.date().isoformat(), "condition": d.condition,
                     "temp_min_c": d.temp_min_c, "temp_max_c": d.temp_max_c,
                     "precipitation_mm": d.precipitation_mm,
                     "precipitation_probability_pct": d.precipitation_probability_pct,
                     "humidity_mean_pct": d.humidity_mean_pct, "wind_max_kmh": d.wind_max_kmh,
                     "et0_mm": d.et0_mm} for d in report.forecast]
        self.set_weather(location, current, forecast, report.timezone)

    def update_from_news(self, report, crop=None):
        """Copy the newest headlines of a news_service.NewsReport (None = news unavailable).

        crop: the crop the items were prioritized for (already ordered by the caller).
        """
        if report is None:
            self.set_news("unavailable")
            return
        items = [i.to_dict(NEWS_SUMMARY_CHARS) for i in report.items[:MAX_NEWS_ITEMS]]
        self.set_news("ok" if items else "empty", items,
                      report.fetched_at.isoformat(timespec="seconds"), crop)
