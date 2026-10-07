"""Weather page: current conditions (OpenWeather) and a daily forecast (Open-Meteo)."""

import pandas as pd
import streamlit as st

from src.context import get_session_context
from src.ui import choose_location
from src.weather import get_api_key, get_weather_report

FORECAST_DAYS = 7


@st.cache_data(ttl=600, show_spinner="Fetching weather...")
def fetch_report(location, api_key):
    return get_weather_report(location, api_key=api_key, days=FORECAST_DAYS)


def show_current(report):
    st.subheader("Current weather", icon=":material/thermostat:")
    if report.current_note:
        st.info(report.current_note, icon=":material/info:")
    if report.current_error:
        st.error(report.current_error, icon=":material/cloud_off:")
        return
    c = report.current
    with st.container(horizontal=True):
        st.metric("Temperature", f"{c.temperature_c:.1f} °C", border=True,
                  help=None if c.feels_like_c is None else f"Feels like {c.feels_like_c:.1f} °C")
        st.metric("Humidity", f"{c.humidity_pct:.0f}%", border=True)
        st.metric("Wind", f"{c.wind_speed_kmh:.1f} km/h", border=True)
        st.metric("Condition", c.condition, border=True)
    details = []
    if c.rain_mm is not None:
        details.append(f"Rain ({c.rain_period}): {c.rain_mm:.1f} mm")
    if c.cloud_pct is not None:
        details.append(f"Cloud cover: {c.cloud_pct:.0f}%")
    if c.observed_at:
        details.append(f"Observed {c.observed_at:%d %b %Y, %H:%M} (local time)")
    details.append(f"Source: {c.source}")
    st.caption(" · ".join(details))


def show_forecast(report):
    st.subheader(f"{FORECAST_DAYS}-day forecast", icon=":material/calendar_month:")
    if report.forecast_error:
        st.error(report.forecast_error, icon=":material/cloud_off:")
        return
    df = pd.DataFrame([{
        "Date": d.date, "Condition": d.condition, "Max °C": d.temp_max_c,
        "Min °C": d.temp_min_c, "Rain mm": d.precipitation_mm,
        "Rain %": d.precipitation_probability_pct,
        "Humidity %": d.humidity_mean_pct, "Wind km/h": d.wind_max_kmh,
        "ET₀ mm": d.et0_mm} for d in report.forecast])
    st.dataframe(
        df, hide_index=True, width="stretch",
        column_config={
            "Date": st.column_config.DateColumn(format="ddd D MMM"),
            "Max °C": st.column_config.NumberColumn(format="%.1f"),
            "Min °C": st.column_config.NumberColumn(format="%.1f"),
            "Rain mm": st.column_config.NumberColumn(format="%.1f",
                                                     help="Total rain expected that day"),
            "Rain %": st.column_config.NumberColumn(
                format="%.0f", help="Highest chance of rain during the day"),
            "Humidity %": st.column_config.NumberColumn(format="%.0f",
                                                        help="Daily mean relative humidity"),
            "Wind km/h": st.column_config.NumberColumn(format="%.1f",
                                                       help="Highest wind speed during the day"),
            "ET₀ mm": st.column_config.NumberColumn(
                format="%.1f", help="Reference evapotranspiration: water a well-watered "
                                    "reference crop loses that day. Higher values mean the "
                                    "crop needs more irrigation."),
        })
    chart = df.set_index("Date")  # a real date axis keeps the days in order
    temp_col, rain_col = st.columns(2)
    with temp_col:
        st.caption("Temperature (°C)")
        st.line_chart(chart[["Max °C", "Min °C"]], height=220, color=["#d9534f", "#2a78d6"])
    with rain_col:
        st.caption("Rain (mm)")
        st.bar_chart(chart[["Rain mm"]], height=220, color="#2a78d6")
    tz = f" Times are in {report.timezone}." if report.timezone else ""
    st.caption(f"Source: Open-Meteo.{tz}")


st.caption("Current weather and a 7-day forecast for your farm's location, to help plan "
           "irrigation, spraying and field work.")

# Saved in the crop context at once, so other pages have it even before the data arrives.
location = choose_location("weather")
if location is None:
    st.info("Search for your city or village to see its weather.", icon=":material/location_on:")
    st.stop()

with st.container(border=True):
    st.subheader(location.label, icon=":material/location_on:")
    source = {"search": "Place search", "coordinates": "Entered coordinates",
              "ip": "Approximate location from IP address"}[location.source]
    st.caption(f"Coordinates: {location.coordinates} · {source}")
    if location.source == "ip":
        st.warning("This location is approximate. Search for your city if it looks wrong.",
                   icon=":material/warning:")
    if st.button("Refresh weather", icon=":material/refresh:", key="weather_refresh"):
        fetch_report.clear()

report = fetch_report(location, get_api_key())
# Share the weather with the other pages.
get_session_context().update_from_weather(report)
show_current(report)
show_forecast(report)
