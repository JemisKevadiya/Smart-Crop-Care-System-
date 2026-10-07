"""Smart Crop Care dashboard: disease, crop care, weather, news, summary and assistant.

Each section reads the shared crop context and fails on its own: weather, news or
assistant problems show a message but never hide the disease result or the advice.
"""

import pandas as pd
import streamlit as st

from src.context import get_session_context
from src.disease_detection import ModelLoadError
from src.disease_detection.preprocessing import (
    ALLOWED_EXTENSIONS,
    ALLOWED_FORMATS,
    ImageValidationError,
    validate_filename,
)
from src.fertilizer import get_recommendation
from src.integration import CropCareAnalyzer
from src.ui import show_news
from src.ui.chat_panel import show_chat
from src.ui.news_panel import md_escape
from src.weather import Location, get_api_key, get_weather_report

UPLOAD_TYPES = list(ALLOWED_EXTENSIONS)


@st.cache_resource(show_spinner="Loading the disease detection model...")
def get_analyzer():
    return CropCareAnalyzer()


@st.cache_data(max_entries=32, show_spinner=False)
def analyze(image_bytes, _analyzer):
    return _analyzer.analyze(image_bytes)


@st.cache_data(ttl=600, show_spinner="Fetching weather...")
def fetch_weather(location):
    # The key is read here, not passed in, so it never becomes part of a cache key.
    return get_weather_report(location, api_key=get_api_key(), days=7)


def show_weather_context():
    """Weather for the location saved in the crop context (set on the Weather page).

    Weather adds environmental context for crop-care decisions; it never changes the
    disease prediction. Any weather failure is shown as a warning, never an error that
    would hide the disease result or advice above it.
    """
    st.subheader("Weather context", icon=":material/partly_cloudy_day:")
    st.caption("Weather provides additional environmental context for crop-care decisions "
               "(e.g. when to spray or irrigate). It does not change the disease prediction.")
    context = get_session_context()
    saved = context.weather.get("location")
    if not saved:
        st.info("Set your location on the Weather page to see the weather here.",
                icon=":material/location_on:")
        st.page_link("app_pages/weather.py", label="Open the Weather page",
                     icon=":material/partly_cloudy_day:")
        return
    location = Location(saved["name"], saved["latitude"], saved["longitude"],
                        source=saved.get("source", "search"))
    try:
        report = fetch_weather(location)
    except Exception:  # noqa: BLE001 - weather must never break the disease results
        report = None
    if report is None or (report.current is None and not report.forecast):
        # Keep the location but drop older readings, so the summary and the assistant
        # also say the weather is unavailable instead of using out-of-date values.
        context.set_weather(saved)
        reason = report and (report.current_error or report.forecast_error)
        st.warning("Weather information is currently unavailable."
                   + (f" {reason}" if reason else ""), icon=":material/cloud_off:")
        return
    context.update_from_weather(report)

    st.markdown(f":material/location_on: **{location.label}**")
    today = report.forecast[0] if report.forecast else None
    c = report.current
    with st.container(horizontal=True):
        if c:
            st.metric("Temperature", f"{c.temperature_c:.1f} °C", border=True)
            st.metric("Humidity", f"{c.humidity_pct:.0f}%", border=True)
            st.metric("Wind", f"{c.wind_speed_kmh:.1f} km/h", border=True)
            st.metric("Condition", c.condition, border=True)
        if today and today.precipitation_mm is not None:
            chance = today.precipitation_probability_pct
            st.metric("Rain today", f"{today.precipitation_mm:.1f} mm", border=True,
                      help=None if chance is None else f"{chance:.0f}% chance of rain")
    if report.current_error:
        st.warning(report.current_error, icon=":material/cloud_off:")
    if report.forecast:
        st.dataframe(pd.DataFrame([{
            "Date": d.date, "Condition": d.condition, "Min °C": d.temp_min_c,
            "Max °C": d.temp_max_c, "Rain mm": d.precipitation_mm,
            "Rain %": d.precipitation_probability_pct} for d in report.forecast[:3]]),
            hide_index=True, width="stretch",
            column_config={"Date": st.column_config.DateColumn(format="ddd D MMM"),
                           "Min °C": st.column_config.NumberColumn(format="%.1f"),
                           "Max °C": st.column_config.NumberColumn(format="%.1f"),
                           "Rain mm": st.column_config.NumberColumn(format="%.1f"),
                           "Rain %": st.column_config.NumberColumn(format="%.0f")})
    else:
        st.warning(report.forecast_error or "The forecast is currently unavailable.",
                   icon=":material/cloud_off:")
    sources = [c.source] if c else []
    if report.forecast:
        sources.append("Open-Meteo (forecast)")
    st.caption("Source: " + ", ".join(sources) + ". Full 7-day forecast on the Weather page.")


def show_advice(rec):
    """The recommendation table entry for the predicted disease."""
    with st.container(border=True):
        st.markdown(f":material/compost: **Fertilizer:** {rec.fertilizer}")
        st.markdown(f":material/medication: **Treatment:** {rec.treatment}")
        st.markdown(f":material/eco: **Eco-friendly treatment:** {rec.eco_friendly_treatment}")
        st.markdown(f":material/science: **Chemical treatment:** {rec.chemical_treatment}")
        if rec.notes:
            st.caption(f"Note: {rec.notes}")
    st.caption(rec.disclaimer)


def summary_lines(context):
    """Plain sentences built only from the crop context (nothing generated or guessed)."""
    lines = []
    if context.has_disease:
        status = (" - low confidence, advice withheld" if context.prediction_status ==
                  "low_confidence" else "")
        lines.append(f":material/eco: **Detected disease:** {context.crop} - {context.disease} "
                     f"({context.confidence:.1%} model confidence{status})")
    rec = context.recommendation
    if rec.get("fertilizer") or rec.get("treatment"):
        if rec.get("fertilizer"):
            lines.append(f":material/compost: **Fertilizer:** {rec['fertilizer']}")
        if rec.get("treatment"):
            lines.append(f":material/medication: **Treatment:** {rec['treatment']}")
    elif context.has_disease:
        lines.append(":material/medication: **Fertilizer and treatment:** not available for "
                     "this result.")
    current = context.weather.get("current") or {}
    days = context.weather.get("forecast") or []
    if current or days:
        text = f":material/partly_cloudy_day: **Weather in {context.weather['location']['name']}:** "
        bits = []
        if current:
            bits.append(f"{current['temperature_c']:.1f} °C, {current['humidity_pct']:.0f}% "
                        f"humidity, {current['condition'].lower()}")
        if days and days[0].get("precipitation_mm") is not None:
            chance = days[0].get("precipitation_probability_pct")
            bits.append(f"rain today {days[0]['precipitation_mm']:.1f} mm"
                        + ("" if chance is None else f" ({chance:.0f}% chance)"))
        lines.append(text + ", ".join(bits))
    elif context.weather.get("location"):
        lines.append(":material/partly_cloudy_day: **Weather:** currently unavailable.")
    else:
        lines.append(":material/partly_cloudy_day: **Weather:** no location set yet.")
    news = context.news.get("items") or []
    if news:
        lines.append(f":material/newspaper: **Latest agriculture update:** "
                     f"{md_escape(news[0]['title'])} ({news[0]['source']})")
    elif context.news.get("status") == "unavailable":
        lines.append(":material/newspaper: **Latest agriculture update:** live farmer news is "
                     "temporarily unavailable.")
    return lines


st.caption("Upload a clear photo of a single leaf to check it for disease and get "
           "fertilizer and treatment advice.")

# `type` makes the browser and Streamlit's server refuse other extensions;
# validate_filename below is a second check, and validate_image checks the content.
uploaded = st.file_uploader(
    "Upload leaf image", type=UPLOAD_TYPES,
    help=f"Supported formats: {', '.join(sorted(ALLOWED_FORMATS))}. Max 15 MB.")

# Streamlit forgets the uploaded file when the user switches pages, so the last leaf is
# kept in session state and shown again while its result is in the crop context.
if uploaded is not None:
    try:
        validate_filename(uploaded.name)    # extension; the content is validated below
    except ImageValidationError as exc:
        st.error(f"This file can't be analyzed: {exc}", icon=":material/broken_image:")
        st.stop()
    leaf = {"name": uploaded.name, "data": uploaded.getvalue()}
    st.session_state.last_leaf = leaf
else:
    leaf = st.session_state.get("last_leaf") if get_session_context().has_disease else None
    if leaf is None:
        st.stop()
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(f"Showing your last analysed leaf ({leaf['name']}). Upload a new photo "
                   "to replace it.")
        if st.button("Start over", icon=":material/restart_alt:", key="leaf_reset"):
            st.session_state.pop("last_leaf", None)
            get_session_context().clear_disease()
            st.rerun()

try:
    analyzer = get_analyzer()
except ModelLoadError as exc:
    st.error(f"The disease detection model could not be loaded. {exc}", icon=":material/error:")
    st.stop()

with st.spinner("Analyzing leaf..."):
    result = analyze(leaf["data"], analyzer)
# Share the result (crop, disease, confidence, advice) with the other pages.
get_session_context().update_from_analysis(result)

if result.status == "invalid_image":
    st.error(f"This file can't be analyzed: {result.message}", icon=":material/broken_image:")
    st.stop()
if result.status == "model_error":
    st.error(result.message, icon=":material/error:")
    st.stop()

prediction = result.prediction
if result.status == "out_of_scope":
    image_col, message_col = st.columns([1, 1], gap="medium")
    with image_col:
        st.subheader("Uploaded image", icon=":material/image:")
        st.image(leaf["data"], caption=leaf["name"], width="stretch")
    with message_col:
        st.subheader("Not a supported leaf", icon=":material/block:")
        st.warning(result.message, icon=":material/image_not_supported:")
        st.caption("No disease or advice is shown, because the model only knows "
                   "leaves of the supported crops listed in the sidebar.")
    st.stop()

# --- 1. Disease detection -----------------------------------------------------------
st.subheader("Disease detection", icon=":material/eco:")
with st.container(border=True):
    image_col, result_col = st.columns([1, 1], gap="medium")
    with image_col:
        st.image(leaf["data"], caption=leaf["name"], width="stretch")
    with result_col:
        st.metric("Crop", prediction.crop)
        st.metric("Disease", prediction.disease)
        st.metric("Confidence", f"{prediction.confidence:.1%}")
        st.progress(prediction.confidence)
        if result.recommendation and result.recommendation.category:
            st.badge(result.recommendation.category,
                     color="green" if prediction.is_healthy else "orange")
    st.caption("Identified from the image by the ResNet50 model. A prediction can be wrong, "
               "so check the symptoms on the plant.")
with st.expander("Other possibilities", icon=":material/format_list_numbered:"):
    for candidate in prediction.top_k:
        st.markdown(f"{candidate.crop} - {candidate.disease}: **{candidate.probability:.1%}**")

if result.status == "low_confidence":
    st.warning(prediction.message, icon=":material/help:")
elif prediction.is_healthy:
    st.success(prediction.message, icon=":material/check_circle:")
else:
    st.info(prediction.message, icon=":material/coronavirus:")

# --- 2. Recommended crop care -------------------------------------------------------
st.subheader("Recommended crop care", icon=":material/compost:")
st.caption("Looked up in the project's fertilizer and treatment table for the predicted "
           "disease. It is not generated by the image model.")
if result.status == "low_confidence":
    st.caption("Advice is hidden because the prediction is uncertain; treating the wrong "
               "disease can waste money and harm the crop.")
    if st.toggle("Show advice for the most likely disease anyway", key="advise_uncertain"):
        rec = get_recommendation(prediction)
        if rec.found:
            show_advice(rec)
        else:
            st.warning(rec.message, icon=":material/warning:")
elif result.status == "recommendation_unavailable":
    reason = result.recommendation.message if result.recommendation else result.message
    st.warning(f"Crop-care advice is unavailable for this result. {reason}",
               icon=":material/warning:")
else:
    show_advice(result.recommendation)

# --- 3. Weather context -------------------------------------------------------------
show_weather_context()

# --- 4. Live farmer news (information only; never changes the result above) ---------
st.subheader("Live Farmer News", icon=":material/newspaper:")
show_news("dashboard", page_size=3)
st.page_link("app_pages/news.py", label="Open all farmer news", icon=":material/newspaper:")

# --- Summary: only what the modules above actually produced --------------------------
st.subheader("Smart Crop Care summary", icon=":material/summarize:")
with st.container(border=True):
    st.markdown("  \n".join(summary_lines(get_session_context())))

# --- 5. Smart Agriculture Assistant -------------------------------------------------
st.subheader("Smart Agriculture Assistant", icon=":material/smart_toy:")
st.caption("Ask about this result. The assistant receives the disease, advice, weather and "
           "news above from the shared crop context, and says when something is missing.")
with st.container():        # inside a container, the chat input stays in the dashboard
    show_chat("dashboard", get_session_context().get_context(), height=420)
st.page_link("app_pages/assistant.py", label="Open the full Agriculture Assistant",
             icon=":material/open_in_full:")
