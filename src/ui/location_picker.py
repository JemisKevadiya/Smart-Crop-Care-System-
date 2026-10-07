"""Location picker used on the Home and Weather pages.

    location = choose_location("weather")   # the shared weather.Location, or None

`prefix` keeps each page's widgets and remembered results separate. A newly picked
place is saved in the shared crop context, so a location chosen on either page is
used everywhere (dashboard weather, assistant).
"""

import streamlit as st

from src.context import get_session_context
from src.weather import Location, WeatherError, from_coordinates, locate_by_ip, search_locations

MODES = ["City or village", "Coordinates", "Approximate (IP)"]


def choose_location(prefix):
    """Show the picker, save a newly picked place, and return the shared location (or None).

    A page's earlier pick is applied only once, so revisiting a page never overwrites a
    location chosen later on another page.
    """
    picked = _picker(prefix)
    applied = f"{prefix}_applied"
    if picked is not None and st.session_state.get(applied) != picked:
        st.session_state[applied] = picked
        get_session_context().set_location(picked.label, picked.latitude, picked.longitude,
                                           picked.source)
    saved = get_session_context().weather.get("location")
    if not saved:
        return None
    return Location(saved["name"], saved["latitude"], saved["longitude"],
                    source=saved.get("source") or "search")


def _picker(prefix):
    """The widgets; returns the place selected on this page or None."""
    mode = st.segmented_control("Find location by", MODES, default=MODES[0],
                                key=f"{prefix}_mode")
    if mode == "City or village":
        with st.form(f"{prefix}_city", border=False):
            with st.container(horizontal=True, vertical_alignment="bottom"):
                query = st.text_input("City or village", placeholder="e.g. Surat",
                                      key=f"{prefix}_query")
                searched = st.form_submit_button("Search", icon=":material/search:")
        if searched:
            try:
                st.session_state[f"{prefix}_matches"] = search_locations(query)
                st.session_state.pop(f"{prefix}_applied", None)
            except WeatherError as exc:
                st.session_state[f"{prefix}_matches"] = None
                st.error(str(exc), icon=":material/wrong_location:")
        matches = st.session_state.get(f"{prefix}_matches")
        if matches:
            return st.selectbox("Choose the place", matches, format_func=lambda loc: loc.label,
                                key=f"{prefix}_match")
    elif mode == "Coordinates":
        with st.form(f"{prefix}_coords_form", border=False):
            with st.container(horizontal=True, vertical_alignment="bottom"):
                lat = st.number_input("Latitude", -90.0, 90.0, value=None, format="%.4f",
                                      placeholder="e.g. 21.1702", key=f"{prefix}_lat")
                lon = st.number_input("Longitude", -180.0, 180.0, value=None, format="%.4f",
                                      placeholder="e.g. 72.8311", key=f"{prefix}_lon")
                submitted = st.form_submit_button("Show weather", icon=":material/my_location:")
        if submitted:
            try:
                st.session_state[f"{prefix}_coords"] = from_coordinates(lat, lon)
                st.session_state.pop(f"{prefix}_applied", None)
            except WeatherError as exc:
                st.error(str(exc), icon=":material/wrong_location:")
        return st.session_state.get(f"{prefix}_coords")
    elif mode == "Approximate (IP)":
        st.caption("Uses this computer's public IP address (sent to ipinfo.io). IP locations "
                   "come from the internet provider and can be off by hundreds of kilometres; "
                   "search for your city for accurate weather.")
        if st.button("Detect my approximate location", icon=":material/travel_explore:",
                     key=f"{prefix}_detect_ip"):
            try:
                st.session_state[f"{prefix}_ip"] = locate_by_ip()
                st.session_state.pop(f"{prefix}_applied", None)
            except WeatherError as exc:
                st.error(str(exc), icon=":material/wrong_location:")
        return st.session_state.get(f"{prefix}_ip")
    return None
