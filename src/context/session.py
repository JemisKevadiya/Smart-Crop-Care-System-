"""Keep one CropContext per user session in Streamlit's session state.

Streamlit reruns the script on every interaction; storing the context in
st.session_state keeps it across reruns and page switches for the current
browser session only. It never holds API keys.
"""

from .crop_context import CropContext

SESSION_KEY = "crop_context"


def get_session_context():
    import streamlit as st

    if not isinstance(st.session_state.get(SESSION_KEY), CropContext):
        st.session_state[SESSION_KEY] = CropContext()
    return st.session_state[SESSION_KEY]
