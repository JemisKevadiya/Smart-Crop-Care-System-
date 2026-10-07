"""Keep one CropContext per user session, and survive a browser refresh.

Streamlit reruns the script on every interaction; storing the context in
st.session_state keeps it across reruns and page switches. A browser refresh
(F5) starts a new Streamlit session, so the session's results are also kept in
server memory under a random session id that is placed in the URL (?sid=...).
Reloading that URL restores the crop context, the chat, the last leaf and the
answer language. Nothing is written to disk, and API keys are never stored.
"""

import copy
import threading
import time
import uuid

from .crop_context import CropContext

SESSION_KEY = "crop_context"
SID_KEY = "_sid"                 # session_state key holding this session's id
SID_PARAM = "sid"                # URL query parameter
PERSISTED_KEYS = (SESSION_KEY, "chat_messages", "last_leaf", "answer_language")
MAX_AGE_SECONDS = 12 * 3600      # saved sessions older than this are forgotten
MAX_SESSIONS = 100               # oldest saved sessions are dropped beyond this
MAX_STORED_IMAGE_BYTES = 200 * 1024 * 1024   # leaf photos kept in memory, all sessions

_store = {}                      # sid -> {"state": {...}, "saved_at": float}
_lock = threading.Lock()


def get_session_context():
    import streamlit as st

    if not isinstance(st.session_state.get(SESSION_KEY), CropContext):
        st.session_state[SESSION_KEY] = CropContext()
    return st.session_state[SESSION_KEY]


def _valid_sid(value):
    try:
        return isinstance(value, str) and uuid.UUID(hex=value).hex == value
    except ValueError:
        return False


def restore_session():
    """Call at the start of every run. Restores a saved session after a refresh.

    Returns this session's id and keeps it in the URL so a refresh can find it.
    """
    import streamlit as st

    sid = st.session_state.get(SID_KEY)
    if sid is None:
        requested = st.query_params.get(SID_PARAM)
        with _lock:
            saved = _store.get(requested) if _valid_sid(requested) else None
        if saved and time.time() - saved["saved_at"] <= MAX_AGE_SECONDS:
            sid = requested
            # A copy, so two tabs opened from the same link do not share objects.
            for key, value in copy.deepcopy(saved["state"]).items():
                st.session_state[key] = value
        else:
            sid = uuid.uuid4().hex
        st.session_state[SID_KEY] = sid
    if st.query_params.get(SID_PARAM) != sid:
        st.query_params[SID_PARAM] = sid
    return sid


def save_session():
    """Call at the end of every run (also when a page stops or reruns early)."""
    import streamlit as st

    sid = st.session_state.get(SID_KEY)
    if not sid:
        return
    state = {key: st.session_state[key] for key in PERSISTED_KEYS if key in st.session_state}
    now = time.time()
    with _lock:
        _store[sid] = {"state": state, "saved_at": now}
        for old in [s for s, v in _store.items() if now - v["saved_at"] > MAX_AGE_SECONDS]:
            del _store[old]
        while len(_store) > MAX_SESSIONS or (
                len(_store) > 1 and _image_bytes() > MAX_STORED_IMAGE_BYTES):
            del _store[min(_store, key=lambda s: _store[s]["saved_at"])]


def _image_bytes():
    """Total size of the leaf photos held in saved sessions (caller holds the lock)."""
    return sum(len((v["state"].get("last_leaf") or {}).get("data", b"")) for v in _store.values())


def forget_saved_sessions():
    """Remove every saved session (used by tests)."""
    with _lock:
        _store.clear()
