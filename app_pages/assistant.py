"""Agriculture Assistant page: a chat about crops, diseases, fertilizer, weather and news."""

import streamlit as st

from src.chatbot import format_context
from src.context import get_session_context
from src.ui import load_news
from src.ui.chat_panel import HISTORY_KEY, language_picker, show_chat

st.caption("Ask about plant diseases, symptoms, fertilizer, treatment, crop care, "
           "weather-related farming or the latest farmer news. Answers are general "
           "information, not professional agricultural advice.")

language = language_picker("chat")

crop_context = get_session_context()
with st.container(horizontal=True, vertical_alignment="center"):
    use_context = st.toggle("Use results from the other pages", value=True,
                            key="chat_use_context",
                            help="Shares the latest disease detection result, weather and "
                                 "farmer news with the assistant.")
    if st.button("Clear chat", icon=":material/delete_sweep:", key="chat_clear"):
        st.session_state[HISTORY_KEY] = []
    if st.button("Clear crop context", icon=":material/layers_clear:", key="context_clear",
                 help="Forget the latest disease result and weather in this session."):
        crop_context.clear_context()
        st.session_state.pop("last_leaf", None)
if use_context:
    load_news()     # cached headlines, so the assistant can answer news questions
context = crop_context.get_context() if use_context else None

context_text = format_context(context)
with st.expander("What the assistant knows from the app", icon=":material/info:",
                 expanded=bool(context and (context["disease"] or context["weather"]))):
    if context_text:
        st.text(context_text)
    else:
        st.caption("Nothing yet. Analyse a leaf on the Disease detection page or look up "
                   "your location on the Weather page, then come back.")

show_chat("chat", context, language)
