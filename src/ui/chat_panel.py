"""Agriculture Assistant chat, used on the dashboard and on the Assistant page.

    show_chat("dashboard", context)       # context: CropContext.get_context() or None

Both places share one chat history (st.session_state.chat_messages) and one answer
language. The Groq key is read from .env for each question and never stored.
"""

import streamlit as st

from src.chatbot import LANGUAGES, AgricultureAssistant, ChatbotError

HISTORY_KEY = "chat_messages"
LANGUAGE_KEY = "answer_language"    # plain value shared by every page's language picker
STARTERS_WITH_CONTEXT = ["What disease was detected?", "What should I do next?",
                         "Will the weather affect my crop-care planning?",
                         "What are the latest agriculture updates?"]
STARTERS = ["How do I recognise early blight on tomatoes?",
            "How should I fertilize potatoes?",
            "What are the latest agriculture updates?"]


def history():
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    return st.session_state[HISTORY_KEY]


def language_picker(prefix):
    language = st.segmented_control(
        "Answer language", list(LANGUAGES),
        default=st.session_state.get(LANGUAGE_KEY, "English"), required=True,
        format_func=LANGUAGES.get, key=f"{prefix}_language",
        help="The assistant answers in this language. You can ask in any of them.")
    st.session_state[LANGUAGE_KEY] = language
    return language


def ask(question, context, language):
    """Append the question and the assistant's answer (or an error) to the chat."""
    messages = history()
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                # Created per question: the key is read from .env and never stored.
                answer = AgricultureAssistant().reply(question, messages[:-1], context, language)
        except ChatbotError as exc:
            messages.pop()      # let the user ask again
            st.error(str(exc), icon=":material/error:")
            return
        st.markdown(answer)
    messages.append({"role": "assistant", "content": answer})


def show_chat(prefix, context, language=None, height=None):
    """Chat history, starter questions and the chat input.

    height: fixed height for the history (scrolls) - used for the inline dashboard chat.
    """
    language = language or language_picker(prefix)
    fixed = height if height and history() else None   # no empty box before the first question
    box = st.container(height=fixed or "content", border=bool(fixed))
    with box:
        for message in history():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    clicked = None
    if not history():
        starters = STARTERS_WITH_CONTEXT if context and context.get("disease") else STARTERS
        with st.container(horizontal=True):
            for i, text in enumerate(starters):
                if st.button(text, key=f"{prefix}_starter_{i}", type="tertiary",
                             icon=":material/chat_bubble:"):
                    clicked = text

    question = st.chat_input("Ask about crops, diseases, fertilizer, weather or news...",
                             key=f"{prefix}_input")
    if question or clicked:
        with box:
            ask(question or clicked, context, language)
