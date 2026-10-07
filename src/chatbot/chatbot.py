"""Agriculture Assistant: answers farming questions with Groq via LangChain.

    assistant = AgricultureAssistant()            # reads GROQ_API_KEY from .env
    answer = assistant.reply("What should I do next?", history, context=crop_context.get_context())

`history` is a list of {"role": "user"|"assistant", "content": str}. Errors are
raised as ChatbotError subclasses whose message can be shown to the user.
"""

from . import llm
from .prompts import build_system_prompt

MAX_HISTORY_MESSAGES = 10      # earlier turns are dropped to keep requests small
MAX_QUESTION_CHARS = 2000
MAX_HISTORY_CHARS = 4000       # per earlier message, so old answers stay small
UNAVAILABLE = "AI assistant is temporarily unavailable."


class ChatbotError(Exception):
    """Base class; str(error) is a message suitable for showing to the user."""


class MissingApiKeyError(ChatbotError):
    pass


class ChatServiceError(ChatbotError):
    pass


def _service_error(exc, model_name=None):
    """Translate Groq / network exceptions into a user-facing ChatServiceError."""
    try:
        import groq
    except ImportError:  # pragma: no cover - langchain-groq always installs groq
        groq = None
    if groq is not None:
        if isinstance(exc, groq.APITimeoutError):      # subclass of APIConnectionError
            return ChatServiceError(f"{UNAVAILABLE} It took too long to answer; please try again.")
        if isinstance(exc, groq.APIConnectionError):
            return ChatServiceError(f"{UNAVAILABLE} Could not connect to Groq; check your "
                                    "internet connection.")
        if isinstance(exc, groq.AuthenticationError):
            return ChatServiceError(f"{UNAVAILABLE} Groq rejected the API key; check "
                                    "GROQ_API_KEY in .env.")
        if isinstance(exc, groq.RateLimitError):
            return ChatServiceError(f"{UNAVAILABLE} The request limit was reached; please "
                                    "wait a minute and try again.")
        if isinstance(exc, groq.NotFoundError):
            return ChatServiceError(f"{UNAVAILABLE} Groq does not offer the model "
                                    f"'{model_name or llm.get_model_name()}'. "
                                    "Set GROQ_MODEL in .env to a current Groq model.")
        if isinstance(exc, groq.APIStatusError):
            return ChatServiceError(f"{UNAVAILABLE} Groq returned an error "
                                    f"(HTTP {exc.status_code}); please try again later.")
    if isinstance(exc, TimeoutError):
        return ChatServiceError(f"{UNAVAILABLE} It took too long to answer; please try again.")
    return ChatServiceError(f"{UNAVAILABLE} ({type(exc).__name__}) Please try again later.")


def build_messages(question, history=(), context=None, language="English"):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages = [SystemMessage(build_system_prompt(context, language=language))]
    for turn in list(history)[-MAX_HISTORY_MESSAGES:]:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(cls(turn["content"][:MAX_HISTORY_CHARS]))
    messages.append(HumanMessage(question))
    return messages


class AgricultureAssistant:
    def __init__(self, chat_model=None, api_key=None):
        """chat_model: any LangChain chat model (tests pass a fake one)."""
        if chat_model is None:
            api_key = api_key or llm.get_api_key()
            if not api_key:
                raise MissingApiKeyError(
                    "GROQ_API_KEY is not set. Add GROQ_API_KEY=your_key to the .env file in "
                    "the project folder (free key: console.groq.com), then ask again.")
            chat_model = llm.create_chat_model(api_key)
        self.chat_model = chat_model

    def reply(self, question, history=(), context=None, language="English"):
        """context: a CropContext.get_context() dict (or None); language: a key of LANGUAGES."""
        question = (question or "").strip()
        if not question:
            raise ChatbotError("Please type a question.")
        question = question[:MAX_QUESTION_CHARS]
        messages = build_messages(question, history, context, language)
        try:
            response = self.chat_model.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a clear message
            raise _service_error(exc, getattr(self.chat_model, "model_name", None)) from exc
        text = response.content if isinstance(response.content, str) else ""
        if not text.strip():
            raise ChatServiceError(f"{UNAVAILABLE} It returned an empty answer; please "
                                   "rephrase your question.")
        return text.strip()
