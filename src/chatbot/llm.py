"""Create the Groq chat model through LangChain.

GROQ_API_KEY (required), GROQ_MODEL and GROQ_FALLBACK_MODEL (optional) come from
the environment or the project's .env file; nothing is hard-coded or stored.
"""

from src.env import get_setting

API_KEY_NAME = "GROQ_API_KEY"
MODEL_SETTING = "GROQ_MODEL"
DEFAULT_MODEL = "openai/gpt-oss-120b"    # override with GROQ_MODEL in .env
FALLBACK_SETTING = "GROQ_FALLBACK_MODEL"
# Used when the main model is rate-limited or failing; Groq limits each model separately.
DEFAULT_FALLBACK_MODEL = "openai/gpt-oss-20b"
TIMEOUT_SECONDS = 30
TEMPERATURE = 0.3          # low: factual, consistent answers
MAX_TOKENS = 1500         # typical answers use 600-800 tokens
REASONING_EFFORT = "low"   # gpt-oss models: less hidden reasoning, faster answers


def get_api_key():
    return get_setting(API_KEY_NAME)


def get_model_name():
    return get_setting(MODEL_SETTING) or DEFAULT_MODEL


def get_fallback_model_name():
    """The backup model, or None if it is switched off ("none") or equals the main one."""
    name = get_setting(FALLBACK_SETTING) or DEFAULT_FALLBACK_MODEL
    return None if name.lower() == "none" or name == get_model_name() else name


def create_chat_model(api_key, model=None, timeout=TIMEOUT_SECONDS):
    """A LangChain ChatGroq model. Imported lazily so the app starts without it."""
    from langchain_groq import ChatGroq

    model = model or get_model_name()
    extra = {"reasoning_effort": REASONING_EFFORT} if model.startswith("openai/gpt-oss") else {}
    # max_retries: the Groq client waits and retries rate-limit (429) and server errors.
    return ChatGroq(model=model, api_key=api_key, timeout=timeout, max_retries=2,
                    temperature=TEMPERATURE, max_tokens=MAX_TOKENS, **extra)
