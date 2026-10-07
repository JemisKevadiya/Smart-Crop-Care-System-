"""Shared crop context passed between disease detection, advice, weather and chatbot."""

from .crop_context import RECOMMENDATION_FIELDS, CropContext
from .session import SESSION_KEY, get_session_context

__all__ = ["CropContext", "RECOMMENDATION_FIELDS", "SESSION_KEY", "get_session_context"]
