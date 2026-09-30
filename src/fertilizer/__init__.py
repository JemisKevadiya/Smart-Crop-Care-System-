"""Fertilizer and treatment recommendations for predicted plant diseases."""

from .fertilizer_data import FertilizerDataError, load_recommendations
from .recommendation import Recommendation, get_recommendation

__all__ = ["get_recommendation", "Recommendation", "FertilizerDataError",
           "load_recommendations"]
