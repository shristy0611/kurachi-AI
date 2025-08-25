"""
NLP Services Module
Provides natural language processing capabilities with graceful degradation.
"""

from .spacy_guard import (
    ensure_spacy_model,
    get_spacy_model,
    ensure_english_model,
    ensure_japanese_model,
    get_english_model,
    get_japanese_model,
    get_available_models,
    clear_model_cache,
    pytest_skip_if_no_model,
    pytest_skip_if_no_english,
    pytest_skip_if_no_japanese,
)

__all__ = [
    "ensure_spacy_model",
    "get_spacy_model", 
    "ensure_english_model",
    "ensure_japanese_model",
    "get_english_model",
    "get_japanese_model",
    "get_available_models",
    "clear_model_cache",
    "pytest_skip_if_no_model",
    "pytest_skip_if_no_english", 
    "pytest_skip_if_no_japanese",
]