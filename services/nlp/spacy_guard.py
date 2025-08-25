"""
spaCy Model Guard Utility
Provides graceful degradation when spaCy models are unavailable.
"""
import logging
from typing import Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Global model cache to avoid repeated loading attempts
_model_cache = {}
_model_availability = {}


def ensure_spacy_model(name: str) -> Tuple[bool, str]:
    """
    Check spaCy model availability and return status with message.
    
    Args:
        name: spaCy model name (e.g., 'en_core_web_sm', 'ja_core_news_sm')
        
    Returns:
        Tuple of (success_bool, message_string)
        - success_bool: True if model is available, False otherwise
        - message_string: Descriptive message about model status
    """
    # Check cache first
    if name in _model_availability:
        success, message = _model_availability[name]
        return success, message
    
    try:
        import spacy
        
        # Attempt to load the model
        nlp = spacy.load(name)
        
        # Cache successful load
        _model_cache[name] = nlp
        _model_availability[name] = (True, f"spaCy model '{name}' loaded successfully")
        
        logger.info(f"spaCy model '{name}' loaded and cached")
        return True, f"spaCy model '{name}' loaded successfully"
        
    except ImportError:
        message = "spaCy not installed. Install with: pip install spacy"
        _model_availability[name] = (False, message)
        logger.warning(message)
        return False, message
        
    except OSError as e:
        if "Can't find model" in str(e):
            message = f"spaCy model '{name}' not found. Install with: python -m spacy download {name}"
        else:
            message = f"spaCy model '{name}' failed to load: {e}"
        
        _model_availability[name] = (False, message)
        logger.warning(message)
        return False, message
        
    except Exception as e:
        message = f"Unexpected error loading spaCy model '{name}': {e}"
        _model_availability[name] = (False, message)
        logger.error(message)
        return False, message


def get_spacy_model(name: str) -> Optional[Any]:
    """
    Get a loaded spaCy model if available, None otherwise.
    
    Args:
        name: spaCy model name
        
    Returns:
        Loaded spaCy model or None if unavailable
    """
    success, message = ensure_spacy_model(name)
    
    if success:
        return _model_cache.get(name)
    else:
        logger.debug(f"spaCy model '{name}' not available: {message}")
        return None


def get_available_models() -> dict:
    """
    Get status of all checked models.
    
    Returns:
        Dictionary mapping model names to (available, message) tuples
    """
    return _model_availability.copy()


def clear_model_cache():
    """Clear the model cache (useful for testing)."""
    global _model_cache, _model_availability
    _model_cache.clear()
    _model_availability.clear()
    logger.debug("spaCy model cache cleared")


# Convenience functions for common models
def ensure_english_model() -> Tuple[bool, str]:
    """Ensure English spaCy model is available."""
    return ensure_spacy_model("en_core_web_sm")


def ensure_japanese_model() -> Tuple[bool, str]:
    """Ensure Japanese spaCy model is available."""
    return ensure_spacy_model("ja_core_news_sm")


def get_english_model() -> Optional[Any]:
    """Get English spaCy model if available."""
    return get_spacy_model("en_core_web_sm")


def get_japanese_model() -> Optional[Any]:
    """Get Japanese spaCy model if available."""
    return get_spacy_model("ja_core_news_sm")


# Pytest integration helpers
def pytest_skip_if_no_model(model_name: str):
    """
    Pytest skip decorator helper for missing spaCy models.
    
    Usage:
        @pytest.mark.skipif(*pytest_skip_if_no_model("en_core_web_sm"))
        def test_english_nlp():
            pass
    """
    success, message = ensure_spacy_model(model_name)
    return not success, f"spaCy model '{model_name}' not available: {message}"


def pytest_skip_if_no_english():
    """Pytest skip helper for English model."""
    return pytest_skip_if_no_model("en_core_web_sm")


def pytest_skip_if_no_japanese():
    """Pytest skip helper for Japanese model."""
    return pytest_skip_if_no_model("ja_core_news_sm")