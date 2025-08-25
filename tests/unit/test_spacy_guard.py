#!/usr/bin/env python3
"""
Unit tests for spaCy guard system
Tests graceful degradation when spaCy models are unavailable.
"""
import pytest
import sys
from unittest.mock import patch, MagicMock, Mock
from services.nlp.spacy_guard import (
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


class TestSpacyGuard:
    """Test spaCy guard functionality"""
    
    def setup_method(self):
        """Clear cache before each test"""
        clear_model_cache()
    
    def test_ensure_spacy_model_success(self):
        """Test successful model loading"""
        mock_nlp = MagicMock()
        mock_spacy = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        
        with patch.dict('sys.modules', {'spacy': mock_spacy}):
            success, message = ensure_spacy_model("en_core_web_sm")
            
            assert success is True
            assert "loaded successfully" in message
            mock_spacy.load.assert_called_once_with("en_core_web_sm")
    
    def test_ensure_spacy_model_import_error(self):
        """Test handling when spaCy is not installed"""
        # Remove spacy from sys.modules if it exists, and prevent import
        with patch.dict('sys.modules', {'spacy': None}):
            success, message = ensure_spacy_model("en_core_web_sm")
            
            assert success is False
            assert "spaCy not installed" in message
            assert "pip install spacy" in message
    
    def test_ensure_spacy_model_model_not_found(self):
        """Test handling when model is not found"""
        mock_spacy = MagicMock()
        mock_spacy.load.side_effect = OSError("Can't find model 'en_core_web_sm'")
        
        with patch.dict('sys.modules', {'spacy': mock_spacy}):
            success, message = ensure_spacy_model("en_core_web_sm")
            
            assert success is False
            assert "not found" in message
            assert "python -m spacy download" in message
    
    def test_ensure_spacy_model_other_os_error(self):
        """Test handling of other OS errors"""
        mock_spacy = MagicMock()
        mock_spacy.load.side_effect = OSError("Permission denied")
        
        with patch.dict('sys.modules', {'spacy': mock_spacy}):
            success, message = ensure_spacy_model("en_core_web_sm")
            
            assert success is False
            assert "failed to load" in message
            assert "Permission denied" in message
    
    def test_ensure_spacy_model_unexpected_error(self):
        """Test handling of unexpected errors"""
        mock_spacy = MagicMock()
        mock_spacy.load.side_effect = ValueError("Unexpected error")
        
        with patch.dict('sys.modules', {'spacy': mock_spacy}):
            success, message = ensure_spacy_model("en_core_web_sm")
            
            assert success is False
            assert "Unexpected error" in message
    
    def test_model_caching(self):
        """Test that models are cached after first load"""
        mock_nlp = MagicMock()
        mock_spacy = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        
        with patch.dict('sys.modules', {'spacy': mock_spacy}):
            # First call should load model
            success1, message1 = ensure_spacy_model("en_core_web_sm")
            assert success1 is True
            
            # Second call should use cache
            success2, message2 = ensure_spacy_model("en_core_web_sm")
            assert success2 is True
            
            # spacy.load should only be called once
            mock_spacy.load.assert_called_once()
    
    def test_get_spacy_model_success(self):
        """Test getting a loaded model"""
        mock_nlp = MagicMock()
        mock_spacy = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        
        with patch.dict('sys.modules', {'spacy': mock_spacy}):
            model = get_spacy_model("en_core_web_sm")
            
            assert model is mock_nlp
    
    def test_get_spacy_model_failure(self):
        """Test getting a model that fails to load"""
        mock_spacy = MagicMock()
        mock_spacy.load.side_effect = OSError("Can't find model")
        
        with patch.dict('sys.modules', {'spacy': mock_spacy}):
            model = get_spacy_model("en_core_web_sm")
            
            assert model is None
    
    def test_convenience_functions(self):
        """Test convenience functions for common models"""
        mock_nlp = MagicMock()
        mock_spacy = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        
        with patch.dict('sys.modules', {'spacy': mock_spacy}):
            # Test English model functions
            success, message = ensure_english_model()
            assert success is True
            
            model = get_english_model()
            assert model is mock_nlp
            
            # Test Japanese model functions
            success, message = ensure_japanese_model()
            assert success is True
            
            model = get_japanese_model()
            assert model is mock_nlp
    
    def test_get_available_models(self):
        """Test getting status of all checked models"""
        mock_nlp = MagicMock()
        mock_spacy = MagicMock()
        
        with patch.dict('sys.modules', {'spacy': mock_spacy}):
            # Load one model successfully
            mock_spacy.load.return_value = mock_nlp
            ensure_spacy_model("en_core_web_sm")
            
            # Fail to load another
            mock_spacy.load.side_effect = OSError("Can't find model")
            ensure_spacy_model("ja_core_news_sm")
            
            available = get_available_models()
            
            assert "en_core_web_sm" in available
            assert "ja_core_news_sm" in available
            assert available["en_core_web_sm"][0] is True  # Success
            assert available["ja_core_news_sm"][0] is False  # Failure
    
    def test_clear_model_cache(self):
        """Test clearing the model cache"""
        mock_nlp = MagicMock()
        mock_spacy = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        
        with patch.dict('sys.modules', {'spacy': mock_spacy}):
            # Load a model
            ensure_spacy_model("en_core_web_sm")
            
            # Verify it's cached
            available = get_available_models()
            assert "en_core_web_sm" in available
            
            # Clear cache
            clear_model_cache()
            
            # Verify cache is empty
            available = get_available_models()
            assert len(available) == 0
    
    def test_pytest_skip_helpers(self):
        """Test pytest skip helper functions"""
        # Test with available model
        mock_nlp = MagicMock()
        mock_spacy = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        
        with patch.dict('sys.modules', {'spacy': mock_spacy}):
            should_skip, reason = pytest_skip_if_no_model("en_core_web_sm")
            assert should_skip is False
            
            should_skip, reason = pytest_skip_if_no_english()
            assert should_skip is False
        
        # Test with unavailable model
        mock_spacy_fail = MagicMock()
        mock_spacy_fail.load.side_effect = OSError("Can't find model")
        
        with patch.dict('sys.modules', {'spacy': mock_spacy_fail}):
            # Clear cache to force reload
            clear_model_cache()
            
            should_skip, reason = pytest_skip_if_no_model("missing_model")
            assert should_skip is True
            assert "not available" in reason
            
            should_skip, reason = pytest_skip_if_no_japanese()
            assert should_skip is True
            assert "not available" in reason


@pytest.mark.nlp
class TestSpacyGuardIntegration:
    """Integration tests that require actual spaCy installation"""
    
    def test_real_spacy_model_detection(self):
        """Test with real spaCy installation if available"""
        # This test will be skipped if spaCy is not installed
        try:
            import spacy
            
            # Try to detect a common model
            success, message = ensure_spacy_model("en_core_web_sm")
            
            # If spaCy is installed but model is missing, we should get a helpful message
            if not success:
                assert "not found" in message or "Can't find model" in message
                assert "python -m spacy download" in message
            else:
                assert "loaded successfully" in message
                
        except ImportError:
            pytest.skip("spaCy not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])