#!/usr/bin/env python3
"""
Unit tests for validation system contracts
Ensures critical interfaces remain stable and prevent regressions
"""
import pytest
from unittest.mock import patch, MagicMock


class TestEnumCoercionContract:
    """Test enum coercion helper functions work correctly"""
    
    def test_translation_service_enum_coercion(self):
        """Test that translation service handles both strings and enums"""
        from services.translation_service import _as_enum, Language, TranslationQuality
        
        # Test string to enum coercion
        assert _as_enum("en", Language) == Language.ENGLISH
        assert _as_enum("ja", Language) == Language.JAPANESE
        assert _as_enum("es", Language) == Language.SPANISH
        
        # Test enum passthrough
        assert _as_enum(Language.ENGLISH, Language) == Language.ENGLISH
        
        # Test quality enum
        assert _as_enum("basic", TranslationQuality) == TranslationQuality.BASIC
        assert _as_enum("business", TranslationQuality) == TranslationQuality.BUSINESS
        
        # Test invalid values raise proper errors
        with pytest.raises(ValueError, match="Cannot coerce"):
            _as_enum("invalid", Language)
    
    def test_translation_service_accepts_mixed_inputs(self):
        """Test translation service accepts both string and enum inputs"""
        from services.translation_service import translation_service
        
        # Mock the actual translation to avoid LLM calls
        with patch.object(translation_service, '_build_translation_prompt') as mock_prompt, \
             patch.object(translation_service, 'llm') as mock_llm:
            
            mock_prompt.return_value = "test prompt"
            mock_llm.invoke.return_value = "Hola"
            
            # Test string inputs
            result = translation_service.translate("Hello", "es", "en", "basic")
            assert result is not None
            assert "translated_text" in result
            
            # Verify enum coercion was called correctly
            mock_prompt.assert_called_once()


class TestTextProcessorContract:
    """Test TXT file processor contract"""
    
    def test_txt_processor_selection(self):
        """Test that TXT processor is correctly selected for .txt files"""
        from services.document_processors import processor_factory
        
        # Test with proper filename
        processor = processor_factory.get_processor("test.txt")
        assert processor is not None
        assert processor.__class__.__name__ == "TextProcessor"
    
    def test_txt_processor_handles_octet_stream(self):
        """Test TXT processor handles application/octet-stream mime type"""
        from services.document_processors import TextProcessor
        
        processor = TextProcessor()
        
        # Test with octet-stream mime type (common for .txt files)
        assert processor.can_process("test.txt", "application/octet-stream")
        assert processor.can_process("test.txt", "text/plain")
        assert processor.can_process("test.txt", None)  # No mime type
        assert processor.can_process("test.txt", "")    # Empty mime type
    
    def test_txt_processor_fallback_processing(self):
        """Test TXT processor fallback when UnstructuredFileLoader fails"""
        from services.document_processors import TextProcessor
        import tempfile
        import os
        
        processor = TextProcessor()
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for validation")
            temp_file = f.name
        
        try:
            # Test processing works
            result = processor.process(temp_file)
            assert result.success
            assert len(result.documents) > 0
            assert "Test content" in result.documents[0].page_content
        finally:
            os.unlink(temp_file)


class TestDatabaseContract:
    """Test database manager contract"""
    
    def test_db_manager_has_get_connection(self):
        """Test that db_manager has get_connection method"""
        from models.database import db_manager
        
        assert hasattr(db_manager, 'get_connection')
        assert callable(db_manager.get_connection)
        
        # Test connection returns something
        conn = db_manager.get_connection()
        assert conn is not None
        conn.close()


class TestPreferenceManagerContract:
    """Test preference manager contract"""
    
    def test_preference_manager_has_get_preference(self):
        """Test that preference_manager has get_preference method"""
        from services.preference_manager import preference_manager
        
        assert hasattr(preference_manager, 'get_preference')
        assert callable(preference_manager.get_preference)
        
        # Test basic functionality
        result = preference_manager.get_preference("ui_language", "default")
        assert result is not None
        
        # Test default value
        result = preference_manager.get_preference("nonexistent_key", "default_value")
        assert result == "default_value"


class TestLanguageDetectionContract:
    """Test language detection service contract"""
    
    def test_language_detection_has_detect_language(self):
        """Test that language detection service has detect_language method"""
        from services.language_detection import language_detection_service
        
        assert hasattr(language_detection_service, 'detect_language')
        assert callable(language_detection_service.detect_language)
        
        # Test basic functionality
        result = language_detection_service.detect_language("Hello world")
        assert isinstance(result, str)
        assert result in ["en", "ja", "es", "fr", "de", "zh", "ko", "unknown"]


class TestConfigurationContract:
    """Test configuration system contract"""
    
    def test_config_has_expected_attributes(self):
        """Test that config module has expected attributes"""
        import config
        
        # Test main config object
        assert hasattr(config, 'config')
        assert hasattr(config, 'ai')
        assert hasattr(config, 'database')
        assert hasattr(config, 'app')
        assert hasattr(config, 'security')
        
        # Test config sections are accessible
        assert config.ai is not None
        assert config.database is not None
        assert config.app is not None
        assert config.security is not None


class TestTranslationCacheContract:
    """Test translation service cache behavior"""
    
    def test_translation_cold_path(self):
        """Test translation service when cache is empty"""
        from services.translation_service import translation_service
        
        # Clear any existing cache for this test
        cache_key = "test_cold_path"
        
        with patch.object(translation_service, '_get_cached_translation') as mock_cache, \
             patch.object(translation_service, 'llm') as mock_llm, \
             patch.object(translation_service, '_build_translation_prompt') as mock_prompt:
            
            mock_cache.return_value = None  # Simulate cache miss
            mock_prompt.return_value = "test prompt"
            mock_llm.invoke.return_value = "Test translation"
            
            result = translation_service.translate("Test", "en", "es")
            
            assert result is not None
            assert "translated_text" in result
            assert not result.get("cached", False)
    
    def test_translation_warm_path(self):
        """Test translation service when cache is warm"""
        from services.translation_service import translation_service
        
        cached_result = {
            "translated_text": "Cached translation",
            "source_language": "en",
            "target_language": "es",
            "confidence": 0.95,
            "method": "cached",
            "quality_level": "business",
            "cached": True
        }
        
        with patch.object(translation_service, '_get_cached_translation') as mock_cache:
            mock_cache.return_value = cached_result
            
            result = translation_service.translate("Test", "en", "es")
            
            assert result is not None
            assert result["cached"] is True
            assert result["translated_text"] == "Cached translation"


@pytest.mark.integration
class TestValidationSystemIntegration:
    """Integration tests for validation system components"""
    
    def test_full_import_validation(self):
        """Test that all critical imports work"""
        from tools.validation_system import ValidationSystem
        
        validator = ValidationSystem()
        results = validator.validate_imports()
        
        assert results["failed"] == 0, f"Import failures: {results['errors']}"
        assert results["passed"] > 0
    
    def test_functionality_validation_passes(self):
        """Test that functionality validation passes"""
        from tools.validation_system import ValidationSystem
        
        validator = ValidationSystem()
        results = validator.validate_functionality()
        
        # Allow some failures for optional services, but core should work
        assert results["passed"] >= 4, "Core functionality should pass"
    
    def test_test_runner_validation(self):
        """Test that test runner validation works"""
        from tools.validation_system import ValidationSystem
        
        validator = ValidationSystem()
        results = validator.validate_test_runners()
        
        assert results["failed"] == 0, f"Test runner failures: {results['errors']}"
        assert results["passed"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])