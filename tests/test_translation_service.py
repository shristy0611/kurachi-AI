"""
Tests for Local Translation Service
"""
import pytest
from unittest.mock import Mock, patch
from services.translation_service import (
    LocalTranslationService, 
    Language, 
    TranslationQuality,
    translation_service
)


class TestLocalTranslationService:
    """Test cases for LocalTranslationService"""
    
    @pytest.fixture
    def service(self):
        """Create a translation service instance for testing"""
        with patch('services.translation_service.Ollama') as mock_ollama:
            mock_llm = Mock()
            mock_ollama.return_value = mock_llm
            service = LocalTranslationService()
            service.llm = mock_llm
            return service
    
    def test_language_detection_japanese(self, service):
        """Test Japanese language detection"""
        japanese_text = "こんにちは、これは日本語のテストです。"
        result = service.detect_language(japanese_text)
        assert result == Language.JAPANESE
    
    def test_language_detection_english(self, service):
        """Test English language detection"""
        english_text = "Hello, this is an English test."
        result = service.detect_language(english_text)
        assert result == Language.ENGLISH
    
    def test_language_detection_mixed(self, service):
        """Test mixed language detection"""
        mixed_text = "Hello こんにちは world 世界"
        result = service.detect_language(mixed_text)
        # Should detect as Japanese due to Japanese characters
        assert result == Language.JAPANESE
    
    def test_language_detection_empty(self, service):
        """Test empty text detection"""
        result = service.detect_language("")
        assert result == Language.ENGLISH  # Default fallback
    
    def test_character_based_detection_japanese(self, service):
        """Test character-based Japanese detection"""
        japanese_text = "日本語のテスト"
        result = service._character_based_detection(japanese_text)
        assert result == Language.JAPANESE
    
    def test_character_based_detection_english(self, service):
        """Test character-based English detection"""
        english_text = "English test text"
        result = service._character_based_detection(english_text)
        assert result == Language.ENGLISH
    
    def test_translate_japanese_to_english(self, service):
        """Test Japanese to English translation"""
        service.llm.invoke.return_value = "Hello, this is a test."
        
        result = service.translate(
            "こんにちは、これはテストです。",
            Language.ENGLISH,
            Language.JAPANESE,
            TranslationQuality.BUSINESS
        )
        
        assert result["translated_text"] == "Hello, this is a test."
        assert result["source_language"] == "ja"
        assert result["target_language"] == "en"
        assert result["confidence"] > 0
        assert not result["skipped"]
    
    def test_translate_english_to_japanese(self, service):
        """Test English to Japanese translation"""
        service.llm.invoke.return_value = "こんにちは、これはテストです。"
        
        result = service.translate(
            "Hello, this is a test.",
            Language.JAPANESE,
            Language.ENGLISH,
            TranslationQuality.BUSINESS
        )
        
        assert result["translated_text"] == "こんにちは、これはテストです。"
        assert result["source_language"] == "en"
        assert result["target_language"] == "ja"
        assert result["confidence"] > 0
        assert not result["skipped"]
    
    def test_translate_same_language_skip(self, service):
        """Test translation skipping when source and target are the same"""
        result = service.translate(
            "Hello, this is a test.",
            Language.ENGLISH,
            Language.ENGLISH
        )
        
        assert result["translated_text"] == "Hello, this is a test."
        assert result["skipped"] == True
        assert result["confidence"] == 1.0
    
    def test_translate_auto_detect(self, service):
        """Test translation with auto language detection"""
        with patch.object(service, 'detect_language', return_value=Language.JAPANESE):
            service.llm.invoke.return_value = "Hello, this is a test."
            
            result = service.translate(
                "こんにちは、これはテストです。",
                Language.ENGLISH,
                Language.AUTO
            )
            
            assert result["source_language"] == "ja"
            assert result["target_language"] == "en"
    
    def test_translate_with_context(self, service):
        """Test translation with context"""
        service.llm.invoke.return_value = "Business meeting tomorrow"
        
        result = service.translate(
            "明日の会議",
            Language.ENGLISH,
            Language.JAPANESE,
            context="Business context"
        )
        
        # Verify the prompt included context
        call_args = service.llm.invoke.call_args[0][0]
        assert "Context: Business context" in call_args
    
    def test_translate_different_quality_levels(self, service):
        """Test translation with different quality levels"""
        service.llm.invoke.return_value = "Test translation"
        
        # Test basic quality
        result_basic = service.translate(
            "テスト",
            Language.ENGLISH,
            Language.JAPANESE,
            TranslationQuality.BASIC
        )
        
        # Test business quality
        result_business = service.translate(
            "テスト",
            Language.ENGLISH,
            Language.JAPANESE,
            TranslationQuality.BUSINESS
        )
        
        # Test technical quality
        result_technical = service.translate(
            "テスト",
            Language.ENGLISH,
            Language.JAPANESE,
            TranslationQuality.TECHNICAL
        )
        
        assert result_basic["quality_level"] == "basic"
        assert result_business["quality_level"] == "business"
        assert result_technical["quality_level"] == "technical"
    
    def test_translate_error_handling(self, service):
        """Test translation error handling"""
        service.llm.invoke.side_effect = Exception("LLM error")
        
        result = service.translate(
            "テスト",
            Language.ENGLISH,
            Language.JAPANESE
        )
        
        assert result["translated_text"] == "テスト"  # Original text returned
        assert result["confidence"] == 0.0
        assert "error" in result
    
    def test_clean_translation(self, service):
        """Test translation cleaning"""
        # Test removing common LLM artifacts
        dirty_translation = "Translation: Hello world"
        clean = service._clean_translation(dirty_translation)
        assert clean == "Hello world"
        
        dirty_translation = "Here is the translation: こんにちは世界"
        clean = service._clean_translation(dirty_translation)
        assert clean == "こんにちは世界"
    
    def test_confidence_calculation(self, service):
        """Test confidence score calculation"""
        # Test normal translation
        confidence = service._calculate_confidence(
            "Hello world", 
            "こんにちは世界", 
            Language.ENGLISH, 
            Language.JAPANESE
        )
        assert 0.0 <= confidence <= 1.0
        
        # Test Japanese target with Japanese characters
        confidence = service._calculate_confidence(
            "Hello", 
            "こんにちは", 
            Language.ENGLISH, 
            Language.JAPANESE
        )
        assert confidence > 0.7  # Should be high due to Japanese characters
    
    def test_build_translation_prompt_japanese_to_english(self, service):
        """Test Japanese to English prompt building"""
        prompt = service._build_translation_prompt(
            "こんにちは",
            Language.JAPANESE,
            Language.ENGLISH,
            TranslationQuality.BUSINESS
        )
        
        assert "Japanese-English translator" in prompt
        assert "professional, business-appropriate translation" in prompt
        assert "keigo (honorific language)" in prompt
        assert "こんにちは" in prompt
    
    def test_build_translation_prompt_english_to_japanese(self, service):
        """Test English to Japanese prompt building"""
        prompt = service._build_translation_prompt(
            "Hello",
            Language.ENGLISH,
            Language.JAPANESE,
            TranslationQuality.BUSINESS
        )
        
        assert "English-Japanese translator" in prompt
        assert "Use appropriate keigo" in prompt
        assert "Hello" in prompt
    
    def test_translate_document_content(self, service):
        """Test document content translation"""
        service.llm.invoke.return_value = "Translated content"
        
        documents = [
            {
                "content": "Original content",
                "metadata": {"title": "Test Doc"}
            }
        ]
        
        result = service.translate_document_content(documents, Language.JAPANESE)
        
        assert len(result) == 1
        assert result[0]["content"] == "Translated content"
        assert result[0]["metadata"]["original_content"] == "Original content"
        assert result[0]["metadata"]["is_translated"] == True
    
    def test_get_supported_languages(self, service):
        """Test getting supported languages"""
        languages = service.get_supported_languages()
        
        assert len(languages) == 2
        assert any(lang["code"] == "ja" for lang in languages)
        assert any(lang["code"] == "en" for lang in languages)
        assert any(lang["native_name"] == "日本語" for lang in languages)


class TestTranslationPrompts:
    """Test translation prompt generation"""
    
    @pytest.fixture
    def service(self):
        with patch('services.translation_service.Ollama'):
            return LocalTranslationService()
    
    def test_business_document_prompts(self, service):
        """Test business document specific prompts"""
        prompt = service._build_translation_prompt(
            "会議の議事録",
            Language.JAPANESE,
            Language.ENGLISH,
            TranslationQuality.BUSINESS,
            context="Meeting minutes"
        )
        
        assert "professional, business-appropriate translation" in prompt
        assert "Context: Meeting minutes" in prompt
        assert "keigo (honorific language)" in prompt
    
    def test_technical_document_prompts(self, service):
        """Test technical document specific prompts"""
        prompt = service._build_translation_prompt(
            "API documentation",
            Language.ENGLISH,
            Language.JAPANESE,
            TranslationQuality.TECHNICAL
        )
        
        assert "accurate technical translation" in prompt
        assert "technical terms and specialized vocabulary" in prompt


class TestTranslationAccuracy:
    """Test translation accuracy with sample business documents"""
    
    @pytest.fixture
    def service(self):
        with patch('services.translation_service.Ollama'):
            return LocalTranslationService()
    
    def test_business_email_translation(self, service):
        """Test business email translation"""
        service.llm.invoke.return_value = "Thank you for your hard work. Please review the attached document."
        
        japanese_email = "お疲れ様です。添付の資料をご確認ください。"
        
        result = service.translate(
            japanese_email,
            Language.ENGLISH,
            Language.JAPANESE,
            TranslationQuality.BUSINESS
        )
        
        # Verify business context is maintained
        assert result["quality_level"] == "business"
        assert result["confidence"] > 0
    
    def test_technical_document_translation(self, service):
        """Test technical document translation"""
        service.llm.invoke.return_value = "データベース接続エラーが発生しました"
        
        technical_text = "Database connection error occurred"
        
        result = service.translate(
            technical_text,
            Language.JAPANESE,
            Language.ENGLISH,
            TranslationQuality.TECHNICAL
        )
        
        assert result["quality_level"] == "technical"
        assert result["confidence"] > 0


# Integration test with global service instance
def test_global_translation_service():
    """Test the global translation service instance"""
    assert translation_service is not None
    assert hasattr(translation_service, 'translate')
    assert hasattr(translation_service, 'detect_language')
    
    # Test supported languages
    languages = translation_service.get_supported_languages()
    assert len(languages) == 2