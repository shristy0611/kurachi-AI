#!/usr/bin/env python3
"""
Test script for Multilingual Conversation Interface
Tests language selection, preference management, cross-language query processing,
and cultural adaptation features
"""
import sys
import os
import pytest

# Mark as slow due to multilingual service initialization
pytestmark = pytest.mark.slow

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from services.multilingual_conversation_interface import (
    multilingual_interface,
    UserLanguagePreferences,
    UILanguage,
    ResponseLanguage,
    CulturalAdaptation
)
from services.language_detection import language_detection_service
from utils.logger import get_logger

logger = get_logger("test_multilingual_interface")


def test_user_preferences():
    """Test user language preferences management"""
    print("🧪 Testing User Language Preferences...")
    
    test_user_id = "test_user_123"
    
    # Test default preferences
    preferences = multilingual_interface.get_user_preferences(test_user_id)
    print(f"✅ Default preferences loaded: UI={preferences.ui_language.value}, Response={preferences.response_language.value}")
    
    # Test updating preferences
    preferences.ui_language = UILanguage.JAPANESE
    preferences.response_language = ResponseLanguage.JAPANESE
    preferences.cultural_adaptation = CulturalAdaptation.BUSINESS
    preferences.auto_translate_queries = True
    preferences.translation_quality_threshold = 0.8
    
    success = multilingual_interface.save_user_preferences(preferences)
    print(f"✅ Preferences saved: {success}")
    
    # Test loading updated preferences
    loaded_preferences = multilingual_interface.get_user_preferences(test_user_id)
    print(f"✅ Updated preferences loaded: UI={loaded_preferences.ui_language.value}, Response={loaded_preferences.response_language.value}")
    
    return success


def test_ui_text_localization():
    """Test UI text localization"""
    print("\n🧪 Testing UI Text Localization...")
    
    # Test English UI text
    en_text = multilingual_interface.get_ui_text("search_placeholder", "en")
    print(f"✅ English UI text: '{en_text}'")
    
    # Test Japanese UI text
    ja_text = multilingual_interface.get_ui_text("search_placeholder", "ja")
    print(f"✅ Japanese UI text: '{ja_text}'")
    
    # Test parameter substitution
    page_ref = multilingual_interface.get_ui_text("page_reference", "en", page=42)
    print(f"✅ Parameter substitution: '{page_ref}'")
    
    return True


def test_multilingual_query_processing():
    """Test multilingual query processing"""
    print("\n🧪 Testing Multilingual Query Processing...")
    
    test_user_id = "test_user_456"
    
    # Set up user preferences for testing
    preferences = multilingual_interface.get_user_preferences(test_user_id)
    preferences.auto_translate_queries = True
    preferences.fallback_languages = ["ja", "en"]
    multilingual_interface.save_user_preferences(preferences)
    
    # Test English query
    english_query = "What is the company's revenue for Q3?"
    en_context = multilingual_interface.process_multilingual_query(english_query, test_user_id)
    print(f"✅ English query processed: detected={en_context.detected_language}, response_lang={en_context.response_language}")
    print(f"   Search languages: {en_context.search_languages}")
    
    # Test Japanese query
    japanese_query = "第3四半期の売上はいくらですか？"
    ja_context = multilingual_interface.process_multilingual_query(japanese_query, test_user_id)
    print(f"✅ Japanese query processed: detected={ja_context.detected_language}, response_lang={ja_context.response_language}")
    print(f"   Search languages: {ja_context.search_languages}")
    
    return True


def test_error_messages():
    """Test localized error messages"""
    print("\n🧪 Testing Error Messages...")
    
    test_user_id = "test_user_789"
    
    # Test English error messages
    en_error = multilingual_interface.get_error_message("translation_failed", "en")
    print(f"✅ English error message: '{en_error[:50]}...'")
    
    # Test Japanese error messages
    ja_error = multilingual_interface.get_error_message("translation_failed", "ja")
    print(f"✅ Japanese error message: '{ja_error[:50]}...'")
    
    # Test error with parameters
    param_error = multilingual_interface.get_error_message("no_results", "en", query="test query")
    print(f"✅ Parameterized error: '{param_error[:50]}...'")
    
    return True


def test_help_text():
    """Test localized help text"""
    print("\n🧪 Testing Help Text...")
    
    # Test English help
    en_help = multilingual_interface.get_help_text("language_selection", "en")
    print(f"✅ English help text: '{en_help[:50]}...'")
    
    # Test Japanese help
    ja_help = multilingual_interface.get_help_text("language_selection", "ja")
    print(f"✅ Japanese help text: '{ja_help[:50]}...'")
    
    return True


def test_supported_languages():
    """Test supported languages list"""
    print("\n🧪 Testing Supported Languages...")
    
    languages = multilingual_interface.get_supported_languages()
    print(f"✅ Supported languages: {len(languages)} languages")
    
    for lang in languages:
        print(f"   {lang['flag']} {lang['native_name']} ({lang['code']}) - UI: {lang['ui_support']}, Translation: {lang['translation_support']}")
    
    return len(languages) > 0


def test_multilingual_response_formatting():
    """Test multilingual response formatting"""
    print("\n🧪 Testing Multilingual Response Formatting...")
    
    test_user_id = "test_user_response"
    
    # Set up preferences
    preferences = multilingual_interface.get_user_preferences(test_user_id)
    preferences.response_language = ResponseLanguage.ENGLISH
    preferences.auto_translate_responses = True
    preferences.show_original_text = True
    multilingual_interface.save_user_preferences(preferences)
    
    # Create test query context
    query_context = multilingual_interface.process_multilingual_query("テストクエリ", test_user_id)
    
    # Test response formatting
    test_response = "これはテスト応答です。"
    test_sources = [
        {
            "content": "テスト文書の内容",
            "filename": "test_document.pdf",
            "page_number": 1,
            "detected_language": "ja"
        }
    ]
    
    formatted_response = multilingual_interface.format_multilingual_response(
        test_response, test_sources, query_context
    )
    
    print(f"✅ Response formatted: language={formatted_response.language}")
    print(f"   Original content available: {formatted_response.original_content is not None}")
    print(f"   Sources processed: {len(formatted_response.sources)}")
    print(f"   Confidence: {formatted_response.confidence:.2f}")
    
    return True


def test_language_detection_integration():
    """Test integration with language detection service"""
    print("\n🧪 Testing Language Detection Integration...")
    
    # Test various text samples
    test_samples = [
        ("Hello, how are you today?", "en"),
        ("こんにちは、今日はいかがですか？", "ja"),
        ("Hello こんにちは mixed text", "mixed"),
        ("", "unknown")
    ]
    
    for text, expected in test_samples:
        if text:
            detection = language_detection_service.detect_document_language(text, detailed=False)
            detected = detection.primary_language.value
            print(f"✅ Text: '{text[:30]}...' -> Detected: {detected} (Expected: {expected})")
        else:
            print(f"✅ Empty text -> Expected: {expected}")
    
    return True


def run_all_tests():
    """Run all multilingual conversation interface tests"""
    print("🚀 Starting Multilingual Conversation Interface Tests\n")
    
    tests = [
        ("User Preferences", test_user_preferences),
        ("UI Text Localization", test_ui_text_localization),
        ("Multilingual Query Processing", test_multilingual_query_processing),
        ("Error Messages", test_error_messages),
        ("Help Text", test_help_text),
        ("Supported Languages", test_supported_languages),
        ("Response Formatting", test_multilingual_response_formatting),
        ("Language Detection Integration", test_language_detection_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ {test_name}: FAILED - {e}")
    
    # Summary
    print(f"\n📊 Test Summary:")
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {passed/total*100:.1f}%")
    
    if passed < total:
        print("\n❌ Failed Tests:")
        for test_name, result, error in results:
            if not result:
                print(f"   - {test_name}: {error or 'Unknown error'}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)