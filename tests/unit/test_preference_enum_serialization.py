#!/usr/bin/env python3
"""
Unit tests for preference enum serialization and validation
Tests the robust handling of enum values in JSON serialization/deserialization
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import unittest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from services.multilingual_conversation_interface import (
    UserLanguagePreferences, UILanguage, ResponseLanguage, CulturalAdaptation
)
from services.preference_manager import preference_manager, PreferenceValidationError
from models.database import _json_safe, _normalize_enums, _validate_preference_schema
from utils.logger import get_logger

logger = get_logger("test_preference_enum")


class TestEnumSerialization(unittest.TestCase):
    """Test enum serialization and deserialization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_preferences = UserLanguagePreferences(
            user_id="test_user_enum",
            ui_language=UILanguage.JAPANESE,
            response_language=ResponseLanguage.AUTO,
            cultural_adaptation=CulturalAdaptation.BUSINESS,
            auto_translate_queries=True,
            translation_quality_threshold=0.85,
            fallback_languages=["ja", "en"]
        )
    
    def test_json_safe_enum_serialization(self):
        """Test _json_safe function with enum values"""
        # Test enum serialization
        result = _json_safe(UILanguage.JAPANESE)
        self.assertEqual(result, "ja")
        
        result = _json_safe(ResponseLanguage.AUTO)
        self.assertEqual(result, "auto")
        
        result = _json_safe(CulturalAdaptation.BUSINESS)
        self.assertEqual(result, "business")
        
        # Test error for non-enum types
        with self.assertRaises(TypeError):
            _json_safe({"not": "enum"})
    
    def test_normalize_enums_function(self):
        """Test _normalize_enums function with nested data"""
        test_data = {
            "ui_language": UILanguage.JAPANESE,
            "response_language": ResponseLanguage.AUTO,
            "nested": {
                "cultural_adaptation": CulturalAdaptation.BUSINESS
            },
            "list_with_enums": [UILanguage.ENGLISH, "string", 42],
            "regular_string": "test"
        }
        
        normalized = _normalize_enums(test_data)
        
        self.assertEqual(normalized["ui_language"], "ja")
        self.assertEqual(normalized["response_language"], "auto")
        self.assertEqual(normalized["nested"]["cultural_adaptation"], "business")
        self.assertEqual(normalized["list_with_enums"], ["en", "string", 42])
        self.assertEqual(normalized["regular_string"], "test")
    
    def test_preference_schema_validation(self):
        """Test preference schema validation"""
        # Valid preferences
        valid_prefs = {
            "ui_language": "ja",
            "response_language": "auto",
            "cultural_adaptation": "business",
            "translation_quality_threshold": 0.75
        }
        self.assertTrue(_validate_preference_schema(valid_prefs, "language"))
        
        # Missing required field
        invalid_prefs = {
            "ui_language": "ja",
            "response_language": "auto"
            # Missing cultural_adaptation
        }
        self.assertFalse(_validate_preference_schema(invalid_prefs, "language"))
        
        # Invalid enum value
        invalid_prefs = {
            "ui_language": "invalid_language",
            "response_language": "auto",
            "cultural_adaptation": "business"
        }
        self.assertFalse(_validate_preference_schema(invalid_prefs, "language"))
        
        # Invalid threshold
        invalid_prefs = {
            "ui_language": "ja",
            "response_language": "auto",
            "cultural_adaptation": "business",
            "translation_quality_threshold": 1.5  # Out of range
        }
        self.assertFalse(_validate_preference_schema(invalid_prefs, "language"))
    
    def test_full_serialization_cycle(self):
        """Test complete serialization and deserialization cycle"""
        # Convert to dict
        prefs_dict = asdict(self.test_preferences)
        prefs_dict.pop('user_id')
        
        # Normalize enums
        normalized = _normalize_enums(prefs_dict)
        
        # Serialize to JSON
        json_str = json.dumps(normalized, default=_json_safe)
        self.assertIsInstance(json_str, str)
        self.assertIn('"ja"', json_str)
        self.assertIn('"auto"', json_str)
        self.assertIn('"business"', json_str)
        
        # Deserialize from JSON
        loaded_data = json.loads(json_str)
        
        # Validate schema
        self.assertTrue(_validate_preference_schema(loaded_data, "language"))
        
        # Convert back to enums
        loaded_data['ui_language'] = UILanguage(loaded_data['ui_language'])
        loaded_data['response_language'] = ResponseLanguage(loaded_data['response_language'])
        loaded_data['cultural_adaptation'] = CulturalAdaptation(loaded_data['cultural_adaptation'])
        
        # Create new preferences object
        restored_prefs = UserLanguagePreferences(
            user_id="test_user_enum",
            **loaded_data
        )
        
        # Verify values match
        self.assertEqual(restored_prefs.ui_language, self.test_preferences.ui_language)
        self.assertEqual(restored_prefs.response_language, self.test_preferences.response_language)
        self.assertEqual(restored_prefs.cultural_adaptation, self.test_preferences.cultural_adaptation)
        self.assertEqual(restored_prefs.translation_quality_threshold, self.test_preferences.translation_quality_threshold)


class TestPreferenceManager(unittest.TestCase):
    """Test enhanced preference manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_user_id = "test_user_manager"
        preference_manager.clear_cache()
    
    def test_default_preference_creation(self):
        """Test default preference creation for different user types"""
        # English user
        en_prefs = preference_manager._create_default_preferences("english_user")
        self.assertEqual(en_prefs.ui_language, UILanguage.ENGLISH)
        self.assertIn("en", en_prefs.fallback_languages)
        
        # Japanese user (with Japanese characters in ID)
        ja_prefs = preference_manager._create_default_preferences("日本語ユーザー")
        self.assertEqual(ja_prefs.ui_language, UILanguage.JAPANESE)
        self.assertIn("ja", ja_prefs.fallback_languages)
    
    def test_preference_validation(self):
        """Test preference validation"""
        # Valid preferences
        valid_prefs = UserLanguagePreferences(
            user_id="test_user",
            ui_language=UILanguage.ENGLISH,
            response_language=ResponseLanguage.AUTO,
            cultural_adaptation=CulturalAdaptation.BASIC,
            translation_quality_threshold=0.75,
            fallback_languages=["en", "ja"]
        )
        self.assertTrue(preference_manager._validate_preferences(valid_prefs))
        
        # Invalid preferences - missing user_id
        invalid_prefs = UserLanguagePreferences(
            user_id="",  # Empty user_id
            ui_language=UILanguage.ENGLISH,
            response_language=ResponseLanguage.AUTO,
            cultural_adaptation=CulturalAdaptation.BASIC
        )
        self.assertFalse(preference_manager._validate_preferences(invalid_prefs))
        
        # Invalid preferences - bad threshold
        invalid_prefs = UserLanguagePreferences(
            user_id="test_user",
            ui_language=UILanguage.ENGLISH,
            response_language=ResponseLanguage.AUTO,
            cultural_adaptation=CulturalAdaptation.BASIC,
            translation_quality_threshold=2.0,  # Out of range
            fallback_languages=["en", "ja"]
        )
        self.assertFalse(preference_manager._validate_preferences(invalid_prefs))
    
    def test_mixed_language_detection(self):
        """Test enhanced mixed language detection"""
        # Clear mixed language case
        mixed_scores = {"en": 0.6, "ja": 0.35}
        text = "This is a mixed language text with some 日本語 content that should be detected properly."
        
        is_mixed = preference_manager.is_mixed_language_robust(text, mixed_scores)
        self.assertTrue(is_mixed)
        
        # Single language case
        single_scores = {"en": 0.95, "ja": 0.05}
        is_mixed = preference_manager.is_mixed_language_robust(text, single_scores)
        self.assertFalse(is_mixed)
        
        # Short text case
        short_text = "Short"
        is_mixed = preference_manager.is_mixed_language_robust(short_text, mixed_scores)
        self.assertFalse(is_mixed)
    
    def test_mixed_language_response_handling(self):
        """Test mixed language response handling"""
        # Mock user preferences
        with patch.object(preference_manager, 'get_user_preferences') as mock_get_prefs:
            mock_prefs = UserLanguagePreferences(
                user_id="test_user",
                ui_language=UILanguage.JAPANESE,
                response_language=ResponseLanguage.AUTO
            )
            mock_get_prefs.return_value = mock_prefs
            
            # High confidence primary language
            response_lang = preference_manager.handle_mixed_language_response(
                "test_user", ["en", "ja"], [0.8, 0.2]
            )
            self.assertEqual(response_lang, "en")
            
            # Low confidence - should fall back to UI language
            response_lang = preference_manager.handle_mixed_language_response(
                "test_user", ["en", "ja"], [0.6, 0.4]
            )
            self.assertEqual(response_lang, "ja")
    
    def test_cache_functionality(self):
        """Test preference caching"""
        # Mock database calls
        with patch('models.database.db_manager.get_user_preferences') as mock_get, \
             patch('models.database.db_manager.save_user_preferences') as mock_save:
            
            mock_get.return_value = None  # No existing preferences
            mock_save.return_value = True
            
            # First call should create defaults and cache
            prefs1 = preference_manager.get_user_preferences(self.test_user_id)
            self.assertEqual(mock_get.call_count, 1)
            
            # Second call should use cache
            prefs2 = preference_manager.get_user_preferences(self.test_user_id)
            self.assertEqual(mock_get.call_count, 1)  # No additional DB call
            self.assertEqual(prefs1.user_id, prefs2.user_id)
            
            # Clear cache and call again
            preference_manager.clear_cache(self.test_user_id)
            prefs3 = preference_manager.get_user_preferences(self.test_user_id)
            self.assertEqual(mock_get.call_count, 2)  # Additional DB call
    
    def test_export_import_preferences(self):
        """Test preference export and import"""
        # Create test preferences
        test_prefs = UserLanguagePreferences(
            user_id="export_test_user",
            ui_language=UILanguage.JAPANESE,
            response_language=ResponseLanguage.AUTO,
            cultural_adaptation=CulturalAdaptation.BUSINESS
        )
        
        # Mock save operation
        with patch.object(preference_manager, 'save_user_preferences', return_value=True):
            # Export preferences
            export_data = preference_manager.export_preferences("export_test_user")
            
            self.assertIn("user_id", export_data)
            self.assertIn("preferences", export_data)
            self.assertIn("exported_at", export_data)
            self.assertEqual(export_data["user_id"], "export_test_user")
            
            # Import preferences
            success = preference_manager.import_preferences(export_data)
            self.assertTrue(success)
    
    def test_metrics_tracking(self):
        """Test metrics tracking"""
        initial_metrics = preference_manager.get_metrics()
        initial_loads = initial_metrics.get("session_metrics", {}).get("load_operations", 0)
        
        # Mock database to force cache miss
        with patch('models.database.db_manager.get_user_preferences', return_value=None), \
             patch('models.database.db_manager.save_user_preferences', return_value=True), \
             patch('models.database.db_manager.increment_metric', return_value=True):
            
            # This should increment load operations and cache misses
            preference_manager.get_user_preferences("metrics_test_user", use_cache=False)
            
            updated_metrics = preference_manager.get_metrics()
            session_metrics = updated_metrics.get("session_metrics", {})
            self.assertEqual(session_metrics.get("load_operations", 0), initial_loads + 1)


def run_enum_serialization_tests():
    """Run all enum serialization tests"""
    print("🧪 Running Enum Serialization and Preference Management Tests\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestEnumSerialization))
    suite.addTests(loader.loadTestsFromTestCase(TestPreferenceManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"   - {test}: {error_msg}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"   - {test}: {error_msg}")
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_enum_serialization_tests()
    sys.exit(0 if success else 1)