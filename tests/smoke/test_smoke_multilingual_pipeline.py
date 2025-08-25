#!/usr/bin/env python3
"""
End-to-End Smoke Test for Multilingual Pipeline
Tests the complete flow: UI prefs → glossary loading → translation → metrics tracking
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
from services.preference_manager import preference_manager
from services.intelligent_translation import IntelligentTranslationService, TranslationContext
from services.multilingual_conversation_interface import (
    multilingual_interface, UILanguage, ResponseLanguage, CulturalAdaptation
)
from models.database import db_manager
from utils.logger import get_logger

logger = get_logger("smoke_test")


def test_smoke_multilingual_pipeline():
    """Complete end-to-end test of the multilingual pipeline"""
    print("🚀 Starting Multilingual Pipeline Smoke Test\n")
    
    test_user_id = "smoke_test_user"
    success_count = 0
    total_tests = 8
    
    try:
        # Test 1: Set User Preferences
        print("1️⃣ Testing User Preference Management...")
        preferences = preference_manager.get_user_preferences(test_user_id)
        preferences.ui_language = UILanguage.JAPANESE
        preferences.response_language = ResponseLanguage.AUTO
        preferences.cultural_adaptation = CulturalAdaptation.BUSINESS
        preferences.translation_quality_threshold = 0.8
        
        save_success = preference_manager.save_user_preferences(preferences)
        if save_success:
            print("   ✅ Preferences saved successfully")
            success_count += 1
        else:
            print("   ❌ Failed to save preferences")
        
        # Test 2: Verify Preference Loading
        print("\n2️⃣ Testing Preference Loading & Caching...")
        loaded_prefs = preference_manager.get_user_preferences(test_user_id)  # Should hit cache
        if (loaded_prefs.ui_language == UILanguage.JAPANESE and 
            loaded_prefs.translation_quality_threshold == 0.8):
            print("   ✅ Preferences loaded correctly from cache")
            success_count += 1
        else:
            print("   ❌ Preference loading failed")
        
        # Test 3: Glossary Loading
        print("\n3️⃣ Testing Glossary Loading...")
        translation_service = IntelligentTranslationService()
        business_glossary = translation_service.glossary_manager.get_glossary("business")
        technical_glossary = translation_service.glossary_manager.get_glossary("technical")
        
        total_terms = len(business_glossary) + len(technical_glossary)
        if total_terms >= 200:  # Should have at least 200 terms
            print(f"   ✅ Glossaries loaded: {len(business_glossary)} business + {len(technical_glossary)} technical = {total_terms} total terms")
            success_count += 1
        else:
            print(f"   ❌ Insufficient glossary terms: {total_terms}")
        
        # Test 4: Translation with Glossary
        print("\n4️⃣ Testing Translation with Glossary Terms...")
        test_text = "The ROI for this API project shows strong quarterly results."
        
        translation_result = translation_service.translate_with_context(
            test_text,
            target_language="ja",
            source_language="en",
            context=TranslationContext(
                domain="business",
                style="formal",
                audience="professional"
            )
        )
        
        if (translation_result.translated_text and 
            translation_result.quality_score.overall_score >= 0.6):
            print(f"   ✅ Translation successful (quality: {translation_result.quality_score.overall_score:.2f})")
            print(f"      Original: {test_text}")
            print(f"      Translated: {translation_result.translated_text}")
            success_count += 1
        else:
            print("   ❌ Translation failed or low quality")
        
        # Test 5: Multilingual Query Processing
        print("\n5️⃣ Testing Multilingual Query Processing...")
        query_context = multilingual_interface.process_multilingual_query(
            "What is the company's quarterly revenue?", test_user_id
        )
        
        if (query_context.detected_language == "en" and 
            query_context.response_language == "en" and
            len(query_context.search_languages) >= 2):
            print(f"   ✅ Query processed: detected={query_context.detected_language}, response={query_context.response_language}")
            print(f"      Search languages: {query_context.search_languages}")
            success_count += 1
        else:
            print("   ❌ Query processing failed")
        
        # Test 6: Japanese Query Processing
        print("\n6️⃣ Testing Japanese Query Processing...")
        ja_query_context = multilingual_interface.process_multilingual_query(
            "四半期の売上はいくらですか？", test_user_id
        )
        
        if (ja_query_context.detected_language == "ja" and 
            ja_query_context.response_language == "ja"):
            print(f"   ✅ Japanese query processed correctly")
            print(f"      Auto-response language matched: {ja_query_context.response_language}")
            success_count += 1
        else:
            print("   ❌ Japanese query processing failed")
        
        # Test 7: Metrics Tracking
        print("\n7️⃣ Testing Metrics Tracking...")
        metrics = preference_manager.get_metrics()
        persistent_metrics = metrics.get("persistent_metrics", {})
        
        loads = persistent_metrics.get("pref_loads", 0)
        saves = persistent_metrics.get("pref_saves", 0)
        
        if loads > 0 and saves > 0:
            print(f"   ✅ Metrics tracking working: {loads} loads, {saves} saves")
            print(f"      Cache hit rate: {metrics.get('computed_metrics', {}).get('cache_hit_rate', 0):.1f}%")
            success_count += 1
        else:
            print("   ❌ Metrics tracking not working")
        
        # Test 8: Database Operations
        print("\n8️⃣ Testing Database Operations...")
        # Test metric increment
        initial_count = db_manager.get_metric("smoke_test_counter")
        db_manager.increment_metric("smoke_test_counter", 5)
        final_count = db_manager.get_metric("smoke_test_counter")
        
        if final_count == initial_count + 5:
            print(f"   ✅ Database operations working: counter {initial_count} → {final_count}")
            success_count += 1
        else:
            print(f"   ❌ Database operations failed: expected {initial_count + 5}, got {final_count}")
        
        # Summary
        print(f"\n📊 Smoke Test Results:")
        print(f"   Passed: {success_count}/{total_tests}")
        print(f"   Success Rate: {success_count/total_tests*100:.1f}%")
        
        if success_count == total_tests:
            print("\n🎉 ALL TESTS PASSED - Multilingual Pipeline is PRODUCTION READY!")
            return True
        else:
            print(f"\n⚠️  {total_tests - success_count} tests failed - Pipeline needs attention")
            return False
        
    except Exception as e:
        print(f"\n💥 Smoke test failed with error: {e}")
        return False


def test_glossary_must_have_terms():
    """Test that critical business terms are always loaded"""
    print("\n🔍 Testing Must-Have Glossary Terms...")
    
    translation_service = IntelligentTranslationService()
    business_glossary = translation_service.glossary_manager.get_glossary("business")
    
    # Critical terms that should always be present (checking keys in glossary)
    must_have_terms = [
        "ROI", "株式会社", "取締役", "営業部", "売上", "利益"
    ]
    
    found_terms = []
    missing_terms = []
    
    for term in must_have_terms:
        if term in business_glossary:
            found_terms.append(term)
        else:
            missing_terms.append(term)
    
    print(f"   Found terms: {found_terms}")
    if missing_terms:
        print(f"   ⚠️  Missing critical terms: {missing_terms}")
        return False
    else:
        print(f"   ✅ All {len(must_have_terms)} critical terms found")
        return True


def test_cache_hit_rate():
    """Test that caching actually works by doing repeated operations"""
    print("\n⚡ Testing Cache Hit Rate...")
    
    test_user = "cache_test_user"
    
    # Clear any existing cache
    preference_manager.clear_cache(test_user)
    
    # First load (should be cache miss)
    start_time = time.time()
    prefs1 = preference_manager.get_user_preferences(test_user)
    first_load_time = time.time() - start_time
    
    # Second load (should be cache hit)
    start_time = time.time()
    prefs2 = preference_manager.get_user_preferences(test_user)
    second_load_time = time.time() - start_time
    
    # Verify same object and faster load
    if (prefs1.user_id == prefs2.user_id and 
        second_load_time < first_load_time):
        print(f"   ✅ Caching working: {first_load_time*1000:.1f}ms → {second_load_time*1000:.1f}ms")
        return True
    else:
        print(f"   ❌ Caching not working: {first_load_time*1000:.1f}ms → {second_load_time*1000:.1f}ms")
        return False


def test_translation_sanitization():
    """Test that translations don't contain reasoning artifacts"""
    print("🧪 Testing Translation Sanitization...")
    
    try:
        from services.intelligent_translation import IntelligentTranslationService
        
        translation_service = IntelligentTranslationService()
        
        # Test text that might trigger reasoning
        test_text = "What is the quarterly revenue for this fiscal year?"
        
        result = translation_service.translate_with_context(
            test_text, 
            target_language="ja",
            source_language="en"
        )
        
        translated = result.translated_text.lower()
        
        # Check for reasoning artifacts
        reasoning_artifacts = [
            "<think>", "</think>", 
            "<reasoning>", "</reasoning>",
            "<analysis>", "</analysis>",
            "let me think", "i need to", "first, i'll",
            "looking at this", "analyzing", "considering"
        ]
        
        found_artifacts = []
        for artifact in reasoning_artifacts:
            if artifact in translated:
                found_artifacts.append(artifact)
        
        if found_artifacts:
            print(f"   ❌ Found reasoning artifacts: {found_artifacts}")
            print(f"   Translation: {result.translated_text[:100]}...")
            return False
        else:
            print(f"   ✅ Translation clean: {result.translated_text[:50]}...")
            return True
            
    except Exception as e:
        print(f"   💥 Sanitization test failed: {e}")
        return False


def run_comprehensive_smoke_test():
    """Run all smoke tests"""
    print("🧪 Comprehensive Multilingual Pipeline Smoke Test")
    print("=" * 60)
    
    tests = [
        ("Main Pipeline", test_smoke_multilingual_pipeline),
        ("Must-Have Terms", test_glossary_must_have_terms),
        ("Cache Performance", test_cache_hit_rate),
        ("Translation Sanitization", test_translation_sanitization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"   💥 ERROR: {e}")
    
    # Final summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n🏁 Final Results:")
    print("=" * 40)
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"   {status} {test_name}")
    
    print(f"\n📈 Overall Success Rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🚀 MULTILINGUAL PIPELINE IS PRODUCTION READY!")
        print("   All systems operational - ready for deployment! 🎉")
    else:
        print(f"\n⚠️  {total - passed} issues detected - review before deployment")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_smoke_test()
    sys.exit(0 if success else 1)