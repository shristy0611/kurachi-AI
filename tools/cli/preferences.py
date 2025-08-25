#!/usr/bin/env python3
"""
CLI tool for managing user language preferences
Provides command-line interface for setting and viewing multilingual preferences
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from typing import Optional

from services.preference_manager import preference_manager
from services.multilingual_conversation_interface import UILanguage, ResponseLanguage, CulturalAdaptation
from utils.logger import get_logger

logger = get_logger("cli_preferences")


def set_ui_language(user_id: str, language: str) -> bool:
    """Set user's UI language"""
    try:
        if language not in ["en", "ja"]:
            print(f"❌ Invalid UI language: {language}. Must be 'en' or 'ja'")
            return False
        
        preferences = preference_manager.get_user_preferences(user_id)
        preferences.ui_language = UILanguage(language)
        
        success = preference_manager.save_user_preferences(preferences)
        if success:
            print(f"✅ UI language set to {language} for user {user_id}")
        else:
            print(f"❌ Failed to save UI language preference")
        
        return success
        
    except Exception as e:
        print(f"❌ Error setting UI language: {e}")
        return False


def set_response_language(user_id: str, language: str) -> bool:
    """Set user's response language"""
    try:
        if language not in ["en", "ja", "auto", "original"]:
            print(f"❌ Invalid response language: {language}. Must be 'en', 'ja', 'auto', or 'original'")
            return False
        
        preferences = preference_manager.get_user_preferences(user_id)
        preferences.response_language = ResponseLanguage(language)
        
        success = preference_manager.save_user_preferences(preferences)
        if success:
            print(f"✅ Response language set to {language} for user {user_id}")
        else:
            print(f"❌ Failed to save response language preference")
        
        return success
        
    except Exception as e:
        print(f"❌ Error setting response language: {e}")
        return False


def set_cultural_adaptation(user_id: str, level: str) -> bool:
    """Set user's cultural adaptation level"""
    try:
        if level not in ["none", "basic", "business", "full"]:
            print(f"❌ Invalid cultural adaptation level: {level}. Must be 'none', 'basic', 'business', or 'full'")
            return False
        
        preferences = preference_manager.get_user_preferences(user_id)
        preferences.cultural_adaptation = CulturalAdaptation(level)
        
        success = preference_manager.save_user_preferences(preferences)
        if success:
            print(f"✅ Cultural adaptation set to {level} for user {user_id}")
        else:
            print(f"❌ Failed to save cultural adaptation preference")
        
        return success
        
    except Exception as e:
        print(f"❌ Error setting cultural adaptation: {e}")
        return False


def set_translation_quality(user_id: str, threshold: float) -> bool:
    """Set user's translation quality threshold"""
    try:
        if not (0.5 <= threshold <= 1.0):
            print(f"❌ Invalid translation quality threshold: {threshold}. Must be between 0.5 and 1.0")
            return False
        
        preferences = preference_manager.get_user_preferences(user_id)
        preferences.translation_quality_threshold = threshold
        
        success = preference_manager.save_user_preferences(preferences)
        if success:
            print(f"✅ Translation quality threshold set to {threshold:.2f} for user {user_id}")
        else:
            print(f"❌ Failed to save translation quality threshold")
        
        return success
        
    except Exception as e:
        print(f"❌ Error setting translation quality threshold: {e}")
        return False


def view_preferences(user_id: str) -> bool:
    """View user's current preferences"""
    try:
        preferences = preference_manager.get_user_preferences(user_id)
        
        print(f"\n📋 Language Preferences for User: {user_id}")
        print("=" * 50)
        print(f"🌐 UI Language: {preferences.ui_language.value}")
        print(f"💬 Response Language: {preferences.response_language.value}")
        print(f"🎭 Cultural Adaptation: {preferences.cultural_adaptation.value}")
        print(f"📊 Translation Quality Threshold: {preferences.translation_quality_threshold:.2f}")
        print(f"🔄 Auto-translate Queries: {preferences.auto_translate_queries}")
        print(f"🔄 Auto-translate Responses: {preferences.auto_translate_responses}")
        print(f"📄 Show Original Text: {preferences.show_original_text}")
        print(f"📅 Date Format: {preferences.date_format}")
        print(f"🔢 Number Format: {preferences.number_format}")
        print(f"🎯 Preferred Formality: {preferences.preferred_formality}")
        print(f"🔄 Fallback Languages: {', '.join(preferences.fallback_languages)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error viewing preferences: {e}")
        return False


def export_preferences(user_id: str, output_file: Optional[str] = None) -> bool:
    """Export user preferences to JSON file"""
    try:
        export_data = preference_manager.export_preferences(user_id)
        
        if not export_data:
            print(f"❌ Failed to export preferences for user {user_id}")
            return False
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Preferences exported to {output_file}")
        else:
            print(f"\n📤 Exported Preferences for {user_id}:")
            print(json.dumps(export_data, indent=2, ensure_ascii=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting preferences: {e}")
        return False


def import_preferences(input_file: str) -> bool:
    """Import user preferences from JSON file"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        success = preference_manager.import_preferences(import_data)
        
        if success:
            user_id = import_data.get("user_id", "unknown")
            print(f"✅ Preferences imported successfully for user {user_id}")
        else:
            print(f"❌ Failed to import preferences")
        
        return success
        
    except Exception as e:
        print(f"❌ Error importing preferences: {e}")
        return False


def show_metrics() -> bool:
    """Show comprehensive preference manager metrics"""
    try:
        metrics = preference_manager.get_metrics()
        
        print(f"\n📊 Preference Manager Metrics")
        print("=" * 50)
        
        # Session metrics (current process)
        session = metrics.get("session_metrics", {})
        print(f"\n🔄 Session Metrics (Current Process):")
        print(f"  Cache Hits: {session.get('cache_hits', 0)}")
        print(f"  Cache Misses: {session.get('cache_misses', 0)}")
        print(f"  Load Operations: {session.get('load_operations', 0)}")
        print(f"  Save Operations: {session.get('save_operations', 0)}")
        print(f"  Validation Failures: {session.get('validation_failures', 0)}")
        
        # Persistent metrics (lifetime)
        persistent = metrics.get("persistent_metrics", {})
        print(f"\n💾 Persistent Metrics (Lifetime):")
        print(f"  Total Loads: {persistent.get('pref_loads', 0)}")
        print(f"  Total Saves: {persistent.get('pref_saves', 0)}")
        print(f"  Cache Hits: {persistent.get('pref_cache_hits', 0)}")
        print(f"  Cache Misses: {persistent.get('pref_cache_misses', 0)}")
        print(f"  Validation Failures: {persistent.get('pref_validation_failures', 0)}")
        
        # Computed metrics
        computed = metrics.get("computed_metrics", {})
        print(f"\n📈 Computed Metrics:")
        print(f"  Cache Hit Rate: {computed.get('cache_hit_rate', 0):.1f}%")
        print(f"  Validation Failure Rate: {computed.get('validation_failure_rate', 0):.1f}%")
        
        # Show any errors
        if "error" in metrics:
            print(f"\n⚠️  Error: {metrics['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error showing metrics: {e}")
        return False


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Manage user language preferences for Kurachi AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s view user123                           # View preferences
  %(prog)s set-ui user123 ja                      # Set UI to Japanese
  %(prog)s set-response user123 auto              # Set response to auto
  %(prog)s set-cultural user123 business          # Set cultural adaptation
  %(prog)s set-quality user123 0.8                # Set translation quality
  %(prog)s export user123 backup.json             # Export to file
  %(prog)s import backup.json                     # Import from file
  %(prog)s metrics                                # Show metrics
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View user preferences')
    view_parser.add_argument('user_id', help='User ID')
    
    # Set UI language command
    ui_parser = subparsers.add_parser('set-ui', help='Set UI language')
    ui_parser.add_argument('user_id', help='User ID')
    ui_parser.add_argument('language', choices=['en', 'ja'], help='UI language')
    
    # Set response language command
    response_parser = subparsers.add_parser('set-response', help='Set response language')
    response_parser.add_argument('user_id', help='User ID')
    response_parser.add_argument('language', choices=['en', 'ja', 'auto', 'original'], help='Response language')
    
    # Set cultural adaptation command
    cultural_parser = subparsers.add_parser('set-cultural', help='Set cultural adaptation level')
    cultural_parser.add_argument('user_id', help='User ID')
    cultural_parser.add_argument('level', choices=['none', 'basic', 'business', 'full'], help='Cultural adaptation level')
    
    # Set translation quality command
    quality_parser = subparsers.add_parser('set-quality', help='Set translation quality threshold')
    quality_parser.add_argument('user_id', help='User ID')
    quality_parser.add_argument('threshold', type=float, help='Quality threshold (0.5-1.0)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export user preferences')
    export_parser.add_argument('user_id', help='User ID')
    export_parser.add_argument('--output', '-o', help='Output file (optional)')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import user preferences')
    import_parser.add_argument('input_file', help='Input JSON file')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show preference manager metrics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute commands
    success = False
    
    if args.command == 'view':
        success = view_preferences(args.user_id)
    elif args.command == 'set-ui':
        success = set_ui_language(args.user_id, args.language)
    elif args.command == 'set-response':
        success = set_response_language(args.user_id, args.language)
    elif args.command == 'set-cultural':
        success = set_cultural_adaptation(args.user_id, args.level)
    elif args.command == 'set-quality':
        success = set_translation_quality(args.user_id, args.threshold)
    elif args.command == 'export':
        success = export_preferences(args.user_id, args.output)
    elif args.command == 'import':
        success = import_preferences(args.input_file)
    elif args.command == 'metrics':
        success = show_metrics()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())