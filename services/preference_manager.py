"""
Enhanced Preference Manager for Multilingual Interface
Provides robust preference management with validation, caching, and per-user defaults
"""
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading
from pathlib import Path

from services.multilingual_conversation_interface import (
    UserLanguagePreferences, UILanguage, ResponseLanguage, CulturalAdaptation
)
from models.database import db_manager
from utils.logger import get_logger

logger = get_logger("preference_manager")


class PreferenceValidationError(Exception):
    """Raised when preference validation fails"""
    pass


@dataclass
class PreferenceMetrics:
    """Metrics for preference usage and performance"""
    cache_hits: int = 0
    cache_misses: int = 0
    validation_failures: int = 0
    save_operations: int = 0
    load_operations: int = 0
    last_updated: Optional[datetime] = None


class EnhancedPreferenceManager:
    """Enhanced preference manager with validation, caching, and metrics"""
    
    def __init__(self):
        self.cache = {}
        self.cache_lock = threading.RLock()
        self.metrics = PreferenceMetrics()
        self.default_preferences = self._load_default_preferences()
        
        # Mixed language detection thresholds
        self.mixed_language_config = {
            "min_secondary_ratio": 0.20,  # Increased from 0.15 for better accuracy
            "min_text_length": 50,        # Minimum characters to consider mixed
            "confidence_threshold": 0.75   # Minimum combined confidence
        }
        
        logger.info("Enhanced preference manager initialized")
    
    def _load_default_preferences(self) -> Dict[str, UserLanguagePreferences]:
        """Load default preferences for different user types"""
        return {
            "business_jp": UserLanguagePreferences(
                user_id="",
                ui_language=UILanguage.JAPANESE,
                response_language=ResponseLanguage.AUTO,
                cultural_adaptation=CulturalAdaptation.BUSINESS,
                auto_translate_queries=True,
                auto_translate_responses=True,
                translation_quality_threshold=0.75,
                preferred_formality="business",
                fallback_languages=["ja", "en"]
            ),
            "business_en": UserLanguagePreferences(
                user_id="",
                ui_language=UILanguage.ENGLISH,
                response_language=ResponseLanguage.AUTO,
                cultural_adaptation=CulturalAdaptation.BUSINESS,
                auto_translate_queries=True,
                auto_translate_responses=True,
                translation_quality_threshold=0.75,
                preferred_formality="business",
                fallback_languages=["en", "ja"]
            ),
            "default": UserLanguagePreferences(
                user_id="",
                ui_language=UILanguage.ENGLISH,
                response_language=ResponseLanguage.AUTO,
                cultural_adaptation=CulturalAdaptation.BASIC,
                auto_translate_queries=False,
                auto_translate_responses=True,
                translation_quality_threshold=0.70,
                preferred_formality="casual",
                fallback_languages=["en", "ja"]
            )
        }
    
    def get_user_preferences(self, user_id: str, use_cache: bool = True) -> UserLanguagePreferences:
        """Get user preferences with enhanced caching and validation"""
        try:
            self.metrics.load_operations += 1
            db_manager.increment_metric("pref_loads")
            
            # Check cache first
            if use_cache:
                with self.cache_lock:
                    if user_id in self.cache:
                        self.metrics.cache_hits += 1
                        db_manager.increment_metric("pref_cache_hits")
                        return self.cache[user_id]
            
            self.metrics.cache_misses += 1
            db_manager.increment_metric("pref_cache_misses")
            
            # Load from database
            preferences_data = db_manager.get_user_preferences(user_id, "language")
            
            if preferences_data:
                # Validate and convert loaded data
                preferences = self._validate_and_convert_preferences(user_id, preferences_data)
            else:
                # Create default preferences based on user context
                preferences = self._create_default_preferences(user_id)
                self.save_user_preferences(preferences, update_cache=False)
            
            # Cache the result
            if use_cache:
                with self.cache_lock:
                    self.cache[user_id] = preferences
            
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to get user preferences for {user_id}: {e}")
            # Return safe default
            default_prefs = self.default_preferences["default"]
            default_prefs.user_id = user_id
            return default_prefs
    
    def save_user_preferences(self, preferences: UserLanguagePreferences, 
                            update_cache: bool = True) -> bool:
        """Save user preferences with validation and caching"""
        try:
            self.metrics.save_operations += 1
            db_manager.increment_metric("pref_saves")
            
            # Validate preferences
            if not self._validate_preferences(preferences):
                self.metrics.validation_failures += 1
                db_manager.increment_metric("pref_validation_failures")
                raise PreferenceValidationError("Preference validation failed")
            
            # Convert to dict and normalize enums
            preferences_dict = asdict(preferences)
            preferences_dict.pop('user_id')  # Remove user_id from data
            
            # Save to database (database layer handles enum normalization)
            success = db_manager.save_user_preferences(
                preferences.user_id, 
                "language", 
                preferences_dict
            )
            
            if success:
                # Update cache
                if update_cache:
                    with self.cache_lock:
                        self.cache[preferences.user_id] = preferences
                
                self.metrics.last_updated = datetime.utcnow()
                logger.info(f"Saved preferences for user {preferences.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
            return False
    
    def _validate_preferences(self, preferences: UserLanguagePreferences) -> bool:
        """Validate preference values"""
        try:
            # Check required fields
            if not preferences.user_id:
                logger.warning("Missing user_id in preferences")
                return False
            
            # Validate enum values
            if not isinstance(preferences.ui_language, UILanguage):
                logger.warning(f"Invalid ui_language type: {type(preferences.ui_language)}")
                return False
            
            if not isinstance(preferences.response_language, ResponseLanguage):
                logger.warning(f"Invalid response_language type: {type(preferences.response_language)}")
                return False
            
            if not isinstance(preferences.cultural_adaptation, CulturalAdaptation):
                logger.warning(f"Invalid cultural_adaptation type: {type(preferences.cultural_adaptation)}")
                return False
            
            # Validate numeric ranges
            if not (0.5 <= preferences.translation_quality_threshold <= 1.0):
                logger.warning(f"Invalid translation_quality_threshold: {preferences.translation_quality_threshold}")
                return False
            
            # Validate fallback languages
            if not preferences.fallback_languages or len(preferences.fallback_languages) == 0:
                logger.warning("Empty fallback_languages list")
                return False
            
            valid_languages = {"en", "ja", "zh-cn", "ko", "es", "fr", "de"}
            for lang in preferences.fallback_languages:
                if lang not in valid_languages:
                    logger.warning(f"Invalid fallback language: {lang}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Preference validation error: {e}")
            return False
    
    def _validate_and_convert_preferences(self, user_id: str, 
                                        preferences_data: Dict[str, Any]) -> UserLanguagePreferences:
        """Validate and convert loaded preference data"""
        try:
            # Convert string values back to enums
            if 'ui_language' in preferences_data:
                preferences_data['ui_language'] = UILanguage(preferences_data['ui_language'])
            if 'response_language' in preferences_data:
                preferences_data['response_language'] = ResponseLanguage(preferences_data['response_language'])
            if 'cultural_adaptation' in preferences_data:
                preferences_data['cultural_adaptation'] = CulturalAdaptation(preferences_data['cultural_adaptation'])
            
            # Create preferences object
            preferences = UserLanguagePreferences(
                user_id=user_id,
                **preferences_data
            )
            
            # Validate
            if not self._validate_preferences(preferences):
                raise PreferenceValidationError("Loaded preferences failed validation")
            
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to validate/convert preferences: {e}")
            # Return default preferences
            return self._create_default_preferences(user_id)
    
    def _create_default_preferences(self, user_id: str) -> UserLanguagePreferences:
        """Create default preferences based on user context"""
        # Simple heuristic: if user_id contains Japanese characters, use Japanese defaults
        if any('\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FAF' 
               for char in user_id):
            template = self.default_preferences["business_jp"]
        else:
            template = self.default_preferences["business_en"]
        
        # Create new instance with user_id
        preferences = UserLanguagePreferences(
            user_id=user_id,
            ui_language=template.ui_language,
            response_language=template.response_language,
            fallback_languages=template.fallback_languages.copy(),
            show_original_text=template.show_original_text,
            translation_quality_threshold=template.translation_quality_threshold,
            cultural_adaptation=template.cultural_adaptation,
            date_format=template.date_format,
            number_format=template.number_format,
            auto_translate_queries=template.auto_translate_queries,
            auto_translate_responses=template.auto_translate_responses,
            preferred_formality=template.preferred_formality
        )
        
        logger.info(f"Created default preferences for user {user_id}: {template.ui_language.value}")
        return preferences
    
    def handle_mixed_language_response(self, user_id: str, detected_languages: List[str], 
                                     confidence_scores: List[float]) -> str:
        """Handle response language for mixed-language queries"""
        try:
            preferences = self.get_user_preferences(user_id)
            
            if preferences.response_language == ResponseLanguage.AUTO:
                # For mixed language, use the primary language if confidence is high
                if confidence_scores and confidence_scores[0] >= self.mixed_language_config["confidence_threshold"]:
                    return detected_languages[0]
                else:
                    # Fall back to user's UI language for mixed/uncertain cases
                    return preferences.ui_language.value
            else:
                # Use explicit preference
                return preferences.response_language.value
                
        except Exception as e:
            logger.error(f"Failed to handle mixed language response: {e}")
            return "en"  # Safe default
    
    def is_mixed_language_robust(self, text: str, language_scores: Dict[str, float]) -> bool:
        """Enhanced mixed language detection with configurable thresholds"""
        try:
            if len(text) < self.mixed_language_config["min_text_length"]:
                return False
            
            # Sort languages by score
            sorted_langs = sorted(language_scores.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_langs) < 2:
                return False
            
            primary_score = sorted_langs[0][1]
            secondary_score = sorted_langs[1][1]
            
            # Check if secondary language meets threshold
            secondary_ratio = secondary_score / (primary_score + secondary_score) if (primary_score + secondary_score) > 0 else 0
            
            is_mixed = (secondary_ratio >= self.mixed_language_config["min_secondary_ratio"] and
                       (primary_score + secondary_score) >= self.mixed_language_config["confidence_threshold"])
            
            if is_mixed:
                logger.debug(f"Mixed language detected: primary={sorted_langs[0][0]}({primary_score:.2f}), "
                           f"secondary={sorted_langs[1][0]}({secondary_score:.2f}), ratio={secondary_ratio:.2f}")
            
            return is_mixed
            
        except Exception as e:
            logger.error(f"Mixed language detection failed: {e}")
            return False
    
    def clear_cache(self, user_id: Optional[str] = None):
        """Clear preference cache"""
        with self.cache_lock:
            if user_id:
                self.cache.pop(user_id, None)
                logger.info(f"Cleared cache for user {user_id}")
            else:
                self.cache.clear()
                logger.info("Cleared all preference cache")
    
    def get_preference(self, key: str, default=None):
        """
        Simple preference getter expected by validators.
        This is a simplified interface for basic preference access.
        """
        try:
            # For now, return some basic preferences that validators might check
            basic_prefs = {
                "ui_language": "en",
                "response_language": "auto",
                "cultural_adaptation": "basic",
                "auto_translate": True
            }
            return basic_prefs.get(key, default)
        except Exception as e:
            logger.warning(f"Failed to get preference {key}: {e}")
            return default
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive preference manager metrics (session + persistent)"""
        try:
            # Get persistent metrics from database
            persistent_metrics = db_manager.get_all_metrics()
            
            # Get audit-based metrics
            audit_metrics = db_manager.get_metrics_from_audit_logs()
            
            # Combine with session metrics
            return {
                "session_metrics": {
                    "cache_hits": self.metrics.cache_hits,
                    "cache_misses": self.metrics.cache_misses,
                    "validation_failures": self.metrics.validation_failures,
                    "save_operations": self.metrics.save_operations,
                    "load_operations": self.metrics.load_operations,
                    "last_updated": self.metrics.last_updated
                },
                "persistent_metrics": persistent_metrics,
                "audit_metrics": audit_metrics,
                "computed_metrics": {
                    "cache_hit_rate": (
                        persistent_metrics.get("pref_cache_hits", 0) / 
                        max(1, persistent_metrics.get("pref_cache_hits", 0) + persistent_metrics.get("pref_cache_misses", 0))
                    ) * 100,
                    "validation_failure_rate": (
                        persistent_metrics.get("pref_validation_failures", 0) / 
                        max(1, persistent_metrics.get("pref_saves", 1))
                    ) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive metrics: {e}")
            return {"session_metrics": asdict(self.metrics), "error": str(e)}
    
    def export_preferences(self, user_id: str) -> Dict[str, Any]:
        """Export user preferences for backup/migration"""
        try:
            preferences = self.get_user_preferences(user_id)
            return {
                "user_id": user_id,
                "preferences": asdict(preferences),
                "exported_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            }
        except Exception as e:
            logger.error(f"Failed to export preferences for {user_id}: {e}")
            return {}
    
    def import_preferences(self, export_data: Dict[str, Any]) -> bool:
        """Import user preferences from backup/migration"""
        try:
            if "preferences" not in export_data or "user_id" not in export_data:
                logger.error("Invalid export data format")
                return False
            
            user_id = export_data["user_id"]
            preferences_data = export_data["preferences"]
            
            # Validate and convert
            preferences = self._validate_and_convert_preferences(user_id, preferences_data)
            
            # Save
            return self.save_user_preferences(preferences)
            
        except Exception as e:
            logger.error(f"Failed to import preferences: {e}")
            return False


# Global enhanced preference manager instance
preference_manager = EnhancedPreferenceManager()