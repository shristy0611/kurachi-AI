"""
Multilingual Conversation Interface for Kurachi AI
Implements language selection, preference management, cross-language query processing,
and cultural adaptation for multilingual conversations
"""
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime

from services.language_detection import language_detection_service, DocumentLanguage
from services.sota_translation_orchestrator import sota_translation_orchestrator, Language, TranslationQuality, TranslationRequest
from services.intelligent_translation import IntelligentTranslationService, IntelligentContext
# SOTA Translation Orchestrator integration
from models.database import db_manager
from utils.logger import get_logger

logger = get_logger("multilingual_conversation_interface")


class UILanguage(Enum):
    """Supported UI languages"""
    ENGLISH = "en"
    JAPANESE = "ja"


class ResponseLanguage(Enum):
    """Response language preferences"""
    ENGLISH = "en"
    JAPANESE = "ja"
    AUTO = "auto"  # Match query language
    ORIGINAL = "original"  # Keep source document language


class CulturalAdaptation(Enum):
    """Cultural adaptation levels"""
    NONE = "none"
    BASIC = "basic"
    BUSINESS = "business"
    FULL = "full"


@dataclass
class UserLanguagePreferences:
    """User language preferences and settings"""
    user_id: str
    ui_language: UILanguage = UILanguage.ENGLISH
    response_language: ResponseLanguage = ResponseLanguage.AUTO
    fallback_languages: List[str] = None
    show_original_text: bool = True
    translation_quality_threshold: float = 0.75
    cultural_adaptation: CulturalAdaptation = CulturalAdaptation.BUSINESS
    date_format: str = "US"  # US, JP, ISO
    number_format: str = "US"  # US, JP
    auto_translate_queries: bool = False
    auto_translate_responses: bool = True
    preferred_formality: str = "business"  # casual, business, formal
    
    def __post_init__(self):
        if self.fallback_languages is None:
            self.fallback_languages = ["ja", "en"]


@dataclass
class MultilingualQueryContext:
    """Context for multilingual query processing"""
    original_query: str
    detected_language: str
    user_preferences: UserLanguagePreferences
    translated_queries: Dict[str, str] = None
    search_languages: List[str] = None
    response_language: str = None
    
    def __post_init__(self):
        if self.translated_queries is None:
            self.translated_queries = {}
        if self.search_languages is None:
            self.search_languages = []


@dataclass
class MultilingualResponse:
    """Multilingual response with translations and cultural adaptations"""
    content: str
    language: str
    original_content: Optional[str] = None
    original_language: Optional[str] = None
    translations: Dict[str, str] = None
    cultural_adaptations: Dict[str, Any] = None
    sources: List[Dict[str, Any]] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.translations is None:
            self.translations = {}
        if self.cultural_adaptations is None:
            self.cultural_adaptations = {}
        if self.sources is None:
            self.sources = []


class MultilingualConversationInterface:
    """Main interface for multilingual conversation management"""
    
    def __init__(self):
        self.intelligent_translation = IntelligentTranslationService()
        self.preferences_cache = {}
        self.ui_strings = self._load_ui_strings()
        self.cultural_adapters = self._init_cultural_adapters()
        
        logger.info("Multilingual conversation interface initialized")
    
    def _load_ui_strings(self) -> Dict[str, Dict[str, str]]:
        """Load UI strings from configuration"""
        try:
            config_path = Path("config/multilingual_ui.json")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    return config_data.get("ui_strings", {})
            
            # Fallback UI strings
            return {
                "en": {
                    "search_placeholder": "Ask a question or search documents...",
                    "language_selector": "Language",
                    "show_original": "Show original text",
                    "hide_original": "Hide original text",
                    "translation_quality": "Translation quality",
                    "source_document": "Source document",
                    "page_reference": "Page {page}",
                    "no_results": "No results found",
                    "processing": "Processing your request...",
                    "error_translation": "Translation failed",
                    "error_search": "Search failed",
                    "settings": "Settings",
                    "preferences": "Preferences",
                    "clear_chat": "Clear conversation",
                    "export_chat": "Export conversation",
                    "feedback": "Provide feedback",
                    "language_detected": "Language detected: {language}",
                    "cross_language_search": "Searching across languages...",
                    "translation_confidence": "Translation confidence: {confidence}%"
                },
                "ja": {
                    "search_placeholder": "質問を入力するか、文書を検索してください...",
                    "language_selector": "言語",
                    "show_original": "原文を表示",
                    "hide_original": "原文を非表示",
                    "translation_quality": "翻訳品質",
                    "source_document": "参照文書",
                    "page_reference": "{page}ページ",
                    "no_results": "結果が見つかりませんでした",
                    "processing": "処理中...",
                    "error_translation": "翻訳に失敗しました",
                    "error_search": "検索に失敗しました",
                    "settings": "設定",
                    "preferences": "設定",
                    "clear_chat": "会話をクリア",
                    "export_chat": "会話をエクスポート",
                    "feedback": "フィードバック",
                    "language_detected": "検出言語: {language}",
                    "cross_language_search": "多言語検索中...",
                    "translation_confidence": "翻訳信頼度: {confidence}%"
                }
            }
        except Exception as e:
            logger.error(f"Failed to load UI strings: {e}")
            return {"en": {}, "ja": {}}
    
    def _init_cultural_adapters(self) -> Dict[str, Any]:
        """Initialize cultural adaptation rules"""
        return {
            "date_formats": {
                "US": "%m/%d/%Y",
                "JP": "%Y年%m月%d日",
                "ISO": "%Y-%m-%d"
            },
            "number_formats": {
                "US": {"decimal": ".", "thousands": ",", "currency": "$", "position": "before"},
                "JP": {"decimal": ".", "thousands": ",", "currency": "¥", "position": "before"}
            },
            "formality_levels": {
                "casual": {"ja": "だ・である調", "en": "casual"},
                "business": {"ja": "です・ます調", "en": "professional"},
                "formal": {"ja": "敬語", "en": "formal"}
            },
            "cultural_references": {
                "business_greetings": {
                    "en": ["Good morning", "Good afternoon", "Thank you for your time"],
                    "ja": ["おはようございます", "お疲れ様です", "よろしくお願いいたします"]
                },
                "meeting_terms": {
                    "en": ["meeting", "discussion", "presentation", "agenda"],
                    "ja": ["会議", "打ち合わせ", "プレゼンテーション", "議題"]
                }
            }
        }
    
    def get_user_preferences(self, user_id: str) -> UserLanguagePreferences:
        """Get user language preferences with caching"""
        if user_id in self.preferences_cache:
            return self.preferences_cache[user_id]
        
        try:
            # Try to load from database
            preferences_data = db_manager.get_user_preferences(user_id, "language")
            
            if preferences_data:
                # Convert string values back to enums
                if 'ui_language' in preferences_data:
                    preferences_data['ui_language'] = UILanguage(preferences_data['ui_language'])
                if 'response_language' in preferences_data:
                    preferences_data['response_language'] = ResponseLanguage(preferences_data['response_language'])
                if 'cultural_adaptation' in preferences_data:
                    preferences_data['cultural_adaptation'] = CulturalAdaptation(preferences_data['cultural_adaptation'])
                
                preferences = UserLanguagePreferences(
                    user_id=user_id,
                    **preferences_data
                )
            else:
                # Create default preferences
                preferences = UserLanguagePreferences(user_id=user_id)
                self.save_user_preferences(preferences)
            
            self.preferences_cache[user_id] = preferences
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to get user preferences for {user_id}: {e}")
            return UserLanguagePreferences(user_id=user_id)
    
    def save_user_preferences(self, preferences: UserLanguagePreferences) -> bool:
        """Save user language preferences"""
        try:
            preferences_dict = asdict(preferences)
            preferences_dict.pop('user_id')  # Remove user_id from data
            
            # Convert enum values to strings for JSON serialization
            if 'ui_language' in preferences_dict:
                preferences_dict['ui_language'] = preferences_dict['ui_language'].value
            if 'response_language' in preferences_dict:
                preferences_dict['response_language'] = preferences_dict['response_language'].value
            if 'cultural_adaptation' in preferences_dict:
                preferences_dict['cultural_adaptation'] = preferences_dict['cultural_adaptation'].value
            
            success = db_manager.save_user_preferences(
                preferences.user_id, 
                "language", 
                preferences_dict
            )
            
            if success:
                self.preferences_cache[preferences.user_id] = preferences
                logger.info(f"Saved language preferences for user {preferences.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
            return False
    
    def get_ui_text(self, key: str, language: str = "en", **kwargs) -> str:
        """Get localized UI text with parameter substitution"""
        try:
            text = self.ui_strings.get(language, {}).get(key, key)
            if kwargs:
                text = text.format(**kwargs)
            return text
        except Exception as e:
            logger.error(f"Failed to get UI text for {key}: {e}")
            return key
    
    def process_multilingual_query(self, query: str, user_id: str, 
                                 conversation_id: Optional[str] = None) -> MultilingualQueryContext:
        """Process query for multilingual search and response"""
        try:
            preferences = self.get_user_preferences(user_id)
            
            # Detect query language
            detection_result = language_detection_service.detect_document_language(query, detailed=False)
            detected_language = detection_result.primary_language.value
            
            context = MultilingualQueryContext(
                original_query=query,
                detected_language=detected_language,
                user_preferences=preferences
            )
            
            # Determine search languages based on preferences
            context.search_languages = self._determine_search_languages(
                detected_language, preferences
            )
            
            # Generate translated queries if needed for cross-language search
            if preferences.auto_translate_queries or len(context.search_languages) > 1:
                context.translated_queries = self._generate_translated_queries(
                    query, detected_language, context.search_languages
                )
            
            # Determine response language
            context.response_language = self._determine_response_language(
                detected_language, preferences
            )
            
            logger.info(f"Processed multilingual query: {detected_language} -> {context.response_language}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to process multilingual query: {e}")
            return MultilingualQueryContext(
                original_query=query,
                detected_language="en",
                user_preferences=self.get_user_preferences(user_id)
            )
    
    def _determine_search_languages(self, query_language: str, 
                                  preferences: UserLanguagePreferences) -> List[str]:
        """Determine which languages to search in"""
        search_languages = [query_language]
        
        # Add fallback languages if different from query language
        for lang in preferences.fallback_languages:
            if lang not in search_languages:
                search_languages.append(lang)
        
        # Always include English and Japanese for comprehensive search
        for lang in ["en", "ja"]:
            if lang not in search_languages:
                search_languages.append(lang)
        
        return search_languages[:3]  # Limit to 3 languages for performance
    
    def _generate_translated_queries(self, query: str, source_language: str, 
                                   target_languages: List[str]) -> Dict[str, str]:
        """Generate translated versions of the query"""
        translated_queries = {}
        
        for target_lang in target_languages:
            if target_lang == source_language:
                translated_queries[target_lang] = query
                continue
            
            try:
                # Use intelligent translation for better context awareness
                translation_result = self.intelligent_translation.translate_with_context(
                    query,
                    target_language=target_lang,
                    source_language=source_language,
                    context=IntelligentContext(
                        domain="business",
                        style="formal",
                        audience="professional"
                    )
                )
                
                if translation_result.quality_score.overall_score >= 0.6:
                    translated_queries[target_lang] = translation_result.translated_text
                    logger.debug(f"Generated query translation {source_language}->{target_lang}")
                else:
                    logger.warning(f"Low quality translation for query: {translation_result.quality_score.overall_score}")
                    
            except Exception as e:
                logger.error(f"Failed to translate query to {target_lang}: {e}")
        
        return translated_queries
    
    def _determine_response_language(self, query_language: str, 
                                   preferences: UserLanguagePreferences) -> str:
        """Determine the language for the response"""
        if preferences.response_language == ResponseLanguage.AUTO:
            return query_language
        elif preferences.response_language == ResponseLanguage.ORIGINAL:
            return "original"  # Keep source document language
        else:
            return preferences.response_language.value
    
    def format_multilingual_response(self, response_content: str, sources: List[Dict[str, Any]],
                                   context: MultilingualQueryContext) -> MultilingualResponse:
        """Format response with multilingual support and cultural adaptation"""
        try:
            preferences = context.user_preferences
            target_language = context.response_language
            
            # If response language matches content language, no translation needed
            if target_language == context.detected_language:
                return MultilingualResponse(
                    content=response_content,
                    language=target_language,
                    sources=sources,
                    confidence=1.0
                )
            
            # Translate response if needed
            translated_content = response_content
            translations = {}
            
            if preferences.auto_translate_responses and target_language != "original":
                try:
                    translation_result = self.intelligent_translation.translate_with_context(
                        response_content,
                        target_language=target_language,
                        source_language=context.detected_language,
                        context=IntelligentContext(
                            domain="business",
                            style=preferences.preferred_formality,
                            audience="professional"
                        )
                    )
                    
                    if translation_result.quality_score.overall_score >= preferences.translation_quality_threshold:
                        translated_content = translation_result.translated_text
                        translations[target_language] = translated_content
                        
                        # Apply cultural adaptations
                        translated_content = self._apply_cultural_adaptations(
                            translated_content, target_language, preferences
                        )
                    
                except Exception as e:
                    logger.error(f"Failed to translate response: {e}")
            
            # Process multilingual sources
            multilingual_sources = self._process_multilingual_sources(
                sources, target_language, preferences
            )
            
            return MultilingualResponse(
                content=translated_content,
                language=target_language,
                original_content=response_content if translated_content != response_content else None,
                original_language=context.detected_language,
                translations=translations,
                sources=multilingual_sources,
                confidence=translation_result.quality_score.overall_score if 'translation_result' in locals() else 1.0
            )
            
        except Exception as e:
            logger.error(f"Failed to format multilingual response: {e}")
            return MultilingualResponse(
                content=response_content,
                language=context.detected_language,
                sources=sources,
                confidence=0.5
            )
    
    def _apply_cultural_adaptations(self, content: str, language: str, 
                                  preferences: UserLanguagePreferences) -> str:
        """Apply cultural adaptations to content"""
        try:
            adapted_content = content
            
            if preferences.cultural_adaptation == CulturalAdaptation.NONE:
                return adapted_content
            
            # Date format adaptation
            if preferences.date_format in self.cultural_adapters["date_formats"]:
                # This would require more sophisticated date parsing and replacement
                # For now, we'll log the intent
                logger.debug(f"Would apply date format: {preferences.date_format}")
            
            # Number format adaptation
            if preferences.number_format in self.cultural_adapters["number_formats"]:
                # This would require number parsing and reformatting
                logger.debug(f"Would apply number format: {preferences.number_format}")
            
            # Business greeting adaptations for Japanese
            if language == "ja" and preferences.cultural_adaptation in [CulturalAdaptation.BUSINESS, CulturalAdaptation.FULL]:
                # Add appropriate business greetings if missing
                if not any(greeting in adapted_content for greeting in ["です", "ます", "ございます"]):
                    # This is a simplified example - real implementation would be more sophisticated
                    logger.debug("Would add Japanese business formality")
            
            return adapted_content
            
        except Exception as e:
            logger.error(f"Failed to apply cultural adaptations: {e}")
            return content
    
    def _process_multilingual_sources(self, sources: List[Dict[str, Any]], 
                                    target_language: str, 
                                    preferences: UserLanguagePreferences) -> List[Dict[str, Any]]:
        """Process sources for multilingual display"""
        processed_sources = []
        
        for source in sources:
            try:
                processed_source = source.copy()
                
                # Translate source excerpts if needed and requested
                if (preferences.show_original_text and 
                    source.get('detected_language') != target_language and
                    preferences.auto_translate_responses):
                    
                    original_content = source.get('content', '')
                    if original_content and len(original_content) > 10:
                        try:
                            translation_result = self.intelligent_translation.translate_with_context(
                                original_content,
                                target_language=target_language,
                                source_language=source.get('detected_language', 'auto'),
                                context=IntelligentContext(domain="business")
                            )
                            
                            if translation_result.quality_score.overall_score >= 0.6:
                                processed_source['translated_content'] = translation_result.translated_text
                                processed_source['translation_confidence'] = translation_result.quality_score.overall_score
                        
                        except Exception as e:
                            logger.error(f"Failed to translate source excerpt: {e}")
                
                processed_sources.append(processed_source)
                
            except Exception as e:
                logger.error(f"Failed to process source: {e}")
                processed_sources.append(source)
        
        return processed_sources
    
    def get_error_message(self, error_type: str, language: str = "en", **kwargs) -> str:
        """Get localized error messages"""
        error_messages = {
            "en": {
                "translation_failed": "Translation failed. Please try again.",
                "search_failed": "Search failed. Please check your query.",
                "language_detection_failed": "Could not detect language. Defaulting to English.",
                "preferences_save_failed": "Failed to save preferences.",
                "no_results": "No results found for your query.",
                "processing_error": "An error occurred while processing your request."
            },
            "ja": {
                "translation_failed": "翻訳に失敗しました。もう一度お試しください。",
                "search_failed": "検索に失敗しました。クエリを確認してください。",
                "language_detection_failed": "言語を検出できませんでした。英語をデフォルトとします。",
                "preferences_save_failed": "設定の保存に失敗しました。",
                "no_results": "クエリに対する結果が見つかりませんでした。",
                "processing_error": "リクエストの処理中にエラーが発生しました。"
            }
        }
        
        try:
            message = error_messages.get(language, error_messages["en"]).get(error_type, error_type)
            if kwargs:
                message = message.format(**kwargs)
            return message
        except Exception:
            return error_type
    
    def get_help_text(self, topic: str, language: str = "en") -> str:
        """Get localized help text"""
        help_texts = {
            "en": {
                "language_selection": "Choose your preferred interface language. This affects menus, buttons, and system messages.",
                "response_language": "Choose the language for AI responses. 'Auto' matches your query language.",
                "cross_language_search": "When enabled, searches across documents in all languages for better results.",
                "translation_quality": "Set minimum quality threshold for automatic translations. Higher values mean more accurate but fewer translations.",
                "cultural_adaptation": "Adapt responses for cultural context, including date formats, business etiquette, and formality levels.",
                "show_original": "Display original text alongside translations for verification and context."
            },
            "ja": {
                "language_selection": "お好みのインターフェース言語を選択してください。メニュー、ボタン、システムメッセージに影響します。",
                "response_language": "AI応答の言語を選択してください。「自動」はクエリ言語に合わせます。",
                "cross_language_search": "有効にすると、すべての言語の文書を横断検索してより良い結果を得られます。",
                "translation_quality": "自動翻訳の最小品質閾値を設定します。高い値はより正確ですが翻訳数が少なくなります。",
                "cultural_adaptation": "日付形式、ビジネスエチケット、敬語レベルなど、文化的文脈に応答を適応させます。",
                "show_original": "検証と文脈のため、翻訳と一緒に原文を表示します。"
            }
        }
        
        return help_texts.get(language, help_texts["en"]).get(topic, f"Help for {topic}")
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages for UI"""
        return [
            {
                "code": "en",
                "name": "English",
                "native_name": "English",
                "flag": "🇺🇸",
                "ui_support": True,
                "translation_support": True
            },
            {
                "code": "ja", 
                "name": "Japanese",
                "native_name": "日本語",
                "flag": "🇯🇵",
                "ui_support": True,
                "translation_support": True
            }
        ]


# Global instance
multilingual_interface = MultilingualConversationInterface()