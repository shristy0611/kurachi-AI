"""
Multilingual UI Components for Streamlit
Provides language selection, preference management, and multilingual display components
"""
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

from services.multilingual_conversation_interface import (
    multilingual_interface, 
    UserLanguagePreferences,
    UILanguage,
    ResponseLanguage,
    CulturalAdaptation
)
from utils.logger import get_logger

logger = get_logger("multilingual_components")


class MultilingualComponents:
    """Streamlit components for multilingual interface"""
    
    def __init__(self):
        self.interface = multilingual_interface
    
    def render_language_selector(self, user_id: str, key_prefix: str = "") -> Tuple[str, bool]:
        """
        Render language selector widget
        
        Returns:
            Tuple of (selected_language, changed)
        """
        try:
            preferences = self.interface.get_user_preferences(user_id)
            supported_languages = self.interface.get_supported_languages()
            
            # Create language options
            language_options = {}
            for lang in supported_languages:
                if lang["ui_support"]:
                    display_name = f"{lang['flag']} {lang['native_name']}"
                    language_options[display_name] = lang["code"]
            
            # Find current selection
            current_lang = preferences.ui_language.value
            current_display = None
            for display, code in language_options.items():
                if code == current_lang:
                    current_display = display
                    break
            
            if current_display is None:
                current_display = list(language_options.keys())[0]
            
            # Render selector
            selected_display = st.selectbox(
                self.interface.get_ui_text("language_selector", current_lang),
                options=list(language_options.keys()),
                index=list(language_options.keys()).index(current_display),
                key=f"{key_prefix}language_selector",
                help=self.interface.get_help_text("language_selection", current_lang)
            )
            
            selected_code = language_options[selected_display]
            changed = selected_code != current_lang
            
            if changed:
                # Update preferences
                preferences.ui_language = UILanguage(selected_code)
                self.interface.save_user_preferences(preferences)
                logger.info(f"UI language changed to {selected_code} for user {user_id}")
            
            return selected_code, changed
            
        except Exception as e:
            logger.error(f"Failed to render language selector: {e}")
            return "en", False
    
    def render_language_preferences_panel(self, user_id: str) -> bool:
        """
        Render comprehensive language preferences panel
        
        Returns:
            True if preferences were changed
        """
        try:
            preferences = self.interface.get_user_preferences(user_id)
            current_lang = preferences.ui_language.value
            changed = False
            
            st.subheader(self.interface.get_ui_text("preferences", current_lang))
            
            # UI Language Selection
            with st.expander(f"🌐 {self.interface.get_ui_text('language_selector', current_lang)}", expanded=True):
                ui_lang, ui_changed = self.render_language_selector(user_id, "prefs_")
                if ui_changed:
                    changed = True
                    current_lang = ui_lang  # Update for subsequent UI text
            
            # Response Language Preference
            with st.expander(f"💬 Response Language", expanded=False):
                response_options = {
                    "🤖 Auto (match query)": ResponseLanguage.AUTO,
                    "🇺🇸 English": ResponseLanguage.ENGLISH,
                    "🇯🇵 Japanese": ResponseLanguage.JAPANESE,
                    "📄 Original (source document)": ResponseLanguage.ORIGINAL
                }
                
                current_response = preferences.response_language
                current_response_display = None
                for display, value in response_options.items():
                    if value == current_response:
                        current_response_display = display
                        break
                
                if current_response_display is None:
                    current_response_display = list(response_options.keys())[0]
                
                selected_response_display = st.selectbox(
                    "Response Language",
                    options=list(response_options.keys()),
                    index=list(response_options.keys()).index(current_response_display),
                    key="response_language_selector",
                    help=self.interface.get_help_text("response_language", current_lang),
                    label_visibility="collapsed"
                )
                
                new_response_lang = response_options[selected_response_display]
                if new_response_lang != preferences.response_language:
                    preferences.response_language = new_response_lang
                    changed = True
            
            # Translation Settings
            with st.expander(f"🔄 Translation Settings", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    auto_translate_queries = st.checkbox(
                        "Auto-translate queries",
                        value=preferences.auto_translate_queries,
                        key="auto_translate_queries",
                        help="Automatically translate queries for cross-language search"
                    )
                    
                    if auto_translate_queries != preferences.auto_translate_queries:
                        preferences.auto_translate_queries = auto_translate_queries
                        changed = True
                
                with col2:
                    auto_translate_responses = st.checkbox(
                        "Auto-translate responses",
                        value=preferences.auto_translate_responses,
                        key="auto_translate_responses",
                        help="Automatically translate AI responses to preferred language"
                    )
                    
                    if auto_translate_responses != preferences.auto_translate_responses:
                        preferences.auto_translate_responses = auto_translate_responses
                        changed = True
                
                # Translation Quality Threshold
                quality_threshold = st.slider(
                    "Translation Quality Threshold",
                    min_value=0.5,
                    max_value=1.0,
                    value=preferences.translation_quality_threshold,
                    step=0.05,
                    key="translation_quality_threshold",
                    help=self.interface.get_help_text("translation_quality", current_lang),
                    format="%.0f%%"
                )
                
                if abs(quality_threshold - preferences.translation_quality_threshold) > 0.01:
                    preferences.translation_quality_threshold = quality_threshold
                    changed = True
                
                # Show Original Text
                show_original = st.checkbox(
                    self.interface.get_ui_text("show_original", current_lang),
                    value=preferences.show_original_text,
                    key="show_original_text",
                    help=self.interface.get_help_text("show_original", current_lang)
                )
                
                if show_original != preferences.show_original_text:
                    preferences.show_original_text = show_original
                    changed = True
            
            # Cultural Adaptation Settings
            with st.expander(f"🎭 Cultural Adaptation", expanded=False):
                adaptation_options = {
                    "None": CulturalAdaptation.NONE,
                    "Basic": CulturalAdaptation.BASIC,
                    "Business": CulturalAdaptation.BUSINESS,
                    "Full": CulturalAdaptation.FULL
                }
                
                current_adaptation = preferences.cultural_adaptation
                adaptation_index = list(adaptation_options.values()).index(current_adaptation)
                
                selected_adaptation = st.selectbox(
                    "Cultural Adaptation Level",
                    options=list(adaptation_options.keys()),
                    index=adaptation_index,
                    key="cultural_adaptation",
                    help=self.interface.get_help_text("cultural_adaptation", current_lang)
                )
                
                new_adaptation = adaptation_options[selected_adaptation]
                if new_adaptation != preferences.cultural_adaptation:
                    preferences.cultural_adaptation = new_adaptation
                    changed = True
                
                # Format Preferences
                col1, col2 = st.columns(2)
                
                with col1:
                    date_format = st.selectbox(
                        "Date Format",
                        options=["US", "JP", "ISO"],
                        index=["US", "JP", "ISO"].index(preferences.date_format),
                        key="date_format",
                        help="Date display format preference"
                    )
                    
                    if date_format != preferences.date_format:
                        preferences.date_format = date_format
                        changed = True
                
                with col2:
                    number_format = st.selectbox(
                        "Number Format",
                        options=["US", "JP"],
                        index=["US", "JP"].index(preferences.number_format),
                        key="number_format",
                        help="Number and currency display format"
                    )
                    
                    if number_format != preferences.number_format:
                        preferences.number_format = number_format
                        changed = True
                
                # Formality Level
                formality_options = ["casual", "business", "formal"]
                formality_index = formality_options.index(preferences.preferred_formality)
                
                formality = st.selectbox(
                    "Preferred Formality Level",
                    options=formality_options,
                    index=formality_index,
                    key="preferred_formality",
                    help="Preferred formality level for responses"
                )
                
                if formality != preferences.preferred_formality:
                    preferences.preferred_formality = formality
                    changed = True
            
            # Fallback Languages
            with st.expander(f"🔄 Fallback Languages", expanded=False):
                st.write("Language priority order for document search:")
                
                available_languages = ["ja", "en", "zh-cn", "ko", "es", "fr", "de"]
                current_fallbacks = preferences.fallback_languages.copy()
                
                # Allow reordering of fallback languages
                new_fallbacks = st.multiselect(
                    "Fallback Language Order",
                    options=available_languages,
                    default=current_fallbacks,
                    key="fallback_languages",
                    help="Languages to search when no results found in primary language"
                )
                
                if new_fallbacks != preferences.fallback_languages:
                    preferences.fallback_languages = new_fallbacks
                    changed = True
            
            # Save changes
            if changed:
                success = self.interface.save_user_preferences(preferences)
                if success:
                    st.success(f"✅ Preferences saved successfully!")
                else:
                    st.error(f"❌ Failed to save preferences")
            
            return changed
            
        except Exception as e:
            logger.error(f"Failed to render language preferences panel: {e}")
            st.error(f"Error loading language preferences: {e}")
            return False
    
    def render_multilingual_response(self, response: Any, user_preferences: UserLanguagePreferences) -> None:
        """Render a multilingual response with translations and sources"""
        try:
            current_lang = user_preferences.ui_language.value
            
            # Main response content
            st.markdown(response.content)
            
            # Show original text if available and requested
            if (user_preferences.show_original_text and 
                response.original_content and 
                response.original_content != response.content):
                
                with st.expander(f"📄 {self.interface.get_ui_text('show_original', current_lang)}"):
                    st.markdown(f"**Original ({response.original_language}):**")
                    st.markdown(response.original_content)
                    
                    if response.confidence < 1.0:
                        confidence_pct = int(response.confidence * 100)
                        st.caption(
                            self.interface.get_ui_text(
                                "translation_confidence", 
                                current_lang, 
                                confidence=confidence_pct
                            )
                        )
            
            # Show sources with multilingual support
            if response.sources:
                self.render_multilingual_sources(response.sources, user_preferences)
                
        except Exception as e:
            logger.error(f"Failed to render multilingual response: {e}")
            st.error("Error displaying response")
    
    def render_multilingual_sources(self, sources: List[Dict[str, Any]], 
                                  user_preferences: UserLanguagePreferences) -> None:
        """Render sources with multilingual support"""
        try:
            current_lang = user_preferences.ui_language.value
            
            with st.expander(f"📚 {self.interface.get_ui_text('sources', current_lang)} ({len(sources)})"):
                for i, source in enumerate(sources):
                    with st.container():
                        # Source header
                        filename = source.get('filename', f'Document {i+1}')
                        page_num = source.get('page_number')
                        
                        if page_num:
                            header = f"**{filename}** - {self.interface.get_ui_text('page_reference', current_lang, page=page_num)}"
                        else:
                            header = f"**{filename}**"
                        
                        st.markdown(header)
                        
                        # Source content
                        content = source.get('content', '')
                        if content:
                            st.markdown(f"*{content[:200]}{'...' if len(content) > 200 else ''}*")
                        
                        # Show translation if available
                        if (user_preferences.show_original_text and 
                            'translated_content' in source and 
                            source['translated_content'] != content):
                            
                            st.markdown(f"**Translation:**")
                            translated = source['translated_content']
                            st.markdown(f"*{translated[:200]}{'...' if len(translated) > 200 else ''}*")
                            
                            # Show translation confidence
                            if 'translation_confidence' in source:
                                confidence = int(source['translation_confidence'] * 100)
                                st.caption(f"Translation confidence: {confidence}%")
                        
                        # Language indicator
                        detected_lang = source.get('detected_language', 'unknown')
                        if detected_lang != 'unknown':
                            lang_flag = {"en": "🇺🇸", "ja": "🇯🇵"}.get(detected_lang, "🌐")
                            st.caption(f"{lang_flag} {detected_lang.upper()}")
                        
                        if i < len(sources) - 1:
                            st.divider()
                            
        except Exception as e:
            logger.error(f"Failed to render multilingual sources: {e}")
            st.error("Error displaying sources")
    
    def render_language_status_indicator(self, user_id: str) -> None:
        """Render current language status indicator"""
        try:
            preferences = self.interface.get_user_preferences(user_id)
            current_lang = preferences.ui_language.value
            
            # Language status in sidebar or header
            lang_info = {"en": {"flag": "🇺🇸", "name": "English"}, 
                        "ja": {"flag": "🇯🇵", "name": "日本語"}}
            
            current_info = lang_info.get(current_lang, {"flag": "🌐", "name": current_lang})
            
            st.caption(f"{current_info['flag']} {current_info['name']}")
            
            # Show cross-language search status
            if preferences.auto_translate_queries:
                st.caption("🌐 Cross-language search enabled")
            
        except Exception as e:
            logger.error(f"Failed to render language status: {e}")
    
    def render_query_language_detector(self, query: str, user_id: str) -> Optional[str]:
        """Render query language detection indicator"""
        try:
            if not query or len(query.strip()) < 3:
                return None
            
            preferences = self.interface.get_user_preferences(user_id)
            context = self.interface.process_multilingual_query(query, user_id)
            
            detected_lang = context.detected_language
            current_ui_lang = preferences.ui_language.value
            
            # Show language detection if different from UI language
            if detected_lang != current_ui_lang:
                lang_names = {"en": "English", "ja": "Japanese", "zh-cn": "Chinese"}
                detected_name = lang_names.get(detected_lang, detected_lang.upper())
                
                st.info(
                    self.interface.get_ui_text(
                        "language_detected", 
                        current_ui_lang, 
                        language=detected_name
                    )
                )
                
                # Show cross-language search status
                if len(context.search_languages) > 1:
                    st.info(self.interface.get_ui_text("cross_language_search", current_ui_lang))
            
            return detected_lang
            
        except Exception as e:
            logger.error(f"Failed to render query language detector: {e}")
            return None
    
    def render_error_message(self, error_type: str, user_id: str, **kwargs) -> None:
        """Render localized error message"""
        try:
            preferences = self.interface.get_user_preferences(user_id)
            current_lang = preferences.ui_language.value
            
            error_message = self.interface.get_error_message(error_type, current_lang, **kwargs)
            st.error(error_message)
            
        except Exception as e:
            logger.error(f"Failed to render error message: {e}")
            st.error(f"Error: {error_type}")


# Global instance
multilingual_components = MultilingualComponents()