"""
Language Detection and Processing Service for Kurachi AI
Implements automatic language detection for uploaded documents with language-specific processing pipelines
"""
import re
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from langchain.schema import Document as LangChainDocument

from config import config
from utils.logger import get_logger
from services.sota_translation_orchestrator import Language, sota_translation_orchestrator

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Mixed language detection thresholds
MIXED_MIN_SECONDARY = 0.12   # Secondary language must be at least 12% (lowered for better detection)
MIN_TEXT_CHARS = 40          # Minimum text length to consider mixed
MIN_COMBINED_CONFIDENCE = 0.80  # Combined confidence threshold (lowered slightly)

logger = get_logger("language_detection")


class DocumentLanguage(Enum):
    """Extended language support for document processing"""
    JAPANESE = "ja"
    ENGLISH = "en"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    KOREAN = "ko"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class LanguageDetectionResult:
    """Result of language detection analysis"""
    primary_language: DocumentLanguage
    confidence: float
    language_distribution: Dict[str, float]
    is_mixed_language: bool
    segments: List[Dict[str, Any]]
    processing_recommendations: Dict[str, Any]


@dataclass
class LanguageSegment:
    """A segment of text with detected language"""
    text: str
    language: DocumentLanguage
    confidence: float
    start_position: int
    end_position: int
    segment_type: str  # 'paragraph', 'sentence', 'phrase'


def is_mixed_language_robust(text: str) -> bool:
    """Robust mixed-language detection using character analysis + langdetect"""
    if len(text) < MIN_TEXT_CHARS:
        return False
    
    # First, use character-based analysis to detect script mixing
    script_counts = {
        'latin': len(re.findall(r'[a-zA-Z]', text)),
        'japanese': len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text)),
        'korean': len(re.findall(r'[\uAC00-\uD7AF]', text)),
        'arabic': len(re.findall(r'[\u0600-\u06FF]', text)),
        'cyrillic': len(re.findall(r'[\u0400-\u04FF]', text))
    }
    
    # Count non-zero scripts
    active_scripts = {script: count for script, count in script_counts.items() if count > 0}
    
    # If we have multiple scripts with significant presence, it's likely mixed
    if len(active_scripts) >= 2:
        total_chars = sum(active_scripts.values())
        if total_chars > 0:
            # Check if any secondary script has at least 15% presence
            sorted_scripts = sorted(active_scripts.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_scripts) >= 2:
                primary_count = sorted_scripts[0][1]
                secondary_count = sorted_scripts[1][1]
                secondary_ratio = secondary_count / total_chars
                
                if secondary_ratio >= 0.15:  # 15% threshold for secondary script
                    return True
    
    # Fallback to langdetect for languages using same script
    try:
        dists = detect_langs(text)
        probs = sorted([(d.lang, d.prob) for d in dists], key=lambda x: x[1], reverse=True)
        
        if len(probs) >= 2:
            _, p1 = probs[0]  # Primary language probability
            _, p2 = probs[1]  # Secondary language probability
            
            # Mixed if secondary language is significant
            return p2 >= MIXED_MIN_SECONDARY and (p1 + p2) >= MIN_COMBINED_CONFIDENCE
        
    except Exception:
        pass
    
    return False


def fast_detect(text: str) -> str:
    """Fast language detection with character-class fallback for short text"""
    t = text.strip()
    
    # For very short text, use character-class heuristic first
    if len(t) < 15:
        char_hint = _charclass_language_hint(t)
        if char_hint in ("ja", "en", "ko", "ar", "zh-cn", "ru"):
            return char_hint
    
    # Use seeded langdetect for longer text
    try:
        return detect(t)
    except LangDetectException:
        # Fallback to character analysis
        return _charclass_language_hint(t)


def _charclass_language_hint(text: str) -> str:
    """Character-class based language hint with improved CJK detection"""
    if not text:
        return "unknown"
    
    # Count script occurrences for better CJK disambiguation
    ja_score, zh_score, ko_score = 0, 0, 0
    
    for ch in text:
        code = ord(ch)
        # Hiragana - definitively Japanese
        if 0x3040 <= code <= 0x309F:
            ja_score += 2
        # Katakana - definitively Japanese  
        elif 0x30A0 <= code <= 0x30FF:
            ja_score += 2
        # CJK Unified Ideographs (Kanji/Hanzi/Hanja)
        elif 0x4E00 <= code <= 0x9FFF:
            # Context-based scoring for shared characters
            if "。" in text or "〜" in text or "・" in text:  # JP punctuation
                ja_score += 1
            elif "，" in text or "、" not in text:  # CN comma style
                zh_score += 1
            else:
                # Default scoring for ambiguous cases
                ja_score += 0.5
                zh_score += 0.5
        # Hangul - definitively Korean
        elif 0xAC00 <= code <= 0xD7AF:
            ko_score += 2
    
    # Non-zero script detection for other languages
    if re.search(r'[\u0600-\u06FF]', text):  # Arabic
        return "ar"
    if re.search(r'[\u0400-\u04FF]', text):  # Cyrillic (Russian)
        return "ru"
    
    # CJK decision based on scores
    if ko_score > ja_score and ko_score > zh_score:
        return "ko"
    elif ja_score > zh_score:
        return "ja"
    elif zh_score > ja_score:
        return "zh-cn"
    elif ja_score > 0 or zh_score > 0:
        # For ambiguous cases, prefer Japanese in business contexts
        return "ja"
    
    # Default to English for Latin script
    if re.search(r'[a-zA-Z]', text):
        return "en"
    
    return "unknown"


class LanguageDetectionService:
    """Advanced language detection service with mixed-language support"""
    
    def __init__(self):
        self.supported_languages = {
            'ja': DocumentLanguage.JAPANESE,
            'en': DocumentLanguage.ENGLISH,
            'zh-cn': DocumentLanguage.CHINESE_SIMPLIFIED,
            'zh-tw': DocumentLanguage.CHINESE_TRADITIONAL,
            'ko': DocumentLanguage.KOREAN,
            'es': DocumentLanguage.SPANISH,
            'fr': DocumentLanguage.FRENCH,
            'de': DocumentLanguage.GERMAN,
            'it': DocumentLanguage.ITALIAN,
            'pt': DocumentLanguage.PORTUGUESE,
            'ru': DocumentLanguage.RUSSIAN,
            'ar': DocumentLanguage.ARABIC
        }
        
        # Character patterns for different writing systems
        self.character_patterns = {
            DocumentLanguage.JAPANESE: r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]',
            DocumentLanguage.CHINESE_SIMPLIFIED: r'[\u4E00-\u9FFF]',
            DocumentLanguage.CHINESE_TRADITIONAL: r'[\u4E00-\u9FFF]',
            DocumentLanguage.KOREAN: r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]',
            DocumentLanguage.ARABIC: r'[\u0600-\u06FF\u0750-\u077F]',
            DocumentLanguage.RUSSIAN: r'[\u0400-\u04FF]',
            DocumentLanguage.ENGLISH: r'[a-zA-Z]'
        }
        
        logger.info("Language detection service initialized")
    
    def detect_document_language(self, text: str, detailed: bool = True) -> LanguageDetectionResult:
        """
        Detect language(s) in document text with detailed analysis
        
        Args:
            text: Document text to analyze
            detailed: Whether to perform detailed segment analysis
            
        Returns:
            LanguageDetectionResult with comprehensive language information
        """
        if not text or not text.strip():
            return LanguageDetectionResult(
                primary_language=DocumentLanguage.UNKNOWN,
                confidence=0.0,
                language_distribution={},
                is_mixed_language=False,
                segments=[],
                processing_recommendations={"error": "Empty text provided"}
            )
        
        try:
            # Primary language detection
            primary_result = self._detect_primary_language(text)
            
            # Character-based analysis
            char_analysis = self._analyze_character_distribution(text)
            
            # Segment analysis for mixed languages (if detailed)
            segments = []
            is_mixed = is_mixed_language_robust(text)  # Use robust detection
            
            if detailed and len(text) > MIN_TEXT_CHARS:  # Only for substantial text
                if is_mixed:
                    segments = self._segment_by_language_robust(text)  # Use robust segmentation
                else:
                    segments = self._segment_by_language(text)  # Fallback to original
            
            # Generate processing recommendations
            recommendations = self._generate_processing_recommendations(
                primary_result['language'],
                char_analysis,
                is_mixed,
                segments
            )
            
            return LanguageDetectionResult(
                primary_language=primary_result['language'],
                confidence=primary_result['confidence'],
                language_distribution=primary_result['distribution'],
                is_mixed_language=is_mixed,
                segments=segments,
                processing_recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return LanguageDetectionResult(
                primary_language=DocumentLanguage.UNKNOWN,
                confidence=0.0,
                language_distribution={},
                is_mixed_language=False,
                segments=[],
                processing_recommendations={"error": str(e)}
            )
    
    def _detect_primary_language(self, text: str) -> Dict[str, Any]:
        """Detect primary language using multiple methods"""
        try:
            # Method 1: langdetect library
            langdetect_results = detect_langs(text)
            langdetect_primary = langdetect_results[0] if langdetect_results else None
            
            # Method 2: Character-based analysis
            char_analysis = self._analyze_character_distribution(text)
            
            # Method 3: Pattern-based detection for Asian languages
            pattern_analysis = self._pattern_based_detection(text)
            
            # Combine results with weighted scoring
            language_scores = {}
            
            # Weight langdetect results (40% weight)
            if langdetect_primary:
                lang_code = langdetect_primary.lang
                if lang_code in self.supported_languages:
                    detected_lang = self.supported_languages[lang_code]
                    language_scores[detected_lang] = langdetect_primary.prob * 0.4
            
            # Weight character analysis (35% weight)
            for lang, ratio in char_analysis.items():
                if lang in language_scores:
                    language_scores[lang] += ratio * 0.35
                else:
                    language_scores[lang] = ratio * 0.35
            
            # Weight pattern analysis (25% weight)
            for lang, confidence in pattern_analysis.items():
                if lang in language_scores:
                    language_scores[lang] += confidence * 0.25
                else:
                    language_scores[lang] = confidence * 0.25
            
            # Find primary language
            if language_scores:
                primary_lang = max(language_scores.keys(), key=lambda k: language_scores[k])
                primary_confidence = language_scores[primary_lang]
                
                # Normalize distribution
                total_score = sum(language_scores.values())
                distribution = {lang.value: score/total_score for lang, score in language_scores.items()}
                
                return {
                    'language': primary_lang,
                    'confidence': min(1.0, primary_confidence),
                    'distribution': distribution
                }
            else:
                # Fallback to English
                return {
                    'language': DocumentLanguage.ENGLISH,
                    'confidence': 0.1,
                    'distribution': {'en': 1.0}
                }
                
        except LangDetectException as e:
            logger.warning(f"langdetect failed: {e}")
            # Fallback to character analysis only
            char_analysis = self._analyze_character_distribution(text)
            if char_analysis:
                primary_lang = max(char_analysis.keys(), key=lambda k: char_analysis[k])
                return {
                    'language': primary_lang,
                    'confidence': char_analysis[primary_lang],
                    'distribution': {lang.value: score for lang, score in char_analysis.items()}
                }
            else:
                return {
                    'language': DocumentLanguage.UNKNOWN,
                    'confidence': 0.0,
                    'distribution': {}
                }
        except Exception as e:
            logger.error(f"Primary language detection failed: {e}")
            return {
                'language': DocumentLanguage.UNKNOWN,
                'confidence': 0.0,
                'distribution': {}
            }
    
    def _analyze_character_distribution(self, text: str) -> Dict[DocumentLanguage, float]:
        """Analyze character distribution to identify languages"""
        char_counts = {}
        total_chars = len(text)
        
        if total_chars == 0:
            return {}
        
        for language, pattern in self.character_patterns.items():
            matches = re.findall(pattern, text)
            char_counts[language] = len(matches) / total_chars
        
        # Filter out languages with very low presence
        return {lang: ratio for lang, ratio in char_counts.items() if ratio > 0.01}
    
    def _pattern_based_detection(self, text: str) -> Dict[DocumentLanguage, float]:
        """Pattern-based detection for specific language features"""
        patterns = {}
        
        # Japanese-specific patterns
        hiragana_count = len(re.findall(r'[\u3040-\u309F]', text))
        katakana_count = len(re.findall(r'[\u30A0-\u30FF]', text))
        kanji_count = len(re.findall(r'[\u4E00-\u9FAF]', text))
        
        if hiragana_count > 0 or katakana_count > 0:
            japanese_confidence = (hiragana_count + katakana_count + kanji_count) / len(text)
            patterns[DocumentLanguage.JAPANESE] = min(1.0, japanese_confidence * 2)
        
        # Korean-specific patterns
        hangul_count = len(re.findall(r'[\uAC00-\uD7AF]', text))
        if hangul_count > 0:
            patterns[DocumentLanguage.KOREAN] = min(1.0, hangul_count / len(text) * 2)
        
        # Arabic-specific patterns
        arabic_count = len(re.findall(r'[\u0600-\u06FF]', text))
        if arabic_count > 0:
            patterns[DocumentLanguage.ARABIC] = min(1.0, arabic_count / len(text) * 2)
        
        # English-specific patterns (common English words)
        english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
        english_matches = sum(1 for word in english_indicators if word in text.lower())
        if english_matches > 0:
            patterns[DocumentLanguage.ENGLISH] = min(1.0, english_matches / 10)
        
        return patterns
    
    def _segment_by_language_robust(self, text: str) -> List[Dict[str, Any]]:
        """Robust sentence-level language segmentation"""
        # Split by sentence boundaries (Japanese and English)
        SENTENCE_SPLIT = re.compile(r'(?<=[。．.!?？！])\s+')
        
        segments = []
        chunks = SENTENCE_SPLIT.split(text)
        
        for chunk in chunks:
            if not chunk.strip():
                continue
            
            # Detect language for this chunk
            lang = fast_detect(chunk)
            segments.append({"lang": lang, "text": chunk.strip()})
        
        # Merge contiguous segments with same language
        merged = []
        for seg in segments:
            if merged and merged[-1]["lang"] == seg["lang"]:
                merged[-1]["text"] += " " + seg["text"]
            else:
                merged.append(seg)
        
        # Convert to expected format
        result_segments = []
        current_position = 0
        
        for i, seg in enumerate(merged):
            segment_info = {
                'text': seg["text"],
                'language': seg["lang"],
                'confidence': 0.8,  # High confidence for merged segments
                'start_position': current_position,
                'end_position': current_position + len(seg["text"]),
                'segment_type': 'sentence_group'
            }
            result_segments.append(segment_info)
            current_position += len(seg["text"]) + 1
        
        return result_segments
    
    def _segment_by_language(self, text: str) -> List[Dict[str, Any]]:
        """Segment text by language for mixed-language documents"""
        segments = []
        
        try:
            # Split text into sentences/paragraphs
            sentences = re.split(r'[.!?。！？]\s+', text)
            current_position = 0
            
            for sentence in sentences:
                if not sentence.strip():
                    current_position += len(sentence) + 1
                    continue
                
                # Detect language for this segment
                try:
                    segment_lang_code = detect(sentence)
                    segment_lang = self.supported_languages.get(
                        segment_lang_code, 
                        DocumentLanguage.UNKNOWN
                    )
                    
                    # Get confidence from character analysis
                    char_analysis = self._analyze_character_distribution(sentence)
                    confidence = char_analysis.get(segment_lang, 0.5)
                    
                except LangDetectException:
                    # Fallback to character analysis
                    char_analysis = self._analyze_character_distribution(sentence)
                    if char_analysis:
                        segment_lang = max(char_analysis.keys(), key=lambda k: char_analysis[k])
                        confidence = char_analysis[segment_lang]
                    else:
                        segment_lang = DocumentLanguage.UNKNOWN
                        confidence = 0.0
                
                segment_info = {
                    'text': sentence,
                    'language': segment_lang.value,
                    'confidence': confidence,
                    'start_position': current_position,
                    'end_position': current_position + len(sentence),
                    'segment_type': 'sentence'
                }
                
                segments.append(segment_info)
                current_position += len(sentence) + 1
            
            return segments
            
        except Exception as e:
            logger.error(f"Text segmentation failed: {e}")
            return []
    
    def _is_mixed_language(self, segments: List[Dict[str, Any]], primary_language: DocumentLanguage) -> bool:
        """Determine if document contains mixed languages"""
        if not segments or len(segments) < 2:
            return False
        
        # Count segments in different languages
        language_counts = {}
        for segment in segments:
            lang = segment['language']
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Consider mixed if more than 20% of segments are in a different language
        primary_count = language_counts.get(primary_language.value, 0)
        total_segments = len(segments)
        
        if total_segments == 0:
            return False
        
        primary_ratio = primary_count / total_segments
        return primary_ratio < 0.8 and len(language_counts) > 1
    
    def _generate_processing_recommendations(
        self, 
        primary_language: DocumentLanguage,
        char_analysis: Dict[DocumentLanguage, float],
        is_mixed: bool,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate processing recommendations based on language analysis"""
        recommendations = {
            'primary_language': primary_language.value,
            'processing_strategy': 'standard',
            'translation_needed': False,
            'special_handling': [],
            'chunking_strategy': 'standard',
            'ocr_language_hints': []
        }
        
        # Language-specific processing recommendations
        if primary_language == DocumentLanguage.JAPANESE:
            recommendations['special_handling'].extend([
                'japanese_tokenization',
                'kanji_handling',
                'keigo_awareness'
            ])
            recommendations['ocr_language_hints'] = ['jpn', 'eng']
            recommendations['chunking_strategy'] = 'japanese_aware'
            
        elif primary_language == DocumentLanguage.CHINESE_SIMPLIFIED:
            recommendations['special_handling'].extend([
                'chinese_tokenization',
                'traditional_simplified_handling'
            ])
            recommendations['ocr_language_hints'] = ['chi_sim', 'eng']
            
        elif primary_language == DocumentLanguage.KOREAN:
            recommendations['special_handling'].extend([
                'korean_tokenization',
                'hangul_handling'
            ])
            recommendations['ocr_language_hints'] = ['kor', 'eng']
            
        elif primary_language == DocumentLanguage.ARABIC:
            recommendations['special_handling'].extend([
                'rtl_text_handling',
                'arabic_tokenization'
            ])
            recommendations['ocr_language_hints'] = ['ara', 'eng']
            
        # Mixed language handling
        if is_mixed:
            recommendations['processing_strategy'] = 'mixed_language'
            recommendations['special_handling'].append('segment_by_language')
            
            # Identify secondary languages
            if segments:
                lang_counts = {}
                for segment in segments:
                    lang = segment['language']
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                
                secondary_languages = [
                    lang for lang, count in lang_counts.items() 
                    if lang != primary_language.value and count > 1
                ]
                recommendations['secondary_languages'] = secondary_languages
        
        # Translation recommendations
        if primary_language not in [DocumentLanguage.ENGLISH, DocumentLanguage.JAPANESE]:
            recommendations['translation_needed'] = True
            recommendations['translation_priority'] = 'high'
        elif primary_language == DocumentLanguage.JAPANESE:
            recommendations['translation_available'] = True
            recommendations['translation_priority'] = 'medium'
        
        return recommendations
    
    def process_documents_with_language_detection(
        self, 
        documents: List[LangChainDocument]
    ) -> List[LangChainDocument]:
        """Process documents with automatic language detection and metadata enhancement"""
        processed_documents = []
        
        for doc in documents:
            try:
                # Detect language for this document
                detection_result = self.detect_document_language(doc.page_content)
                
                # Enhance document metadata
                enhanced_metadata = {
                    **doc.metadata,
                    'detected_language': detection_result.primary_language.value,
                    'language_confidence': detection_result.confidence,
                    'is_mixed_language': detection_result.is_mixed_language,
                    'language_distribution': detection_result.language_distribution,
                    'processing_recommendations': detection_result.processing_recommendations
                }
                
                # Add language-specific processing hints
                if detection_result.processing_recommendations:
                    enhanced_metadata.update({
                        'chunking_strategy': detection_result.processing_recommendations.get('chunking_strategy', 'standard'),
                        'special_handling': detection_result.processing_recommendations.get('special_handling', []),
                        'ocr_language_hints': detection_result.processing_recommendations.get('ocr_language_hints', [])
                    })
                
                # Create enhanced document
                enhanced_doc = LangChainDocument(
                    page_content=doc.page_content,
                    metadata=enhanced_metadata
                )
                
                processed_documents.append(enhanced_doc)
                
                logger.debug(f"Language detection completed for document: {detection_result.primary_language.value} "
                           f"(confidence: {detection_result.confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Failed to process document with language detection: {e}")
                # Add document with error metadata
                error_metadata = {
                    **doc.metadata,
                    'detected_language': 'unknown',
                    'language_confidence': 0.0,
                    'language_detection_error': str(e)
                }
                
                error_doc = LangChainDocument(
                    page_content=doc.page_content,
                    metadata=error_metadata
                )
                processed_documents.append(error_doc)
        
        return processed_documents
    
    def get_language_specific_processing_pipeline(self, language: DocumentLanguage) -> Dict[str, Any]:
        """Get language-specific processing configuration"""
        pipelines = {
            DocumentLanguage.JAPANESE: {
                'tokenizer': 'japanese',
                'text_splitter': 'japanese_aware',
                'ocr_languages': ['jpn', 'eng'],
                'translation_available': True,
                'special_features': ['kanji_handling', 'keigo_detection']
            },
            DocumentLanguage.ENGLISH: {
                'tokenizer': 'english',
                'text_splitter': 'standard',
                'ocr_languages': ['eng'],
                'translation_available': True,
                'special_features': ['standard_processing']
            },
            DocumentLanguage.CHINESE_SIMPLIFIED: {
                'tokenizer': 'chinese',
                'text_splitter': 'chinese_aware',
                'ocr_languages': ['chi_sim', 'eng'],
                'translation_available': False,
                'special_features': ['chinese_tokenization']
            },
            DocumentLanguage.KOREAN: {
                'tokenizer': 'korean',
                'text_splitter': 'korean_aware',
                'ocr_languages': ['kor', 'eng'],
                'translation_available': False,
                'special_features': ['hangul_processing']
            }
        }
        
        return pipelines.get(language, pipelines[DocumentLanguage.ENGLISH])
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with metadata"""
        return [
            {'code': 'ja', 'name': 'Japanese', 'native_name': '日本語', 'processing_level': 'full'},
            {'code': 'en', 'name': 'English', 'native_name': 'English', 'processing_level': 'full'},
            {'code': 'zh-cn', 'name': 'Chinese (Simplified)', 'native_name': '简体中文', 'processing_level': 'basic'},
            {'code': 'zh-tw', 'name': 'Chinese (Traditional)', 'native_name': '繁體中文', 'processing_level': 'basic'},
            {'code': 'ko', 'name': 'Korean', 'native_name': '한국어', 'processing_level': 'basic'},
            {'code': 'es', 'name': 'Spanish', 'native_name': 'Español', 'processing_level': 'basic'},
            {'code': 'fr', 'name': 'French', 'native_name': 'Français', 'processing_level': 'basic'},
            {'code': 'de', 'name': 'German', 'native_name': 'Deutsch', 'processing_level': 'basic'},
            {'code': 'it', 'name': 'Italian', 'native_name': 'Italiano', 'processing_level': 'basic'},
            {'code': 'pt', 'name': 'Portuguese', 'native_name': 'Português', 'processing_level': 'basic'},
            {'code': 'ru', 'name': 'Russian', 'native_name': 'Русский', 'processing_level': 'basic'},
            {'code': 'ar', 'name': 'Arabic', 'native_name': 'العربية', 'processing_level': 'basic'}
        ]
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection method expected by validators.
        Returns a plain BCP-47 language code string.
        """
        try:
            result = self.detect_document_language(text, detailed=False)
            # Return the language code as a string, ensure it's not None
            lang_code = result.primary_language.value
            if lang_code and lang_code != "unknown":
                return lang_code
            else:
                # Fallback to fast detection
                return fast_detect(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            # Fallback to fast detection
            try:
                return fast_detect(text)
            except Exception:
                return "en"  # Ultimate fallback to English


# Global language detection service instance
language_detection_service = LanguageDetectionService()