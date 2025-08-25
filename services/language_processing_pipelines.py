"""
Language-Specific Processing Pipelines for Kurachi AI
Implements optimized text extraction and processing for different languages
"""
import re
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from langchain.schema import Document as LangChainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

from services.language_detection import DocumentLanguage, language_detection_service
from utils.logger import get_logger

logger = get_logger("language_processing_pipelines")


@dataclass
class ProcessingPipelineResult:
    """Result of language-specific processing"""
    documents: List[LangChainDocument]
    processing_metadata: Dict[str, Any]
    language_specific_features: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class LanguageProcessingPipeline(ABC):
    """Abstract base class for language-specific processing pipelines"""
    
    @abstractmethod
    def process_documents(self, documents: List[LangChainDocument]) -> ProcessingPipelineResult:
        """Process documents with language-specific optimizations"""
        pass
    
    @abstractmethod
    def get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get language-optimized text splitter"""
        pass
    
    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for optimal extraction"""
        pass
    
    @abstractmethod
    def get_ocr_languages(self) -> List[str]:
        """Get OCR language codes for this language"""
        pass


class JapaneseProcessingPipeline(LanguageProcessingPipeline):
    """Japanese-specific processing pipeline with advanced features"""
    
    def __init__(self):
        self.language = DocumentLanguage.JAPANESE
        
        # Japanese text patterns
        self.hiragana_pattern = re.compile(r'[\u3040-\u309F]')
        self.katakana_pattern = re.compile(r'[\u30A0-\u30FF]')
        self.kanji_pattern = re.compile(r'[\u4E00-\u9FAF]')
        self.japanese_punctuation = re.compile(r'[。、！？：；]')
        
        # Business Japanese patterns
        self.keigo_patterns = [
            r'いらっしゃい', r'ございます', r'させていただ', r'申し上げ',
            r'お疲れ様', r'よろしくお願い', r'恐れ入り', r'失礼'
        ]
        
        logger.info("Japanese processing pipeline initialized")
    
    def process_documents(self, documents: List[LangChainDocument]) -> ProcessingPipelineResult:
        """Process documents with Japanese-specific optimizations"""
        try:
            processed_docs = []
            japanese_features = {
                'total_kanji_count': 0,
                'total_hiragana_count': 0,
                'total_katakana_count': 0,
                'keigo_usage': 0,
                'business_terms_detected': [],
                'text_complexity': 'medium'
            }
            
            for doc in documents:
                # Preprocess text
                processed_text = self.preprocess_text(doc.page_content)
                
                # Analyze Japanese text features
                features = self._analyze_japanese_features(processed_text)
                
                # Update global features
                japanese_features['total_kanji_count'] += features['kanji_count']
                japanese_features['total_hiragana_count'] += features['hiragana_count']
                japanese_features['total_katakana_count'] += features['katakana_count']
                japanese_features['keigo_usage'] += features['keigo_count']
                japanese_features['business_terms_detected'].extend(features['business_terms'])
                
                # Enhanced metadata
                enhanced_metadata = {
                    **doc.metadata,
                    'japanese_features': features,
                    'text_complexity': self._assess_text_complexity(features),
                    'reading_level': self._estimate_reading_level(features),
                    'business_context': features['keigo_count'] > 0 or len(features['business_terms']) > 0
                }
                
                # Create processed document
                processed_doc = LangChainDocument(
                    page_content=processed_text,
                    metadata=enhanced_metadata
                )
                processed_docs.append(processed_doc)
            
            # Assess overall complexity
            if japanese_features['total_kanji_count'] > 100:
                japanese_features['text_complexity'] = 'high'
            elif japanese_features['total_kanji_count'] < 20:
                japanese_features['text_complexity'] = 'low'
            
            return ProcessingPipelineResult(
                documents=processed_docs,
                processing_metadata={
                    'pipeline': 'japanese',
                    'documents_processed': len(processed_docs),
                    'preprocessing_applied': True
                },
                language_specific_features=japanese_features,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Japanese processing pipeline failed: {e}")
            return ProcessingPipelineResult(
                documents=documents,  # Return original documents
                processing_metadata={'pipeline': 'japanese', 'error': str(e)},
                language_specific_features={},
                success=False,
                error_message=str(e)
            )
    
    def get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get Japanese-optimized text splitter"""
        return RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for Japanese due to information density
            chunk_overlap=100,
            separators=[
                "\n\n",  # Paragraph breaks
                "。\n",   # Japanese period with newline
                "。",     # Japanese period
                "！\n",   # Japanese exclamation with newline
                "！",     # Japanese exclamation
                "？\n",   # Japanese question with newline
                "？",     # Japanese question
                "、",     # Japanese comma
                "\n",     # Regular newline
                " ",      # Space
                ""        # Character level
            ],
            length_function=len,
            is_separator_regex=False
        )
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess Japanese text for optimal extraction"""
        # Normalize Japanese text
        processed_text = text
        
        # Normalize full-width characters to half-width where appropriate
        processed_text = self._normalize_japanese_characters(processed_text)
        
        # Clean up spacing around Japanese punctuation
        processed_text = re.sub(r'\s+([。、！？：；])', r'\1', processed_text)
        processed_text = re.sub(r'([。、！？：；])\s+', r'\1 ', processed_text)
        
        # Normalize line breaks
        processed_text = re.sub(r'\n\s*\n', '\n\n', processed_text)
        
        return processed_text.strip()
    
    def get_ocr_languages(self) -> List[str]:
        """Get OCR language codes for Japanese"""
        return ['jpn', 'eng']  # Japanese and English for mixed content
    
    def _analyze_japanese_features(self, text: str) -> Dict[str, Any]:
        """Analyze Japanese-specific text features"""
        features = {
            'kanji_count': len(self.kanji_pattern.findall(text)),
            'hiragana_count': len(self.hiragana_pattern.findall(text)),
            'katakana_count': len(self.katakana_pattern.findall(text)),
            'keigo_count': 0,
            'business_terms': [],
            'character_distribution': {}
        }
        
        # Count keigo usage
        for pattern in self.keigo_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            features['keigo_count'] += len(matches)
            if matches:
                features['business_terms'].extend(matches)
        
        # Calculate character distribution
        total_chars = len(text)
        if total_chars > 0:
            features['character_distribution'] = {
                'kanji_ratio': features['kanji_count'] / total_chars,
                'hiragana_ratio': features['hiragana_count'] / total_chars,
                'katakana_ratio': features['katakana_count'] / total_chars
            }
        
        return features
    
    def _normalize_japanese_characters(self, text: str) -> str:
        """Normalize Japanese character variations"""
        # Full-width to half-width for numbers and basic punctuation
        full_to_half = {
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
            '（': '(', '）': ')', '［': '[', '］': ']',
            '｛': '{', '｝': '}', '＜': '<', '＞': '>',
            '＋': '+', '－': '-', '＝': '=', '＊': '*',
            '／': '/', '＼': '\\', '｜': '|'
        }
        
        for full, half in full_to_half.items():
            text = text.replace(full, half)
        
        return text
    
    def _assess_text_complexity(self, features: Dict[str, Any]) -> str:
        """Assess Japanese text complexity based on features"""
        kanji_ratio = features.get('character_distribution', {}).get('kanji_ratio', 0)
        keigo_count = features.get('keigo_count', 0)
        
        if kanji_ratio > 0.3 or keigo_count > 5:
            return 'high'
        elif kanji_ratio > 0.15 or keigo_count > 0:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_reading_level(self, features: Dict[str, Any]) -> str:
        """Estimate Japanese reading level"""
        kanji_count = features.get('kanji_count', 0)
        keigo_count = features.get('keigo_count', 0)
        
        if kanji_count > 50 and keigo_count > 3:
            return 'business_advanced'
        elif kanji_count > 20 or keigo_count > 0:
            return 'business_intermediate'
        elif kanji_count > 10:
            return 'intermediate'
        else:
            return 'basic'


class EnglishProcessingPipeline(LanguageProcessingPipeline):
    """English-specific processing pipeline"""
    
    def __init__(self):
        self.language = DocumentLanguage.ENGLISH
        
        # Business English patterns
        self.business_patterns = [
            r'\b(pursuant to|in accordance with|notwithstanding)\b',
            r'\b(hereby|whereas|therefore|furthermore)\b',
            r'\b(stakeholder|deliverable|actionable|synergy)\b'
        ]
        
        logger.info("English processing pipeline initialized")
    
    def process_documents(self, documents: List[LangChainDocument]) -> ProcessingPipelineResult:
        """Process documents with English-specific optimizations"""
        try:
            processed_docs = []
            english_features = {
                'total_word_count': 0,
                'average_sentence_length': 0,
                'business_terms_detected': [],
                'readability_score': 'medium',
                'technical_content': False
            }
            
            for doc in documents:
                # Preprocess text
                processed_text = self.preprocess_text(doc.page_content)
                
                # Analyze English text features
                features = self._analyze_english_features(processed_text)
                
                # Update global features
                english_features['total_word_count'] += features['word_count']
                english_features['business_terms_detected'].extend(features['business_terms'])
                
                # Enhanced metadata
                enhanced_metadata = {
                    **doc.metadata,
                    'english_features': features,
                    'readability_level': features['readability_level'],
                    'business_context': len(features['business_terms']) > 0,
                    'technical_content': features['technical_indicators'] > 0
                }
                
                # Create processed document
                processed_doc = LangChainDocument(
                    page_content=processed_text,
                    metadata=enhanced_metadata
                )
                processed_docs.append(processed_doc)
            
            # Calculate average sentence length
            if processed_docs:
                total_sentences = sum(
                    doc.metadata.get('english_features', {}).get('sentence_count', 0) 
                    for doc in processed_docs
                )
                if total_sentences > 0:
                    english_features['average_sentence_length'] = english_features['total_word_count'] / total_sentences
            
            return ProcessingPipelineResult(
                documents=processed_docs,
                processing_metadata={
                    'pipeline': 'english',
                    'documents_processed': len(processed_docs),
                    'preprocessing_applied': True
                },
                language_specific_features=english_features,
                success=True
            )
            
        except Exception as e:
            logger.error(f"English processing pipeline failed: {e}")
            return ProcessingPipelineResult(
                documents=documents,
                processing_metadata={'pipeline': 'english', 'error': str(e)},
                language_specific_features={},
                success=False,
                error_message=str(e)
            )
    
    def get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get English-optimized text splitter"""
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",   # Paragraph breaks
                "\n",     # Line breaks
                ". ",     # Sentence endings
                "! ",     # Exclamations
                "? ",     # Questions
                "; ",     # Semicolons
                ", ",     # Commas
                " ",      # Spaces
                ""        # Character level
            ],
            length_function=len,
            is_separator_regex=False
        )
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess English text for optimal extraction"""
        # Normalize whitespace
        processed_text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        processed_text = re.sub(r'\b(\w+)\s+(\w+)\b', r'\1\2', processed_text)  # Fix split words
        processed_text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', processed_text)  # Fix sentence spacing
        
        # Normalize punctuation
        processed_text = re.sub(r'\s+([.!?,:;])', r'\1', processed_text)
        processed_text = re.sub(r'([.!?])\s+', r'\1 ', processed_text)
        
        return processed_text.strip()
    
    def get_ocr_languages(self) -> List[str]:
        """Get OCR language codes for English"""
        return ['eng']
    
    def _analyze_english_features(self, text: str) -> Dict[str, Any]:
        """Analyze English-specific text features"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        features = {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'business_terms': [],
            'technical_indicators': 0,
            'readability_level': 'medium'
        }
        
        # Detect business terms
        for pattern in self.business_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            features['business_terms'].extend(matches)
        
        # Detect technical content
        technical_indicators = [
            r'\b\d+\.\d+\b',  # Version numbers
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+\(\)\b',   # Function calls
            r'\b\w+\.\w+\b'   # Dot notation
        ]
        
        for pattern in technical_indicators:
            features['technical_indicators'] += len(re.findall(pattern, text))
        
        # Assess readability
        if features['average_word_length'] > 6 or len(features['business_terms']) > 5:
            features['readability_level'] = 'advanced'
        elif features['average_word_length'] < 4 and len(features['business_terms']) == 0:
            features['readability_level'] = 'basic'
        
        return features


class MixedLanguageProcessingPipeline(LanguageProcessingPipeline):
    """Processing pipeline for mixed-language documents"""
    
    def __init__(self):
        self.japanese_pipeline = JapaneseProcessingPipeline()
        self.english_pipeline = EnglishProcessingPipeline()
        
        logger.info("Mixed language processing pipeline initialized")
    
    def process_documents(self, documents: List[LangChainDocument]) -> ProcessingPipelineResult:
        """Process mixed-language documents with segment-aware processing"""
        try:
            processed_docs = []
            mixed_features = {
                'languages_detected': [],
                'segment_count': 0,
                'processing_strategies': []
            }
            
            for doc in documents:
                # Detect language segments
                detection_result = language_detection_service.detect_document_language(
                    doc.page_content, detailed=True
                )
                
                if detection_result.is_mixed_language and detection_result.segments:
                    # Process each segment with appropriate pipeline
                    processed_segments = []
                    
                    for segment_info in detection_result.segments:
                        segment_text = segment_info['text']
                        segment_lang = segment_info['language']
                        
                        # Choose appropriate pipeline
                        if segment_lang == 'ja':
                            segment_result = self.japanese_pipeline.preprocess_text(segment_text)
                        elif segment_lang == 'en':
                            segment_result = self.english_pipeline.preprocess_text(segment_text)
                        else:
                            segment_result = segment_text  # No specific processing
                        
                        processed_segments.append(segment_result)
                    
                    # Combine processed segments
                    processed_text = ' '.join(processed_segments)
                    mixed_features['segment_count'] += len(detection_result.segments)
                    mixed_features['languages_detected'] = list(detection_result.language_distribution.keys())
                    
                else:
                    # Single language processing
                    primary_lang = detection_result.primary_language
                    if primary_lang == DocumentLanguage.JAPANESE:
                        processed_text = self.japanese_pipeline.preprocess_text(doc.page_content)
                        mixed_features['processing_strategies'].append('japanese')
                    elif primary_lang == DocumentLanguage.ENGLISH:
                        processed_text = self.english_pipeline.preprocess_text(doc.page_content)
                        mixed_features['processing_strategies'].append('english')
                    else:
                        processed_text = doc.page_content
                        mixed_features['processing_strategies'].append('generic')
                
                # Enhanced metadata
                enhanced_metadata = {
                    **doc.metadata,
                    'mixed_language_processing': True,
                    'language_segments': detection_result.segments,
                    'primary_language': detection_result.primary_language.value,
                    'language_distribution': detection_result.language_distribution
                }
                
                # Create processed document
                processed_doc = LangChainDocument(
                    page_content=processed_text,
                    metadata=enhanced_metadata
                )
                processed_docs.append(processed_doc)
            
            return ProcessingPipelineResult(
                documents=processed_docs,
                processing_metadata={
                    'pipeline': 'mixed_language',
                    'documents_processed': len(processed_docs),
                    'preprocessing_applied': True
                },
                language_specific_features=mixed_features,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Mixed language processing pipeline failed: {e}")
            return ProcessingPipelineResult(
                documents=documents,
                processing_metadata={'pipeline': 'mixed_language', 'error': str(e)},
                language_specific_features={},
                success=False,
                error_message=str(e)
            )
    
    def get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get mixed-language optimized text splitter"""
        return RecursiveCharacterTextSplitter(
            chunk_size=900,  # Medium chunk size for mixed content
            chunk_overlap=150,
            separators=[
                "\n\n",   # Paragraph breaks
                "。\n",   # Japanese period with newline
                ". ",     # English period
                "。",     # Japanese period
                "！\n",   # Japanese exclamation with newline
                "! ",     # English exclamation
                "？\n",   # Japanese question with newline
                "? ",     # English question
                "、",     # Japanese comma
                ", ",     # English comma
                "\n",     # Line breaks
                " ",      # Spaces
                ""        # Character level
            ],
            length_function=len,
            is_separator_regex=False
        )
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess mixed-language text"""
        # Apply both Japanese and English preprocessing
        text = self.japanese_pipeline.preprocess_text(text)
        text = self.english_pipeline.preprocess_text(text)
        return text
    
    def get_ocr_languages(self) -> List[str]:
        """Get OCR language codes for mixed content"""
        return ['jpn', 'eng']  # Support both Japanese and English


class LanguageProcessingPipelineFactory:
    """Factory for creating language-specific processing pipelines"""
    
    def __init__(self):
        self.pipelines = {
            DocumentLanguage.JAPANESE: JapaneseProcessingPipeline(),
            DocumentLanguage.ENGLISH: EnglishProcessingPipeline(),
            DocumentLanguage.MIXED: MixedLanguageProcessingPipeline()
        }
        
        logger.info("Language processing pipeline factory initialized")
    
    def get_pipeline(self, language: DocumentLanguage, is_mixed: bool = False) -> LanguageProcessingPipeline:
        """Get appropriate processing pipeline for language"""
        if is_mixed:
            return self.pipelines[DocumentLanguage.MIXED]
        
        return self.pipelines.get(language, self.pipelines[DocumentLanguage.ENGLISH])
    
    def process_documents_with_language_awareness(
        self, 
        documents: List[LangChainDocument]
    ) -> ProcessingPipelineResult:
        """Process documents with automatic language detection and appropriate pipeline selection"""
        try:
            if not documents:
                return ProcessingPipelineResult(
                    documents=[],
                    processing_metadata={'error': 'No documents provided'},
                    language_specific_features={},
                    success=False,
                    error_message="No documents provided"
                )
            
            # Detect language for the first document to determine overall strategy
            sample_text = documents[0].page_content
            detection_result = language_detection_service.detect_document_language(sample_text, detailed=True)
            
            # Choose pipeline based on detection result
            if detection_result.is_mixed_language:
                pipeline = self.get_pipeline(DocumentLanguage.MIXED, is_mixed=True)
                logger.info("Using mixed language processing pipeline")
            else:
                pipeline = self.get_pipeline(detection_result.primary_language)
                logger.info(f"Using {detection_result.primary_language.value} processing pipeline")
            
            # Process documents
            result = pipeline.process_documents(documents)
            
            # Add detection metadata to result
            result.processing_metadata.update({
                'language_detection': {
                    'primary_language': detection_result.primary_language.value,
                    'confidence': detection_result.confidence,
                    'is_mixed_language': detection_result.is_mixed_language,
                    'language_distribution': detection_result.language_distribution
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Language-aware document processing failed: {e}")
            return ProcessingPipelineResult(
                documents=documents,  # Return original documents
                processing_metadata={'error': str(e)},
                language_specific_features={},
                success=False,
                error_message=str(e)
            )
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages for processing"""
        return [lang.value for lang in self.pipelines.keys() if lang != DocumentLanguage.MIXED]


# Global pipeline factory instance
language_pipeline_factory = LanguageProcessingPipelineFactory()