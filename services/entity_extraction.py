"""
Entity extraction and recognition service for Kurachi AI
Implements NLP pipeline for named entity recognition with custom entity types
"""
import re
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

import spacy
from spacy.tokens import Doc, Span
from spacy.matcher import Matcher, PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config import config
from utils.logger import get_logger

logger = get_logger("entity_extraction")


@dataclass
class Entity:
    """Entity model for knowledge graph"""
    id: str
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float
    source_document_id: str
    source_page: Optional[int] = None
    context: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.properties is None:
            self.properties = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create Entity from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class EntityLink:
    """Entity linking result"""
    entity_id: str
    linked_entity_id: str
    confidence: float
    link_type: str  # 'exact_match', 'fuzzy_match', 'semantic_match'
    evidence: Dict[str, Any]


class CustomEntityPatterns:
    """Custom entity patterns for organizational knowledge"""
    
    def __init__(self):
        # Japanese business patterns
        self.japanese_patterns = {
            "COMPANY_JP": [
                r"株式会社[\w\s]+",
                r"[\w\s]+株式会社",
                r"[\w\s]+会社",
                r"[\w\s]+法人",
                r"[\w\s]+組合",
                r"[\w\s]+協会"
            ],
            "PERSON_JP": [
                r"[\u4e00-\u9faf]{1,4}[\s]*[\u4e00-\u9faf]{1,4}(?:さん|様|氏|先生|部長|課長|主任|取締役|社長|専務|常務)",
                r"[\u3040-\u309f]{2,8}[\s]*[\u3040-\u309f]{2,8}(?:さん|様|氏|先生)"
            ],
            "DEPARTMENT_JP": [
                r"[\w\s]*部(?:門)?",
                r"[\w\s]*課",
                r"[\w\s]*室",
                r"[\w\s]*チーム",
                r"[\w\s]*グループ"
            ],
            "PROJECT_JP": [
                r"[\w\s]*プロジェクト",
                r"[\w\s]*計画",
                r"[\w\s]*企画",
                r"[\w\s]*案件"
            ]
        }
        
        # English business patterns
        self.english_patterns = {
            "COMPANY_EN": [
                r"\b[A-Z][a-zA-Z\s&]+ (?:Inc|Corp|LLC|Ltd|Co|Company|Corporation|Limited)\b",
                r"\b[A-Z][a-zA-Z\s&]+ (?:Group|Holdings|Enterprises|Solutions|Systems|Technologies)\b"
            ],
            "DEPARTMENT_EN": [
                r"\b(?:Department of|Dept of|Division of)[\s\w]+",
                r"\b[\w\s]+ (?:Department|Dept|Division|Team|Group|Unit)\b",
                r"\b(?:HR|IT|Finance|Marketing|Sales|Operations|Engineering|R&D|Legal|Admin)\b"
            ],
            "PROJECT_EN": [
                r"\bProject\s+[A-Z][\w\s]+",
                r"\b[A-Z][\w\s]+ Project\b",
                r"\b(?:Initiative|Program|Campaign|Phase)\s+[A-Z][\w\s]+",
                r"\b[A-Z][\w\s]+ (?:Initiative|Program|Campaign|Phase)\b"
            ],
            "ROLE_EN": [
                r"\b(?:CEO|CTO|CFO|COO|VP|Director|Manager|Lead|Senior|Junior|Principal|Chief)\s+[\w\s]+",
                r"\b[\w\s]+ (?:Manager|Director|Lead|Engineer|Analyst|Specialist|Coordinator|Administrator)\b"
            ]
        }
        
        # Technical patterns
        self.technical_patterns = {
            "EMAIL": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            "PHONE_JP": [r"\b0\d{1,4}-\d{1,4}-\d{4}\b", r"\b0\d{9,10}\b"],
            "PHONE_EN": [r"\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"],
            "URL": [r"https?://[^\s]+"],
            "DATE_JP": [
                r"\d{4}年\d{1,2}月\d{1,2}日",
                r"令和\d{1,2}年\d{1,2}月\d{1,2}日",
                r"平成\d{1,2}年\d{1,2}月\d{1,2}日"
            ],
            "MONEY_JP": [r"\d+円", r"\d+万円", r"\d+億円"],
            "MONEY_EN": [r"\$\d+(?:,\d{3})*(?:\.\d{2})?", r"\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars?)"],
            "VERSION": [r"v?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?"],
            "IP_ADDRESS": [r"\b(?:\d{1,3}\.){3}\d{1,3}\b"],
            "FILE_PATH": [r"[a-zA-Z]:\\[^\s]+", r"/[^\s]+"],
            "DOCUMENT_ID": [r"\b[A-Z]{2,}-\d{4,}\b", r"\b\d{4}-\d{2}-\d{2}-[A-Z0-9]+\b"]
        }
        
        # Organizational knowledge patterns
        self.organizational_patterns = {
            "SKILL_JP": [
                r"[\w\s]*スキル",
                r"[\w\s]*技術",
                r"[\w\s]*能力",
                r"[\w\s]*経験"
            ],
            "SKILL_EN": [
                r"\b(?:Python|Java|JavaScript|React|Angular|Vue|Node\.js|Docker|Kubernetes|AWS|Azure|GCP)\b",
                r"\b[\w\s]+ (?:skills?|experience|expertise|knowledge)\b",
                r"\b(?:programming|development|design|management|leadership|communication) skills?\b"
            ],
            "PROCESS_JP": [
                r"[\w\s]*プロセス",
                r"[\w\s]*手順",
                r"[\w\s]*フロー",
                r"[\w\s]*ワークフロー"
            ],
            "PROCESS_EN": [
                r"\b[\w\s]+ (?:process|procedure|workflow|methodology)\b",
                r"\b(?:Agile|Scrum|Kanban|DevOps|CI/CD) (?:process|methodology)?\b"
            ],
            "CONCEPT_JP": [
                r"[\w\s]*概念",
                r"[\w\s]*理論",
                r"[\w\s]*原則",
                r"[\w\s]*方法論"
            ],
            "CONCEPT_EN": [
                r"\b[\w\s]+ (?:concept|theory|principle|methodology|framework)\b",
                r"\b(?:Machine Learning|AI|Data Science|Cloud Computing|Microservices)\b"
            ]
        }


class EntityExtractor:
    """Main entity extraction service"""
    
    def __init__(self):
        self.nlp_en = None
        self.nlp_ja = None
        self.custom_patterns = CustomEntityPatterns()
        self.matcher_en = None
        self.matcher_ja = None
        self.phrase_matcher_en = None
        self.phrase_matcher_ja = None
        self.entity_cache = {}
        self.similarity_threshold = 0.85
        self._init_models()
    
    def _init_models(self):
        """Initialize spaCy models and matchers"""
        try:
            # Load English model
            try:
                self.nlp_en = spacy.load("en_core_web_sm")
                logger.info("Loaded English spaCy model")
            except OSError:
                logger.warning("English spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp_en = None
            
            # Load Japanese model
            try:
                self.nlp_ja = spacy.load("ja_core_news_sm")
                logger.info("Loaded Japanese spaCy model")
            except OSError:
                logger.warning("Japanese spaCy model not found. Install with: python -m spacy download ja_core_news_sm")
                self.nlp_ja = None
            
            # Initialize matchers
            if self.nlp_en:
                self.matcher_en = Matcher(self.nlp_en.vocab)
                self.phrase_matcher_en = PhraseMatcher(self.nlp_en.vocab)
                self._setup_custom_patterns(self.nlp_en, self.matcher_en, "en")
            
            if self.nlp_ja:
                self.matcher_ja = Matcher(self.nlp_ja.vocab)
                self.phrase_matcher_ja = PhraseMatcher(self.nlp_ja.vocab)
                self._setup_custom_patterns(self.nlp_ja, self.matcher_ja, "ja")
                
        except Exception as e:
            logger.error("Failed to initialize NLP models", error=e)
            raise
    
    def _setup_custom_patterns(self, nlp, matcher, language: str):
        """Setup custom patterns for matcher"""
        try:
            patterns_dict = {}
            
            if language == "en":
                patterns_dict.update(self.custom_patterns.english_patterns)
            elif language == "ja":
                patterns_dict.update(self.custom_patterns.japanese_patterns)
            
            patterns_dict.update(self.custom_patterns.technical_patterns)
            
            for label, patterns in patterns_dict.items():
                for pattern in patterns:
                    try:
                        # Convert regex to spaCy pattern (simplified approach)
                        # For more complex patterns, we'll use regex matching separately
                        pass
                    except Exception as e:
                        logger.warning(f"Failed to add pattern {pattern} for {label}", error=e)
                        
        except Exception as e:
            logger.error("Failed to setup custom patterns", error=e)
    
    def detect_language(self, text: str) -> str:
        """Detect text language (simple heuristic)"""
        # Count Japanese characters
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]', text))
        total_chars = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        
        if total_chars == 0:
            return "en"
        
        japanese_ratio = japanese_chars / total_chars
        # Lower threshold for mixed content - even small amount of Japanese indicates Japanese content
        return "ja" if japanese_ratio > 0.05 else "en"
    
    def extract_entities_spacy(self, text: str, document_id: str, page_number: Optional[int] = None) -> List[Entity]:
        """Extract entities using spaCy NER"""
        entities = []
        
        try:
            language = self.detect_language(text)
            nlp = self.nlp_ja if language == "ja" else self.nlp_en
            
            if not nlp:
                logger.warning(f"No {language} model available for entity extraction")
                return entities
            
            doc = nlp(text)
            
            for ent in doc.ents:
                # Skip very short entities or common words
                if len(ent.text.strip()) < 2:
                    continue
                
                # Calculate confidence based on entity type and context
                confidence = self._calculate_confidence(ent, doc)
                
                # Get context around entity
                context = self._get_entity_context(ent, doc)
                
                entity = Entity(
                    id=str(uuid.uuid4()),
                    text=ent.text.strip(),
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=confidence,
                    source_document_id=document_id,
                    source_page=page_number,
                    context=context,
                    properties={
                        "language": language,
                        "spacy_label": ent.label_,
                        "lemma": ent.lemma_ if hasattr(ent, 'lemma_') else None
                    }
                )
                
                entities.append(entity)
                
        except Exception as e:
            logger.error("Failed to extract entities with spaCy", error=e)
        
        return entities
    
    def extract_entities_regex(self, text: str, document_id: str, page_number: Optional[int] = None) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        
        try:
            language = self.detect_language(text)
            
            # Select appropriate patterns
            patterns_dict = {}
            if language == "ja":
                patterns_dict.update(self.custom_patterns.japanese_patterns)
            else:
                patterns_dict.update(self.custom_patterns.english_patterns)
            
            patterns_dict.update(self.custom_patterns.technical_patterns)
            patterns_dict.update(self.custom_patterns.organizational_patterns)
            
            for label, patterns in patterns_dict.items():
                for pattern in patterns:
                    try:
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        for match in matches:
                            entity_text = match.group().strip()
                            
                            # Skip very short matches
                            if len(entity_text) < 2:
                                continue
                            
                            # Calculate confidence based on pattern type
                            confidence = self._calculate_regex_confidence(label, entity_text)
                            
                            entity = Entity(
                                id=str(uuid.uuid4()),
                                text=entity_text,
                                label=label,
                                start_char=match.start(),
                                end_char=match.end(),
                                confidence=confidence,
                                source_document_id=document_id,
                                source_page=page_number,
                                context=self._get_regex_context(text, match.start(), match.end()),
                                properties={
                                    "language": language,
                                    "extraction_method": "regex",
                                    "pattern": pattern
                                }
                            )
                            
                            entities.append(entity)
                            
                    except Exception as e:
                        logger.warning(f"Failed to apply regex pattern {pattern}", error=e)
                        
        except Exception as e:
            logger.error("Failed to extract entities with regex", error=e)
        
        return entities
    
    def _calculate_confidence(self, ent: Span, doc: Doc) -> float:
        """Calculate confidence score for spaCy entity"""
        base_confidence = 0.7
        
        # Boost confidence for certain entity types
        high_confidence_types = {"PERSON", "ORG", "GPE", "MONEY", "DATE"}
        if ent.label_ in high_confidence_types:
            base_confidence += 0.1
        
        # Boost confidence for longer entities
        if len(ent.text) > 10:
            base_confidence += 0.1
        
        # Boost confidence if entity appears multiple times
        entity_count = sum(1 for e in doc.ents if e.text.lower() == ent.text.lower())
        if entity_count > 1:
            base_confidence += min(0.1, entity_count * 0.02)
        
        return min(1.0, base_confidence)
    
    def _calculate_regex_confidence(self, label: str, text: str) -> float:
        """Calculate confidence score for regex-extracted entity"""
        base_confidence = 0.6
        
        # Higher confidence for technical patterns
        technical_patterns = {"EMAIL", "URL", "PHONE_JP", "PHONE_EN", "DATE_JP", "MONEY_JP", "MONEY_EN"}
        if label in technical_patterns:
            base_confidence = 0.9
        
        # Boost confidence for longer text
        if len(text) > 15:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _get_entity_context(self, ent: Span, doc: Doc, window: int = 50) -> str:
        """Get context around entity"""
        start = max(0, ent.start_char - window)
        end = min(len(doc.text), ent.end_char + window)
        return doc.text[start:end].strip()
    
    def _get_regex_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around regex match"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def extract_entities(self, text: str, document_id: str, page_number: Optional[int] = None) -> List[Entity]:
        """Extract entities using spaCy, regex, and organizational knowledge methods"""
        all_entities = []
        
        # Extract with spaCy
        spacy_entities = self.extract_entities_spacy(text, document_id, page_number)
        all_entities.extend(spacy_entities)
        
        # Extract with regex
        regex_entities = self.extract_entities_regex(text, document_id, page_number)
        all_entities.extend(regex_entities)
        
        # Extract organizational entities
        org_entities = self.extract_organizational_entities(text, document_id, page_number)
        all_entities.extend(org_entities)
        
        # Deduplicate entities
        deduplicated_entities = self._deduplicate_entities(all_entities)
        
        logger.info(f"Extracted {len(deduplicated_entities)} entities from text", extra={
            "document_id": document_id,
            "page_number": page_number,
            "spacy_count": len(spacy_entities),
            "regex_count": len(regex_entities),
            "organizational_count": len(org_entities)
        })
        
        return deduplicated_entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on text and position"""
        seen = set()
        deduplicated = []
        
        # Sort by confidence (highest first)
        entities.sort(key=lambda x: x.confidence, reverse=True)
        
        for entity in entities:
            # Create key based on text and approximate position
            key = (entity.text.lower(), entity.start_char // 10)  # Group by 10-char windows
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated
    
    def link_entities(self, entities: List[Entity]) -> List[EntityLink]:
        """Link similar entities across documents"""
        links = []
        
        try:
            # Group entities by label
            entities_by_label = defaultdict(list)
            for entity in entities:
                entities_by_label[entity.label].append(entity)
            
            # Find links within each label group
            for label, label_entities in entities_by_label.items():
                if len(label_entities) < 2:
                    continue
                
                # Create text vectors for similarity comparison
                texts = [entity.text for entity in label_entities]
                
                try:
                    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    similarity_matrix = cosine_similarity(tfidf_matrix)
                    
                    # Find similar entities
                    for i in range(len(label_entities)):
                        for j in range(i + 1, len(label_entities)):
                            similarity = similarity_matrix[i][j]
                            
                            if similarity >= self.similarity_threshold:
                                link = EntityLink(
                                    entity_id=label_entities[i].id,
                                    linked_entity_id=label_entities[j].id,
                                    confidence=similarity,
                                    link_type="semantic_match",
                                    evidence={
                                        "similarity_score": similarity,
                                        "method": "tfidf_cosine",
                                        "text1": label_entities[i].text,
                                        "text2": label_entities[j].text
                                    }
                                )
                                links.append(link)
                
                except Exception as e:
                    logger.warning(f"Failed to compute similarity for label {label}", error=e)
                    
                # Also check for exact matches (case-insensitive)
                text_to_entities = defaultdict(list)
                for entity in label_entities:
                    text_to_entities[entity.text.lower()].append(entity)
                
                for text, matching_entities in text_to_entities.items():
                    if len(matching_entities) > 1:
                        # Link all matching entities
                        for i in range(len(matching_entities)):
                            for j in range(i + 1, len(matching_entities)):
                                link = EntityLink(
                                    entity_id=matching_entities[i].id,
                                    linked_entity_id=matching_entities[j].id,
                                    confidence=1.0,
                                    link_type="exact_match",
                                    evidence={
                                        "method": "exact_text_match",
                                        "text": text
                                    }
                                )
                                links.append(link)
        
        except Exception as e:
            logger.error("Failed to link entities", error=e)
        
        logger.info(f"Created {len(links)} entity links")
        return links
    
    def disambiguate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Disambiguate entities using context and properties"""
        disambiguated = []
        
        try:
            # Group entities by text (case-insensitive)
            entities_by_text = defaultdict(list)
            for entity in entities:
                entities_by_text[entity.text.lower()].append(entity)
            
            for text, entity_group in entities_by_text.items():
                if len(entity_group) == 1:
                    # No ambiguity
                    disambiguated.extend(entity_group)
                    continue
                
                # Resolve ambiguity
                resolved_entities = self._resolve_entity_ambiguity(entity_group)
                disambiguated.extend(resolved_entities)
        
        except Exception as e:
            logger.error("Failed to disambiguate entities", error=e)
            return entities
        
        return disambiguated
    
    def extract_organizational_entities(self, text: str, document_id: str, page_number: Optional[int] = None) -> List[Entity]:
        """Extract entities specific to organizational knowledge"""
        entities = []
        
        try:
            language = self.detect_language(text)
            
            # Extract skills, processes, and concepts
            org_patterns = self.custom_patterns.organizational_patterns
            
            for label, patterns in org_patterns.items():
                # Skip patterns not matching the language
                if language == "ja" and not label.endswith("_JP"):
                    continue
                if language == "en" and not label.endswith("_EN"):
                    continue
                
                for pattern in patterns:
                    try:
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        for match in matches:
                            entity_text = match.group().strip()
                            
                            # Skip very short matches
                            if len(entity_text) < 3:
                                continue
                            
                            # Calculate confidence based on pattern type and context
                            confidence = self._calculate_organizational_confidence(label, entity_text, text, match.start())
                            
                            entity = Entity(
                                id=str(uuid.uuid4()),
                                text=entity_text,
                                label=label.replace("_JP", "").replace("_EN", ""),  # Remove language suffix
                                start_char=match.start(),
                                end_char=match.end(),
                                confidence=confidence,
                                source_document_id=document_id,
                                source_page=page_number,
                                context=self._get_regex_context(text, match.start(), match.end()),
                                properties={
                                    "language": language,
                                    "extraction_method": "organizational_regex",
                                    "pattern": pattern,
                                    "entity_category": "organizational_knowledge"
                                }
                            )
                            
                            entities.append(entity)
                            
                    except Exception as e:
                        logger.warning(f"Failed to apply organizational pattern {pattern}", error=e)
                        
        except Exception as e:
            logger.error("Failed to extract organizational entities", error=e)
        
        return entities
    
    def _calculate_organizational_confidence(self, label: str, text: str, full_text: str, position: int) -> float:
        """Calculate confidence for organizational entities"""
        base_confidence = 0.7
        
        # Higher confidence for technical skills
        if label.startswith("SKILL") and any(tech in text.upper() for tech in ["PYTHON", "JAVA", "REACT", "AWS", "DOCKER"]):
            base_confidence = 0.9
        
        # Boost confidence if entity appears in a list or structured context
        context_window = full_text[max(0, position-100):position+100]
        if any(indicator in context_window.lower() for indicator in ["skills:", "technologies:", "experience:", "スキル:", "技術:"]):
            base_confidence += 0.1
        
        # Boost confidence for longer, more specific terms
        if len(text) > 20:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _resolve_entity_ambiguity(self, entities: List[Entity]) -> List[Entity]:
        """Resolve ambiguity between entities with same text"""
        if len(entities) <= 1:
            return entities
        
        # Group by label - if all have same label, they're likely the same entity
        labels = set(entity.label for entity in entities)
        if len(labels) == 1:
            # Same label - merge into highest confidence entity
            best_entity = max(entities, key=lambda x: x.confidence)
            
            # Merge properties and sources
            all_sources = set()
            all_properties = {}
            
            for entity in entities:
                all_sources.add(entity.source_document_id)
                if entity.properties:
                    all_properties.update(entity.properties)
            
            best_entity.properties = best_entity.properties or {}
            best_entity.properties.update(all_properties)
            best_entity.properties["merged_from_sources"] = list(all_sources)
            best_entity.properties["merged_count"] = len(entities)
            
            return [best_entity]
        
        # Different labels - keep all but boost confidence of most common label
        label_counts = Counter(entity.label for entity in entities)
        most_common_label = label_counts.most_common(1)[0][0]
        
        for entity in entities:
            if entity.label == most_common_label:
                entity.confidence = min(1.0, entity.confidence + 0.1)
        
        return entities
    
    def get_entity_statistics(self, entities: List[Entity]) -> Dict[str, Any]:
        """Get statistics about extracted entities"""
        stats = {
            "total_entities": len(entities),
            "entities_by_label": defaultdict(int),
            "entities_by_language": defaultdict(int),
            "entities_by_confidence": {"high": 0, "medium": 0, "low": 0},
            "entities_by_source": defaultdict(int),
            "average_confidence": 0.0,
            "organizational_entities": 0
        }
        
        if not entities:
            return dict(stats)
        
        total_confidence = 0
        for entity in entities:
            # Count by label
            stats["entities_by_label"][entity.label] += 1
            
            # Count by language
            if entity.properties and "language" in entity.properties:
                stats["entities_by_language"][entity.properties["language"]] += 1
            
            # Count by confidence level
            if entity.confidence >= 0.8:
                stats["entities_by_confidence"]["high"] += 1
            elif entity.confidence >= 0.6:
                stats["entities_by_confidence"]["medium"] += 1
            else:
                stats["entities_by_confidence"]["low"] += 1
            
            # Count by source document
            stats["entities_by_source"][entity.source_document_id] += 1
            
            # Count organizational entities
            if entity.properties and entity.properties.get("entity_category") == "organizational_knowledge":
                stats["organizational_entities"] += 1
            
            total_confidence += entity.confidence
        
        stats["average_confidence"] = total_confidence / len(entities)
        
        # Convert defaultdicts to regular dicts
        stats["entities_by_label"] = dict(stats["entities_by_label"])
        stats["entities_by_language"] = dict(stats["entities_by_language"])
        stats["entities_by_source"] = dict(stats["entities_by_source"])
        
        return stats
    
    def validate_entities(self, entities: List[Entity]) -> Dict[str, Any]:
        """Validate entity extraction quality and identify potential issues"""
        validation_results = {
            "total_entities": len(entities),
            "validation_passed": True,
            "issues": [],
            "warnings": [],
            "quality_score": 0.0
        }
        
        if not entities:
            validation_results["issues"].append("No entities extracted")
            validation_results["validation_passed"] = False
            return validation_results
        
        # Check for very low confidence entities
        low_confidence_entities = [e for e in entities if e.confidence < 0.5]
        if low_confidence_entities:
            validation_results["warnings"].append(f"{len(low_confidence_entities)} entities with very low confidence")
        
        # Check for duplicate entities (same text, same position)
        entity_signatures = set()
        duplicates = 0
        for entity in entities:
            signature = (entity.text.lower(), entity.start_char, entity.end_char)
            if signature in entity_signatures:
                duplicates += 1
            else:
                entity_signatures.add(signature)
        
        if duplicates > 0:
            validation_results["issues"].append(f"{duplicates} duplicate entities found")
        
        # Check for entities with missing required fields
        incomplete_entities = []
        for entity in entities:
            if not entity.text or not entity.label or entity.confidence is None:
                incomplete_entities.append(entity.id)
        
        if incomplete_entities:
            validation_results["issues"].append(f"{len(incomplete_entities)} entities with missing required fields")
        
        # Calculate quality score
        avg_confidence = sum(e.confidence for e in entities) / len(entities)
        quality_score = avg_confidence
        
        # Penalize for issues
        if duplicates > 0:
            quality_score -= 0.1
        if incomplete_entities:
            quality_score -= 0.2
        if len(low_confidence_entities) > len(entities) * 0.3:  # More than 30% low confidence
            quality_score -= 0.1
        
        validation_results["quality_score"] = max(0.0, quality_score)
        
        if validation_results["issues"]:
            validation_results["validation_passed"] = False
        
        return validation_results


# Global entity extractor instance
entity_extractor = EntityExtractor()