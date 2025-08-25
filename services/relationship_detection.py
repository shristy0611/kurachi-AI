"""
Relationship detection and mapping service for Kurachi AI
Implements relationship extraction between identified entities with confidence scoring,
temporal tracking, and conflict resolution
"""
import re
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from enum import Enum

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.entity_extraction import Entity, EntityExtractor
from config import config
from utils.logger import get_logger

logger = get_logger("relationship_detection")


class RelationshipType(Enum):
    """Types of relationships between entities"""
    # Organizational relationships
    WORKS_FOR = "works_for"
    MANAGES = "manages"
    REPORTS_TO = "reports_to"
    COLLABORATES_WITH = "collaborates_with"
    MEMBER_OF = "member_of"
    
    # Project relationships
    PARTICIPATES_IN = "participates_in"
    LEADS = "leads"
    CONTRIBUTES_TO = "contributes_to"
    DEPENDS_ON = "depends_on"
    
    # Knowledge relationships
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    CONTAINS = "contains"
    USES = "uses"
    IMPLEMENTS = "implements"
    
    # Temporal relationships
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CONCURRENT_WITH = "concurrent_with"
    
    # Semantic relationships
    SYNONYM_OF = "synonym_of"
    ANTONYM_OF = "antonym_of"
    HYPERNYM_OF = "hypernym_of"  # is-a relationship
    HYPONYM_OF = "hyponym_of"    # instance-of relationship
    
    # Document relationships
    MENTIONED_IN = "mentioned_in"
    DEFINED_IN = "defined_in"
    REFERENCED_BY = "referenced_by"


@dataclass
class RelationshipEvidence:
    """Evidence supporting a relationship"""
    document_id: str
    page_number: Optional[int]
    sentence: str
    context: str
    extraction_method: str
    confidence: float
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationshipEvidence':
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class EntityRelationship:
    """Relationship between two entities"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    confidence: float
    evidence: List[RelationshipEvidence]
    created_at: datetime
    updated_at: datetime
    version: int = 1
    is_active: bool = True
    temporal_info: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.temporal_info is None:
            self.temporal_info = {}
        if self.properties is None:
            self.properties = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['relationship_type'] = self.relationship_type.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['evidence'] = [ev.to_dict() for ev in self.evidence]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityRelationship':
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if 'relationship_type' in data and isinstance(data['relationship_type'], str):
            data['relationship_type'] = RelationshipType(data['relationship_type'])
        if 'evidence' in data:
            data['evidence'] = [RelationshipEvidence.from_dict(ev) for ev in data['evidence']]
        return cls(**data)


@dataclass
class RelationshipConflict:
    """Represents a conflict between relationships"""
    id: str
    conflicting_relationships: List[str]  # relationship IDs
    conflict_type: str
    description: str
    confidence: float
    resolution_status: str  # 'pending', 'resolved', 'ignored'
    created_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data


class RelationshipPatterns:
    """Patterns for detecting relationships in text"""
    
    def __init__(self):
        # English relationship patterns
        self.english_patterns = {
            RelationshipType.WORKS_FOR: [
                r"(\w+(?:\s+\w+)*)\s+works?\s+(?:for|at)\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+(?:is|are)\s+(?:employed|working)\s+(?:by|at)\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+(?:employee|staff|member)\s+(?:of|at)\s+(\w+(?:\s+\w+)*)"
            ],
            RelationshipType.MANAGES: [
                r"(\w+(?:\s+\w+)*)\s+manages?\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+(?:is|are)\s+(?:manager|supervisor|director)\s+of\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+leads?\s+(?:the\s+)?(\w+(?:\s+\w+)*)\s+(?:team|department|group)"
            ],
            RelationshipType.PARTICIPATES_IN: [
                r"(\w+(?:\s+\w+)*)\s+participates?\s+in\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+(?:is|are)\s+(?:involved|engaged)\s+in\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+(?:joins?|joined)\s+(\w+(?:\s+\w+)*)"
            ],
            RelationshipType.USES: [
                r"(\w+(?:\s+\w+)*)\s+uses?\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+(?:utilizes?|employs?)\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+(?:is|are)\s+based\s+on\s+(\w+(?:\s+\w+)*)"
            ],
            RelationshipType.RELATED_TO: [
                r"(\w+(?:\s+\w+)*)\s+(?:is|are)\s+related\s+to\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+(?:connects?|links?)\s+(?:to|with)\s+(\w+(?:\s+\w+)*)",
                r"(\w+(?:\s+\w+)*)\s+(?:and|&)\s+(\w+(?:\s+\w+)*)\s+(?:are|work)\s+(?:together|connected)"
            ]
        }
        
        # Japanese relationship patterns
        self.japanese_patterns = {
            RelationshipType.WORKS_FOR: [
                r"([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+(?:\s*[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)*)\s*(?:は|が)\s*([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+(?:\s*[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)*)\s*(?:で|に)\s*(?:働いて|勤めて|勤務して)",
                r"([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+(?:\s*[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)*)\s*(?:の|が)\s*([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+(?:\s*[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)*)\s*(?:社員|職員|メンバー)"
            ],
            RelationshipType.MANAGES: [
                r"([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+(?:\s*[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)*)\s*(?:は|が)\s*([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+(?:\s*[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)*)\s*(?:を|の)\s*(?:管理|マネージ|指導)",
                r"([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+(?:\s*[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)*)\s*(?:の|が)\s*([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+(?:\s*[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)*)\s*(?:部長|課長|マネージャー|リーダー)"
            ],
            RelationshipType.PARTICIPATES_IN: [
                r"([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+(?:\s*[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)*)\s*(?:は|が)\s*([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+(?:\s*[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)*)\s*(?:に|で)\s*(?:参加|参画|関与)",
                r"([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+(?:\s*[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)*)\s*(?:の|が)\s*([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+(?:\s*[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)*)\s*(?:メンバー|参加者)"
            ]
        }
        
        # Temporal patterns
        self.temporal_patterns = {
            "before": [r"before", r"prior to", r"earlier than", r"以前", r"前に"],
            "after": [r"after", r"following", r"later than", r"以降", r"後に"],
            "during": [r"during", r"while", r"throughout", r"間に", r"中に"],
            "concurrent": [r"at the same time", r"simultaneously", r"concurrently", r"同時に", r"並行して"]
        }


class RelationshipDetector:
    """Main relationship detection service"""
    
    def __init__(self):
        self.patterns = RelationshipPatterns()
        self.entity_extractor = EntityExtractor()
        self.relationships_cache = {}
        self.conflict_threshold = 0.3
        self.similarity_threshold = 0.7
        
    def detect_relationships(self, entities: List[Entity], text: str, document_id: str) -> List[EntityRelationship]:
        """Detect relationships between entities in text"""
        relationships = []
        
        try:
            # Pattern-based relationship detection
            pattern_relationships = self._detect_pattern_relationships(entities, text, document_id)
            relationships.extend(pattern_relationships)
            
            # Co-occurrence based relationship detection
            cooccurrence_relationships = self._detect_cooccurrence_relationships(entities, text, document_id)
            relationships.extend(cooccurrence_relationships)
            
            # Semantic similarity based relationships
            semantic_relationships = self._detect_semantic_relationships(entities, text, document_id)
            relationships.extend(semantic_relationships)
            
            # Temporal relationship detection
            temporal_relationships = self._detect_temporal_relationships(entities, text, document_id)
            relationships.extend(temporal_relationships)
            
            # Deduplicate and merge similar relationships
            deduplicated_relationships = self._deduplicate_relationships(relationships)
            
            logger.info(f"Detected {len(deduplicated_relationships)} relationships", extra={
                "document_id": document_id,
                "pattern_count": len(pattern_relationships),
                "cooccurrence_count": len(cooccurrence_relationships),
                "semantic_count": len(semantic_relationships),
                "temporal_count": len(temporal_relationships)
            })
            
            return deduplicated_relationships
            
        except Exception as e:
            logger.error("Failed to detect relationships", error=e, extra={"document_id": document_id})
            return []
    
    def _detect_pattern_relationships(self, entities: List[Entity], text: str, document_id: str) -> List[EntityRelationship]:
        """Detect relationships using predefined patterns"""
        relationships = []
        
        try:
            # Detect language
            language = self.entity_extractor.detect_language(text)
            patterns_dict = self.patterns.japanese_patterns if language == "ja" else self.patterns.english_patterns
            
            # Create entity lookup by text
            entity_lookup = {entity.text.lower(): entity for entity in entities}
            
            for relationship_type, patterns in patterns_dict.items():
                for pattern in patterns:
                    try:
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        for match in matches:
                            groups = match.groups()
                            if len(groups) >= 2:
                                source_text = groups[0].strip().lower()
                                target_text = groups[1].strip().lower()
                                
                                # Find matching entities
                                source_entity = self._find_matching_entity(source_text, entity_lookup)
                                target_entity = self._find_matching_entity(target_text, entity_lookup)
                                
                                if source_entity and target_entity and source_entity.id != target_entity.id:
                                    # Create evidence
                                    evidence = RelationshipEvidence(
                                        document_id=document_id,
                                        page_number=None,  # Could be extracted from entity if available
                                        sentence=match.group(),
                                        context=self._get_context(text, match.start(), match.end()),
                                        extraction_method="pattern_matching",
                                        confidence=0.8,
                                        created_at=datetime.utcnow()
                                    )
                                    
                                    # Create relationship
                                    relationship = EntityRelationship(
                                        id=str(uuid.uuid4()),
                                        source_entity_id=source_entity.id,
                                        target_entity_id=target_entity.id,
                                        relationship_type=relationship_type,
                                        confidence=0.8,
                                        evidence=[evidence],
                                        created_at=datetime.utcnow(),
                                        updated_at=datetime.utcnow(),
                                        properties={
                                            "extraction_method": "pattern_matching",
                                            "pattern": pattern,
                                            "language": language
                                        }
                                    )
                                    
                                    relationships.append(relationship)
                                    
                    except Exception as e:
                        logger.warning(f"Failed to apply pattern {pattern}", error=e)
                        
        except Exception as e:
            logger.error("Failed to detect pattern relationships", error=e)
        
        return relationships
    
    def _detect_cooccurrence_relationships(self, entities: List[Entity], text: str, document_id: str) -> List[EntityRelationship]:
        """Detect relationships based on entity co-occurrence"""
        relationships = []
        
        try:
            # Define co-occurrence window (characters)
            window_size = 200
            
            # Sort entities by position
            sorted_entities = sorted(entities, key=lambda x: x.start_char)
            
            for i, entity1 in enumerate(sorted_entities):
                for j, entity2 in enumerate(sorted_entities[i+1:], i+1):
                    # Check if entities are within window
                    distance = entity2.start_char - entity1.end_char
                    if distance <= window_size:
                        # Calculate confidence based on distance and entity types
                        confidence = self._calculate_cooccurrence_confidence(entity1, entity2, distance)
                        
                        if confidence >= 0.5:
                            # Extract context
                            context_start = max(0, entity1.start_char - 50)
                            context_end = min(len(text), entity2.end_char + 50)
                            context = text[context_start:context_end]
                            
                            # Create evidence
                            evidence = RelationshipEvidence(
                                document_id=document_id,
                                page_number=entity1.source_page,
                                sentence=context,
                                context=context,
                                extraction_method="cooccurrence",
                                confidence=confidence,
                                created_at=datetime.utcnow()
                            )
                            
                            # Determine relationship type based on entity types
                            relationship_type = self._infer_relationship_type(entity1, entity2)
                            
                            # Create relationship
                            relationship = EntityRelationship(
                                id=str(uuid.uuid4()),
                                source_entity_id=entity1.id,
                                target_entity_id=entity2.id,
                                relationship_type=relationship_type,
                                confidence=confidence,
                                evidence=[evidence],
                                created_at=datetime.utcnow(),
                                updated_at=datetime.utcnow(),
                                properties={
                                    "extraction_method": "cooccurrence",
                                    "distance": distance,
                                    "window_size": window_size
                                }
                            )
                            
                            relationships.append(relationship)
                    else:
                        # Entities too far apart, break inner loop
                        break
                        
        except Exception as e:
            logger.error("Failed to detect co-occurrence relationships", error=e)
        
        return relationships
    
    def _detect_semantic_relationships(self, entities: List[Entity], text: str, document_id: str) -> List[EntityRelationship]:
        """Detect relationships based on semantic similarity"""
        relationships = []
        
        try:
            if len(entities) < 2:
                return relationships
            
            # Group entities by type for more accurate similarity
            entities_by_type = defaultdict(list)
            for entity in entities:
                entities_by_type[entity.label].append(entity)
            
            # Find semantic relationships within each type
            for entity_type, type_entities in entities_by_type.items():
                if len(type_entities) < 2:
                    continue
                
                # Create text vectors
                entity_texts = [entity.text for entity in type_entities]
                
                try:
                    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
                    tfidf_matrix = vectorizer.fit_transform(entity_texts)
                    similarity_matrix = cosine_similarity(tfidf_matrix)
                    
                    # Find similar entities
                    for i in range(len(type_entities)):
                        for j in range(i + 1, len(type_entities)):
                            similarity = similarity_matrix[i][j]
                            
                            if similarity >= self.similarity_threshold:
                                # Create evidence
                                evidence = RelationshipEvidence(
                                    document_id=document_id,
                                    page_number=None,
                                    sentence=f"Semantic similarity between '{type_entities[i].text}' and '{type_entities[j].text}'",
                                    context="",
                                    extraction_method="semantic_similarity",
                                    confidence=similarity,
                                    created_at=datetime.utcnow()
                                )
                                
                                # Create relationship
                                relationship = EntityRelationship(
                                    id=str(uuid.uuid4()),
                                    source_entity_id=type_entities[i].id,
                                    target_entity_id=type_entities[j].id,
                                    relationship_type=RelationshipType.SIMILAR_TO,
                                    confidence=similarity,
                                    evidence=[evidence],
                                    created_at=datetime.utcnow(),
                                    updated_at=datetime.utcnow(),
                                    properties={
                                        "extraction_method": "semantic_similarity",
                                        "similarity_score": similarity,
                                        "entity_type": entity_type
                                    }
                                )
                                
                                relationships.append(relationship)
                                
                except Exception as e:
                    logger.warning(f"Failed to compute semantic similarity for type {entity_type}", error=e)
                    
        except Exception as e:
            logger.error("Failed to detect semantic relationships", error=e)
        
        return relationships
    
    def _detect_temporal_relationships(self, entities: List[Entity], text: str, document_id: str) -> List[EntityRelationship]:
        """Detect temporal relationships between entities"""
        relationships = []
        
        try:
            # Find date/time entities
            temporal_entities = [e for e in entities if e.label in ["DATE", "TIME", "DATE_JP"]]
            other_entities = [e for e in entities if e.label not in ["DATE", "TIME", "DATE_JP"]]
            
            if not temporal_entities:
                return relationships
            
            # Look for temporal patterns around entities
            for temporal_entity in temporal_entities:
                for other_entity in other_entities:
                    # Check if entities are close to each other
                    distance = abs(temporal_entity.start_char - other_entity.start_char)
                    if distance <= 300:  # Within 300 characters
                        
                        # Extract context around both entities
                        context_start = min(temporal_entity.start_char, other_entity.start_char) - 100
                        context_end = max(temporal_entity.end_char, other_entity.end_char) + 100
                        context_start = max(0, context_start)
                        context_end = min(len(text), context_end)
                        context = text[context_start:context_end]
                        
                        # Look for temporal indicators
                        temporal_type = self._identify_temporal_relationship(context)
                        
                        if temporal_type:
                            confidence = 0.7
                            
                            # Create evidence
                            evidence = RelationshipEvidence(
                                document_id=document_id,
                                page_number=temporal_entity.source_page,
                                sentence=context,
                                context=context,
                                extraction_method="temporal_analysis",
                                confidence=confidence,
                                created_at=datetime.utcnow()
                            )
                            
                            # Create relationship
                            relationship = EntityRelationship(
                                id=str(uuid.uuid4()),
                                source_entity_id=other_entity.id,
                                target_entity_id=temporal_entity.id,
                                relationship_type=RelationshipType.RELATED_TO,
                                confidence=confidence,
                                evidence=[evidence],
                                created_at=datetime.utcnow(),
                                updated_at=datetime.utcnow(),
                                temporal_info={
                                    "temporal_type": temporal_type,
                                    "temporal_entity": temporal_entity.text,
                                    "temporal_context": context
                                },
                                properties={
                                    "extraction_method": "temporal_analysis",
                                    "temporal_relationship": temporal_type
                                }
                            )
                            
                            relationships.append(relationship)
                            
        except Exception as e:
            logger.error("Failed to detect temporal relationships", error=e)
        
        return relationships
    
    def _find_matching_entity(self, text: str, entity_lookup: Dict[str, Entity]) -> Optional[Entity]:
        """Find entity that matches the given text"""
        # Exact match
        if text in entity_lookup:
            return entity_lookup[text]
        
        # Fuzzy match - check if text is contained in any entity
        for entity_text, entity in entity_lookup.items():
            if text in entity_text or entity_text in text:
                return entity
        
        return None
    
    def _get_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Get context around a text span"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _calculate_cooccurrence_confidence(self, entity1: Entity, entity2: Entity, distance: int) -> float:
        """Calculate confidence for co-occurrence relationship"""
        base_confidence = 0.5
        
        # Closer entities have higher confidence
        distance_factor = max(0, 1 - (distance / 200))
        base_confidence += distance_factor * 0.3
        
        # Same entity types have higher confidence
        if entity1.label == entity2.label:
            base_confidence += 0.1
        
        # High confidence entities boost relationship confidence
        entity_confidence_factor = (entity1.confidence + entity2.confidence) / 2
        base_confidence += entity_confidence_factor * 0.2
        
        return min(1.0, base_confidence)
    
    def _infer_relationship_type(self, entity1: Entity, entity2: Entity) -> RelationshipType:
        """Infer relationship type based on entity types"""
        # Person-Organization relationships
        if entity1.label == "PERSON" and entity2.label in ["ORG", "COMPANY_JP", "COMPANY_EN"]:
            return RelationshipType.WORKS_FOR
        elif entity1.label in ["ORG", "COMPANY_JP", "COMPANY_EN"] and entity2.label == "PERSON":
            return RelationshipType.WORKS_FOR
        
        # Person-Project relationships
        elif entity1.label == "PERSON" and entity2.label in ["PROJECT_JP", "PROJECT_EN"]:
            return RelationshipType.PARTICIPATES_IN
        elif entity1.label in ["PROJECT_JP", "PROJECT_EN"] and entity2.label == "PERSON":
            return RelationshipType.PARTICIPATES_IN
        
        # Skill-Person relationships
        elif entity1.label in ["SKILL_JP", "SKILL_EN"] and entity2.label == "PERSON":
            return RelationshipType.USES
        elif entity1.label == "PERSON" and entity2.label in ["SKILL_JP", "SKILL_EN"]:
            return RelationshipType.USES
        
        # Same type entities
        elif entity1.label == entity2.label:
            return RelationshipType.SIMILAR_TO
        
        # Default relationship
        else:
            return RelationshipType.RELATED_TO
    
    def _identify_temporal_relationship(self, context: str) -> Optional[str]:
        """Identify temporal relationship type from context"""
        context_lower = context.lower()
        
        for temporal_type, patterns in self.patterns.temporal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context_lower):
                    return temporal_type
        
        return None
    
    def _deduplicate_relationships(self, relationships: List[EntityRelationship]) -> List[EntityRelationship]:
        """Remove duplicate relationships and merge similar ones"""
        # Group relationships by entity pair and type
        relationship_groups = defaultdict(list)
        
        for rel in relationships:
            # Create key for grouping (normalize entity order for bidirectional relationships)
            entity_pair = tuple(sorted([rel.source_entity_id, rel.target_entity_id]))
            key = (entity_pair, rel.relationship_type)
            relationship_groups[key].append(rel)
        
        deduplicated = []
        
        for key, group in relationship_groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge relationships with same entity pair and type
                merged_relationship = self._merge_relationships(group)
                deduplicated.append(merged_relationship)
        
        return deduplicated
    
    def _merge_relationships(self, relationships: List[EntityRelationship]) -> EntityRelationship:
        """Merge multiple relationships into one"""
        # Use the relationship with highest confidence as base
        base_relationship = max(relationships, key=lambda x: x.confidence)
        
        # Merge evidence from all relationships
        all_evidence = []
        for rel in relationships:
            all_evidence.extend(rel.evidence)
        
        # Calculate merged confidence (weighted average)
        total_confidence = sum(rel.confidence for rel in relationships)
        merged_confidence = min(1.0, total_confidence / len(relationships) + 0.1)  # Boost for multiple sources
        
        # Merge properties
        merged_properties = base_relationship.properties.copy()
        merged_properties["merged_from_count"] = len(relationships)
        merged_properties["extraction_methods"] = list(set(
            rel.properties.get("extraction_method", "unknown") for rel in relationships
        ))
        
        # Create merged relationship
        merged_relationship = EntityRelationship(
            id=base_relationship.id,
            source_entity_id=base_relationship.source_entity_id,
            target_entity_id=base_relationship.target_entity_id,
            relationship_type=base_relationship.relationship_type,
            confidence=merged_confidence,
            evidence=all_evidence,
            created_at=base_relationship.created_at,
            updated_at=datetime.utcnow(),
            version=base_relationship.version + 1,
            temporal_info=base_relationship.temporal_info,
            properties=merged_properties
        )
        
        return merged_relationship
    
    def detect_relationship_conflicts(self, relationships: List[EntityRelationship]) -> List[RelationshipConflict]:
        """Detect conflicts between relationships"""
        conflicts = []
        
        try:
            # Group relationships by entity pairs
            entity_pair_relationships = defaultdict(list)
            
            for rel in relationships:
                entity_pair = tuple(sorted([rel.source_entity_id, rel.target_entity_id]))
                entity_pair_relationships[entity_pair].append(rel)
            
            # Check for conflicting relationships
            for entity_pair, pair_relationships in entity_pair_relationships.items():
                if len(pair_relationships) > 1:
                    # Check for contradictory relationship types
                    relationship_types = [rel.relationship_type for rel in pair_relationships]
                    
                    # Define conflicting relationship types
                    conflicting_pairs = [
                        (RelationshipType.MANAGES, RelationshipType.REPORTS_TO),
                        (RelationshipType.PRECEDES, RelationshipType.FOLLOWS),
                        (RelationshipType.SYNONYM_OF, RelationshipType.ANTONYM_OF)
                    ]
                    
                    for type1, type2 in conflicting_pairs:
                        if type1 in relationship_types and type2 in relationship_types:
                            conflicting_rels = [rel for rel in pair_relationships 
                                              if rel.relationship_type in [type1, type2]]
                            
                            conflict = RelationshipConflict(
                                id=str(uuid.uuid4()),
                                conflicting_relationships=[rel.id for rel in conflicting_rels],
                                conflict_type="contradictory_relationship_types",
                                description=f"Conflicting relationship types: {type1.value} vs {type2.value}",
                                confidence=0.8,
                                resolution_status="pending",
                                created_at=datetime.utcnow()
                            )
                            
                            conflicts.append(conflict)
                    
                    # Check for low confidence conflicts
                    low_confidence_rels = [rel for rel in pair_relationships if rel.confidence < self.conflict_threshold]
                    if len(low_confidence_rels) > 1:
                        conflict = RelationshipConflict(
                            id=str(uuid.uuid4()),
                            conflicting_relationships=[rel.id for rel in low_confidence_rels],
                            conflict_type="low_confidence_relationships",
                            description="Multiple low-confidence relationships between same entities",
                            confidence=0.6,
                            resolution_status="pending",
                            created_at=datetime.utcnow()
                        )
                        
                        conflicts.append(conflict)
            
            logger.info(f"Detected {len(conflicts)} relationship conflicts")
            
        except Exception as e:
            logger.error("Failed to detect relationship conflicts", error=e)
        
        return conflicts
    
    def resolve_relationship_conflicts(self, conflicts: List[RelationshipConflict], 
                                     relationships: List[EntityRelationship]) -> List[EntityRelationship]:
        """Resolve relationship conflicts"""
        resolved_relationships = relationships.copy()
        relationship_lookup = {rel.id: rel for rel in relationships}
        
        try:
            for conflict in conflicts:
                if conflict.resolution_status != "pending":
                    continue
                
                conflicting_rels = [relationship_lookup[rel_id] for rel_id in conflict.conflicting_relationships 
                                  if rel_id in relationship_lookup]
                
                if not conflicting_rels:
                    continue
                
                if conflict.conflict_type == "contradictory_relationship_types":
                    # Keep the relationship with highest confidence
                    best_rel = max(conflicting_rels, key=lambda x: x.confidence)
                    
                    # Remove other conflicting relationships
                    for rel in conflicting_rels:
                        if rel.id != best_rel.id:
                            resolved_relationships = [r for r in resolved_relationships if r.id != rel.id]
                    
                    # Mark conflict as resolved
                    conflict.resolution_status = "resolved"
                    conflict.resolved_at = datetime.utcnow()
                    conflict.resolution_method = "keep_highest_confidence"
                
                elif conflict.conflict_type == "low_confidence_relationships":
                    # Merge low confidence relationships if they have same type
                    same_type_groups = defaultdict(list)
                    for rel in conflicting_rels:
                        same_type_groups[rel.relationship_type].append(rel)
                    
                    for rel_type, type_rels in same_type_groups.items():
                        if len(type_rels) > 1:
                            # Merge relationships of same type
                            merged_rel = self._merge_relationships(type_rels)
                            
                            # Remove original relationships
                            for rel in type_rels:
                                resolved_relationships = [r for r in resolved_relationships if r.id != rel.id]
                            
                            # Add merged relationship
                            resolved_relationships.append(merged_rel)
                    
                    # Mark conflict as resolved
                    conflict.resolution_status = "resolved"
                    conflict.resolved_at = datetime.utcnow()
                    conflict.resolution_method = "merge_same_type"
            
            logger.info(f"Resolved {len([c for c in conflicts if c.resolution_status == 'resolved'])} conflicts")
            
        except Exception as e:
            logger.error("Failed to resolve relationship conflicts", error=e)
        
        return resolved_relationships
    
    def update_relationship_versions(self, existing_relationships: List[EntityRelationship], 
                                   new_relationships: List[EntityRelationship]) -> List[EntityRelationship]:
        """Update relationship versions and track changes"""
        updated_relationships = []
        existing_lookup = {}
        
        # Create lookup for existing relationships
        for rel in existing_relationships:
            entity_pair = tuple(sorted([rel.source_entity_id, rel.target_entity_id]))
            key = (entity_pair, rel.relationship_type)
            existing_lookup[key] = rel
        
        # Process new relationships
        for new_rel in new_relationships:
            entity_pair = tuple(sorted([new_rel.source_entity_id, new_rel.target_entity_id]))
            key = (entity_pair, new_rel.relationship_type)
            
            if key in existing_lookup:
                # Update existing relationship
                existing_rel = existing_lookup[key]
                
                # Check if relationship has changed significantly
                confidence_change = abs(new_rel.confidence - existing_rel.confidence)
                evidence_change = len(new_rel.evidence) != len(existing_rel.evidence)
                
                if confidence_change > 0.1 or evidence_change:
                    # Create new version
                    updated_rel = EntityRelationship(
                        id=existing_rel.id,
                        source_entity_id=existing_rel.source_entity_id,
                        target_entity_id=existing_rel.target_entity_id,
                        relationship_type=existing_rel.relationship_type,
                        confidence=new_rel.confidence,
                        evidence=existing_rel.evidence + new_rel.evidence,
                        created_at=existing_rel.created_at,
                        updated_at=datetime.utcnow(),
                        version=existing_rel.version + 1,
                        temporal_info=new_rel.temporal_info or existing_rel.temporal_info,
                        properties={
                            **existing_rel.properties,
                            **new_rel.properties,
                            "version_history": existing_rel.properties.get("version_history", []) + [
                                {
                                    "version": existing_rel.version,
                                    "confidence": existing_rel.confidence,
                                    "updated_at": existing_rel.updated_at.isoformat()
                                }
                            ]
                        }
                    )
                    
                    updated_relationships.append(updated_rel)
                else:
                    # No significant change, keep existing
                    updated_relationships.append(existing_rel)
                
                # Remove from lookup to track processed relationships
                del existing_lookup[key]
            else:
                # New relationship
                updated_relationships.append(new_rel)
        
        # Add remaining existing relationships that weren't updated
        updated_relationships.extend(existing_lookup.values())
        
        return updated_relationships
    
    def get_relationship_statistics(self, relationships: List[EntityRelationship]) -> Dict[str, Any]:
        """Get statistics about relationships"""
        stats = {
            "total_relationships": len(relationships),
            "relationships_by_type": defaultdict(int),
            "relationships_by_confidence": {"high": 0, "medium": 0, "low": 0},
            "relationships_by_method": defaultdict(int),
            "average_confidence": 0.0,
            "temporal_relationships": 0,
            "versioned_relationships": 0
        }
        
        if not relationships:
            return dict(stats)
        
        total_confidence = 0
        for rel in relationships:
            # Count by type
            stats["relationships_by_type"][rel.relationship_type.value] += 1
            
            # Count by confidence level
            if rel.confidence >= 0.8:
                stats["relationships_by_confidence"]["high"] += 1
            elif rel.confidence >= 0.6:
                stats["relationships_by_confidence"]["medium"] += 1
            else:
                stats["relationships_by_confidence"]["low"] += 1
            
            # Count by extraction method
            if rel.properties and "extraction_method" in rel.properties:
                stats["relationships_by_method"][rel.properties["extraction_method"]] += 1
            
            # Count temporal relationships
            if rel.temporal_info:
                stats["temporal_relationships"] += 1
            
            # Count versioned relationships
            if rel.version > 1:
                stats["versioned_relationships"] += 1
            
            total_confidence += rel.confidence
        
        stats["average_confidence"] = total_confidence / len(relationships)
        
        return dict(stats)


class RelationshipManager:
    """Manager for relationship operations and storage"""
    
    def __init__(self):
        self.detector = RelationshipDetector()
        self.relationships_storage = {}  # In-memory storage for now
        self.conflicts_storage = {}
        
    def process_document_relationships(self, entities: List[Entity], text: str, document_id: str) -> Dict[str, Any]:
        """Process relationships for a document"""
        try:
            # Detect relationships
            new_relationships = self.detector.detect_relationships(entities, text, document_id)
            
            # Get existing relationships for the document
            existing_relationships = self.get_document_relationships(document_id)
            
            # Update relationship versions
            updated_relationships = self.detector.update_relationship_versions(existing_relationships, new_relationships)
            
            # Detect conflicts
            conflicts = self.detector.detect_relationship_conflicts(updated_relationships)
            
            # Resolve conflicts
            resolved_relationships = self.detector.resolve_relationship_conflicts(conflicts, updated_relationships)
            
            # Store relationships
            self.store_relationships(document_id, resolved_relationships)
            
            # Store conflicts
            self.store_conflicts(document_id, conflicts)
            
            # Get statistics
            stats = self.detector.get_relationship_statistics(resolved_relationships)
            
            result = {
                "relationships": resolved_relationships,
                "conflicts": conflicts,
                "statistics": stats,
                "document_id": document_id,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            logger.info("Processed document relationships", extra={
                "document_id": document_id,
                "relationship_count": len(resolved_relationships),
                "conflict_count": len(conflicts)
            })
            
            return result
            
        except Exception as e:
            logger.error("Failed to process document relationships", error=e, extra={"document_id": document_id})
            return {
                "relationships": [],
                "conflicts": [],
                "statistics": {},
                "document_id": document_id,
                "error": str(e)
            }
    
    def store_relationships(self, document_id: str, relationships: List[EntityRelationship]):
        """Store relationships (in-memory for now, should be database in production)"""
        self.relationships_storage[document_id] = relationships
    
    def store_conflicts(self, document_id: str, conflicts: List[RelationshipConflict]):
        """Store conflicts (in-memory for now, should be database in production)"""
        self.conflicts_storage[document_id] = conflicts
    
    def get_document_relationships(self, document_id: str) -> List[EntityRelationship]:
        """Get relationships for a document"""
        return self.relationships_storage.get(document_id, [])
    
    def get_document_conflicts(self, document_id: str) -> List[RelationshipConflict]:
        """Get conflicts for a document"""
        return self.conflicts_storage.get(document_id, [])
    
    def get_all_relationships(self) -> List[EntityRelationship]:
        """Get all relationships across all documents"""
        all_relationships = []
        for relationships in self.relationships_storage.values():
            all_relationships.extend(relationships)
        return all_relationships
    
    def get_entity_relationships(self, entity_id: str) -> List[EntityRelationship]:
        """Get all relationships involving a specific entity"""
        entity_relationships = []
        for relationships in self.relationships_storage.values():
            for rel in relationships:
                if rel.source_entity_id == entity_id or rel.target_entity_id == entity_id:
                    entity_relationships.append(rel)
        return entity_relationships
    
    def get_relationship_by_id(self, relationship_id: str) -> Optional[EntityRelationship]:
        """Get a specific relationship by ID"""
        for relationships in self.relationships_storage.values():
            for rel in relationships:
                if rel.id == relationship_id:
                    return rel
        return None


# Global relationship manager instance
relationship_manager = RelationshipManager()