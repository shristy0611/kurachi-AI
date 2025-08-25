"""
Knowledge Graph Construction Service for Kurachi AI
Builds and manages knowledge graphs from extracted entities and relationships
"""
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import networkx as nx
import numpy as np

from config import config
from models.database import db_manager, Document
from services.entity_extraction import entity_extractor, Entity, EntityLink
from services.language_detection import language_detection_service
from services.neo4j_service import neo4j_service
from utils.logger import get_logger

logger = get_logger("knowledge_graph")


@dataclass
class Relationship:
    """Relationship between entities in the knowledge graph"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float
    source_document_id: str
    evidence: Dict[str, Any]
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
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Create Relationship from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph representing an entity"""
    entity: Entity
    relationships: List[str]  # List of relationship IDs
    centrality_score: float = 0.0
    importance_score: float = 0.0
    cluster_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "entity": self.entity.to_dict(),
            "relationships": self.relationships,
            "centrality_score": self.centrality_score,
            "importance_score": self.importance_score,
            "cluster_id": self.cluster_id
        }


@dataclass
class KnowledgeCluster:
    """Cluster of related entities and concepts"""
    id: str
    name: str
    entities: List[str]  # Entity IDs
    central_concepts: List[str]
    cluster_type: str  # "topic", "organization", "project", etc.
    confidence: float
    properties: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


class RelationshipExtractor:
    """Extracts relationships between entities"""
    
    def __init__(self):
        # Relationship patterns for different entity types
        self.relationship_patterns = {
            # Co-occurrence patterns
            "WORKS_AT": {
                "pattern": r"({person})\s+(?:works?\s+at|is\s+employed\s+at|is\s+with)\s+({organization})",
                "confidence": 0.8
            },
            "MEMBER_OF": {
                "pattern": r"({person})\s+(?:is\s+a\s+member\s+of|belongs\s+to|is\s+part\s+of)\s+({organization})",
                "confidence": 0.8
            },
            "LOCATED_IN": {
                "pattern": r"({organization})\s+(?:is\s+located\s+in|is\s+based\s+in|is\s+in)\s+({location})",
                "confidence": 0.7
            },
            "MANAGES": {
                "pattern": r"({person})\s+(?:manages|leads|heads|supervises)\s+({organization}|{project})",
                "confidence": 0.8
            },
            "PARTICIPATES_IN": {
                "pattern": r"({person}|{organization})\s+(?:participates\s+in|is\s+involved\s+in|takes\s+part\s+in)\s+({project})",
                "confidence": 0.7
            }
        }
        
        # Japanese relationship patterns
        self.japanese_patterns = {
            "WORKS_AT": {
                "pattern": r"({person})(?:は|が)({organization})(?:に|で)(?:勤務|働|勤め)",
                "confidence": 0.8
            },
            "MEMBER_OF": {
                "pattern": r"({person})(?:は|が)({organization})(?:の|に)(?:所属|メンバー|一員)",
                "confidence": 0.8
            },
            "MANAGES": {
                "pattern": r"({person})(?:は|が)({organization}|{project})(?:を|の)(?:管理|運営|指揮|担当)",
                "confidence": 0.8
            }
        }
    
    def extract_relationships(self, entities: List[Entity], document_text: str, 
                            document_id: str) -> List[Relationship]:
        """Extract relationships between entities in a document"""
        relationships = []
        
        try:
            # Extract proximity-based relationships
            proximity_rels = self._extract_proximity_relationships(entities, document_text, document_id)
            relationships.extend(proximity_rels)
            
            # Extract pattern-based relationships
            pattern_rels = self._extract_pattern_relationships(entities, document_text, document_id)
            relationships.extend(pattern_rels)
            
            # Extract co-occurrence relationships
            cooccur_rels = self._extract_cooccurrence_relationships(entities, document_text, document_id)
            relationships.extend(cooccur_rels)
            
            logger.info(f"Extracted {len(relationships)} relationships from document {document_id}")
            
        except Exception as e:
            logger.error("Failed to extract relationships", error=e)
        
        return relationships
    
    def _extract_proximity_relationships(self, entities: List[Entity], text: str, 
                                       document_id: str) -> List[Relationship]:
        """Extract relationships based on entity proximity"""
        relationships = []
        proximity_threshold = 100  # characters
        
        try:
            # Sort entities by position
            sorted_entities = sorted(entities, key=lambda x: x.start_char)
            
            for i, entity1 in enumerate(sorted_entities):
                for j in range(i + 1, len(sorted_entities)):
                    entity2 = sorted_entities[j]
                    
                    # Check if entities are close enough
                    distance = entity2.start_char - entity1.end_char
                    if distance > proximity_threshold:
                        break  # Entities are sorted, so no point checking further
                    
                    # Create proximity relationship
                    relationship_type = self._determine_proximity_relationship_type(entity1, entity2, text)
                    if relationship_type:
                        confidence = max(0.3, 1.0 - (distance / proximity_threshold))
                        
                        relationship = Relationship(
                            id=str(uuid.uuid4()),
                            source_entity_id=entity1.id,
                            target_entity_id=entity2.id,
                            relationship_type=relationship_type,
                            confidence=confidence,
                            source_document_id=document_id,
                            evidence={
                                "method": "proximity",
                                "distance": distance,
                                "context": text[entity1.start_char:entity2.end_char]
                            }
                        )
                        relationships.append(relationship)
                        
        except Exception as e:
            logger.error("Failed to extract proximity relationships", error=e)
        
        return relationships
    
    def _determine_proximity_relationship_type(self, entity1: Entity, entity2: Entity, text: str) -> Optional[str]:
        """Determine relationship type based on entity types and context"""
        # Entity type mappings
        person_types = {"PERSON", "PERSON_JP", "ROLE_EN"}
        org_types = {"ORG", "COMPANY_JP", "COMPANY_EN", "DEPARTMENT_JP", "DEPARTMENT_EN"}
        project_types = {"PROJECT_JP", "PROJECT_EN"}
        location_types = {"GPE", "LOC"}
        
        e1_type = self._categorize_entity_type(entity1.label, person_types, org_types, project_types, location_types)
        e2_type = self._categorize_entity_type(entity2.label, person_types, org_types, project_types, location_types)
        
        # Determine relationship based on entity type combinations
        if e1_type == "person" and e2_type == "organization":
            return "ASSOCIATED_WITH"
        elif e1_type == "person" and e2_type == "project":
            return "PARTICIPATES_IN"
        elif e1_type == "organization" and e2_type == "location":
            return "LOCATED_IN"
        elif e1_type == "organization" and e2_type == "project":
            return "INVOLVED_IN"
        elif e1_type == e2_type:
            return "RELATED_TO"
        
        return "MENTIONED_WITH"
    
    def _categorize_entity_type(self, label: str, person_types: Set[str], org_types: Set[str], 
                               project_types: Set[str], location_types: Set[str]) -> str:
        """Categorize entity label into general types"""
        if label in person_types:
            return "person"
        elif label in org_types:
            return "organization"
        elif label in project_types:
            return "project"
        elif label in location_types:
            return "location"
        else:
            return "other"
    
    def _extract_pattern_relationships(self, entities: List[Entity], text: str, 
                                     document_id: str) -> List[Relationship]:
        """Extract relationships using predefined patterns"""
        relationships = []
        
        # Detect language using SOTA detector
        lang_code = language_detection_service.detect_language(text)
        patterns = self.japanese_patterns if lang_code == 'ja' else self.relationship_patterns
        
        # Implementation would use regex patterns to find relationships
        # This is a simplified version for demonstration
        
        return relationships
    
    def _extract_cooccurrence_relationships(self, entities: List[Entity], text: str, 
                                          document_id: str) -> List[Relationship]:
        """Extract relationships based on entity co-occurrence in sentences"""
        relationships = []
        
        try:
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                # Find entities in this sentence
                sentence_entities = []
                for entity in entities:
                    if entity.start_char >= sentence['start'] and entity.end_char <= sentence['end']:
                        sentence_entities.append(entity)
                
                # Create co-occurrence relationships
                if len(sentence_entities) >= 2:
                    for i, entity1 in enumerate(sentence_entities):
                        for entity2 in sentence_entities[i + 1:]:
                            relationship = Relationship(
                                id=str(uuid.uuid4()),
                                source_entity_id=entity1.id,
                                target_entity_id=entity2.id,
                                relationship_type="CO_OCCURS",
                                confidence=0.6,
                                source_document_id=document_id,
                                evidence={
                                    "method": "sentence_cooccurrence",
                                    "sentence": sentence['text']
                                }
                            )
                            relationships.append(relationship)
                            
        except Exception as e:
            logger.error("Failed to extract co-occurrence relationships", error=e)
        
        return relationships
    
    def _split_into_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sentences with position information"""
        sentences = []
        
        # Simple sentence splitting - could be enhanced with proper NLP
        import re
        sentence_pattern = r'[.!?。！？]+\s*'
        
        start = 0
        for match in re.finditer(sentence_pattern, text):
            end = match.end()
            sentence_text = text[start:end].strip()
            if sentence_text:
                sentences.append({
                    'text': sentence_text,
                    'start': start,
                    'end': end
                })
            start = end
        
        # Add remaining text as last sentence
        if start < len(text):
            remaining = text[start:].strip()
            if remaining:
                sentences.append({
                    'text': remaining,
                    'start': start,
                    'end': len(text)
                })
        
        return sentences


class KnowledgeGraphBuilder:
    """Main knowledge graph construction service"""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.clusters: Dict[str, KnowledgeCluster] = {}
        self.graph = nx.DiGraph()
        self.relationship_extractor = RelationshipExtractor()
        logger.info("Knowledge graph builder initialized")
    
    def process_document(self, document_id: str) -> Dict[str, Any]:
        """Process a document to extract entities and build knowledge graph"""
        try:
            # Get document from database
            document = db_manager.get_document(document_id)
            if not document:
                logger.error(f"Document not found: {document_id}")
                return {"success": False, "error": "Document not found"}
            
            # Read document content (simplified - would need to get processed text)
            # For now, we'll work with a placeholder
            document_text = f"Sample text for document {document.original_filename}"
            
            logger.info(f"Processing document for knowledge graph: {document.original_filename}")
            
            # Extract entities
            entities = entity_extractor.extract_entities(document_text, document_id)
            
            # Extract relationships
            relationships = self.relationship_extractor.extract_relationships(
                entities, document_text, document_id
            )
            
            # Add to knowledge graph
            self._add_entities_to_graph(entities)
            self._add_relationships_to_graph(relationships)
            
            # Update graph analytics
            self._update_graph_analytics()
            
            result = {
                "success": True,
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "graph_stats": self._get_graph_statistics()
            }
            
            logger.info(f"Document processed successfully for knowledge graph", extra=result)
            return result
            
        except Exception as e:
            logger.error("Failed to process document for knowledge graph", error=e)
            return {"success": False, "error": str(e)}
    
    def _add_entities_to_graph(self, entities: List[Entity]):
        """Add entities to the knowledge graph"""
        for entity in entities:
            if entity.id not in self.entities:
                self.entities[entity.id] = entity
                self.nodes[entity.id] = KnowledgeNode(
                    entity=entity,
                    relationships=[]
                )
                
                # Add to NetworkX graph (for in-memory operations)
                self.graph.add_node(entity.id, 
                                  label=entity.label,
                                  text=entity.text,
                                  confidence=entity.confidence)
                
                # Add to Neo4j (for persistent storage)
                if neo4j_service.is_connected():
                    success = neo4j_service.create_entity_node(entity, entity.source_document_id)
                    if not success:
                        logger.warning(f"Failed to persist entity to Neo4j: {entity.id}")
                else:
                    logger.warning("Neo4j not connected, entity not persisted")
    
    def _add_relationships_to_graph(self, relationships: List[Relationship]):
        """Add relationships to the knowledge graph"""
        for rel in relationships:
            if rel.id not in self.relationships:
                self.relationships[rel.id] = rel
                
                # Update entity nodes
                if rel.source_entity_id in self.nodes:
                    self.nodes[rel.source_entity_id].relationships.append(rel.id)
                if rel.target_entity_id in self.nodes:
                    self.nodes[rel.target_entity_id].relationships.append(rel.id)
                
                # Add to NetworkX graph (for in-memory operations)
                self.graph.add_edge(rel.source_entity_id, rel.target_entity_id,
                                  relationship_type=rel.relationship_type,
                                  confidence=rel.confidence,
                                  id=rel.id)
                
                # Add to Neo4j (for persistent storage)
                if neo4j_service.is_connected():
                    success = neo4j_service.create_relationship(
                        rel.source_entity_id,
                        rel.target_entity_id,
                        rel.relationship_type,
                        rel.confidence,
                        rel.evidence,
                        rel.source_document_id
                    )
                    if not success:
                        logger.warning(f"Failed to persist relationship to Neo4j: {rel.id}")
                else:
                    logger.warning("Neo4j not connected, relationship not persisted")
    
    def _update_graph_analytics(self):
        """Update graph analytics like centrality and importance scores"""
        try:
            if self.graph.number_of_nodes() == 0:
                return
            
            # Calculate centrality measures
            try:
                centrality = nx.degree_centrality(self.graph)
                betweenness = nx.betweenness_centrality(self.graph)
                pagerank = nx.pagerank(self.graph)
                
                # Update node scores
                for node_id in self.nodes:
                    if node_id in centrality:
                        self.nodes[node_id].centrality_score = centrality[node_id]
                        # Combine different centrality measures for importance
                        importance = (
                            centrality.get(node_id, 0) * 0.3 +
                            betweenness.get(node_id, 0) * 0.3 +
                            pagerank.get(node_id, 0) * 0.4
                        )
                        self.nodes[node_id].importance_score = importance
                        
            except Exception as e:
                logger.warning(f"Failed to calculate graph centrality: {e}")
            
        except Exception as e:
            logger.error(f"Failed to update graph analytics: {e}")
    
    def _get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        try:
            stats = {
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "total_clusters": len(self.clusters),
                "graph_density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
                "connected_components": nx.number_weakly_connected_components(self.graph),
                "average_degree": sum(degree for node, degree in self.graph.degree()) / max(1, self.graph.number_of_nodes())
            }
            
            # Entity type distribution
            entity_types = Counter(entity.label for entity in self.entities.values())
            stats["entity_type_distribution"] = dict(entity_types)
            
            # Relationship type distribution
            rel_types = Counter(rel.relationship_type for rel in self.relationships.values())
            stats["relationship_type_distribution"] = dict(rel_types)
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get graph statistics", error=e)
            return {}
    
    def get_entity_neighbors(self, entity_id: str, max_distance: int = 2) -> Dict[str, Any]:
        """Get neighboring entities within max_distance"""
        try:
            # Try Neo4j first for persistent data
            if neo4j_service.is_connected():
                return neo4j_service.get_entity_neighbors(entity_id, max_distance)
            
            # Fallback to in-memory NetworkX graph
            if entity_id not in self.graph:
                return {"entities": [], "relationships": []}
            
            # Get neighbors using NetworkX
            neighbors = set()
            current_level = {entity_id}
            
            for distance in range(max_distance):
                next_level = set()
                for node in current_level:
                    node_neighbors = set(self.graph.neighbors(node)) | set(self.graph.predecessors(node))
                    next_level.update(node_neighbors)
                
                neighbors.update(next_level)
                current_level = next_level
            
            # Remove the original entity
            neighbors.discard(entity_id)
            
            # Get entity and relationship data
            neighbor_entities = [self.entities[eid].to_dict() for eid in neighbors if eid in self.entities]
            neighbor_relationships = []
            
            for rel in self.relationships.values():
                if (rel.source_entity_id in neighbors or rel.target_entity_id in neighbors) and \
                   (rel.source_entity_id == entity_id or rel.target_entity_id == entity_id or \
                    (rel.source_entity_id in neighbors and rel.target_entity_id in neighbors)):
                    neighbor_relationships.append(rel.to_dict())
            
            return {
                "entities": neighbor_entities,
                "relationships": neighbor_relationships,
                "neighbor_count": len(neighbors)
            }
            
        except Exception as e:
            logger.error("Failed to get entity neighbors", error=e)
            return {"entities": [], "relationships": []}
    
    def search_entities(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search entities by text or label"""
        try:
            # Try Neo4j first for persistent data
            if neo4j_service.is_connected():
                neo4j_results = neo4j_service.search_entities(query, limit)
                return [node.to_dict() for node in neo4j_results]
            
            # Fallback to in-memory search
            query_lower = query.lower()
            matches = []
            
            for entity in self.entities.values():
                # Simple text matching - could be enhanced with fuzzy matching
                if (query_lower in entity.text.lower() or 
                    query_lower in entity.label.lower() or
                    (entity.context and query_lower in entity.context.lower())):
                    
                    # Calculate relevance score
                    relevance = 0.0
                    if query_lower == entity.text.lower():
                        relevance = 1.0
                    elif entity.text.lower().startswith(query_lower):
                        relevance = 0.8
                    elif query_lower in entity.text.lower():
                        relevance = 0.6
                    else:
                        relevance = 0.4
                    
                    match = entity.to_dict()
                    match["relevance_score"] = relevance
                    match["importance_score"] = self.nodes[entity.id].importance_score
                    matches.append(match)
            
            # Sort by relevance and importance
            matches.sort(key=lambda x: (x["relevance_score"], x["importance_score"]), reverse=True)
            
            return matches[:limit]
            
        except Exception as e:
            logger.error("Failed to search entities", error=e)
            return []
    
    def find_shortest_path(self, source_id: str, target_id: str, max_length: int = 6) -> Optional[Dict[str, Any]]:
        """Find shortest path between two entities"""
        try:
            # Try Neo4j first for persistent data
            if neo4j_service.is_connected():
                path = neo4j_service.find_shortest_path(source_id, target_id, max_length)
                return path.to_dict() if path else None
            
            # Fallback to NetworkX
            if source_id not in self.graph or target_id not in self.graph:
                return None
            
            try:
                path = nx.shortest_path(self.graph, source_id, target_id)
                return {
                    "nodes": [{"id": node_id, "entity": self.entities[node_id].to_dict()} 
                             for node_id in path if node_id in self.entities],
                    "length": len(path) - 1
                }
            except nx.NetworkXNoPath:
                return None
                
        except Exception as e:
            logger.error("Failed to find shortest path", error=e)
            return None
    
    def find_common_neighbors(self, entity1_id: str, entity2_id: str) -> List[Dict[str, Any]]:
        """Find entities that are connected to both given entities"""
        try:
            # Try Neo4j first
            if neo4j_service.is_connected():
                common_nodes = neo4j_service.find_common_neighbors(entity1_id, entity2_id)
                return [node.to_dict() for node in common_nodes]
            
            # Fallback to NetworkX
            if entity1_id not in self.graph or entity2_id not in self.graph:
                return []
            
            neighbors1 = set(self.graph.neighbors(entity1_id)) | set(self.graph.predecessors(entity1_id))
            neighbors2 = set(self.graph.neighbors(entity2_id)) | set(self.graph.predecessors(entity2_id))
            
            common = neighbors1.intersection(neighbors2)
            return [self.entities[eid].to_dict() for eid in common if eid in self.entities]
            
        except Exception as e:
            logger.error("Failed to find common neighbors", error=e)
            return []
    
    def get_most_connected_entities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get entities with the highest degree centrality"""
        try:
            # Try Neo4j first
            if neo4j_service.is_connected():
                return neo4j_service.get_most_connected_entities(limit)
            
            # Fallback to NetworkX
            if self.graph.number_of_nodes() == 0:
                return []
            
            degree_centrality = nx.degree_centrality(self.graph)
            sorted_entities = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
            
            result = []
            for entity_id, centrality in sorted_entities[:limit]:
                if entity_id in self.entities:
                    result.append({
                        "entity": self.entities[entity_id].to_dict(),
                        "degree": self.graph.degree(entity_id),
                        "centrality": centrality
                    })
            
            return result
            
        except Exception as e:
            logger.error("Failed to get most connected entities", error=e)
            return []
    
    def detect_communities(self, min_community_size: int = 3) -> List[Dict[str, Any]]:
        """Detect communities in the knowledge graph"""
        try:
            # Try Neo4j first (if GDS library is available)
            if neo4j_service.is_connected():
                return neo4j_service.detect_communities(min_community_size)
            
            # Fallback to NetworkX community detection
            if self.graph.number_of_nodes() < min_community_size:
                return []
            
            # Use simple connected components as communities
            communities = []
            for i, component in enumerate(nx.weakly_connected_components(self.graph)):
                if len(component) >= min_community_size:
                    community_entities = []
                    for entity_id in component:
                        if entity_id in self.entities:
                            community_entities.append(self.entities[entity_id].to_dict())
                    
                    if community_entities:
                        communities.append({
                            "community_id": i,
                            "entities": community_entities,
                            "size": len(community_entities)
                        })
            
            return communities
            
        except Exception as e:
            logger.error("Failed to detect communities", error=e)
            return []
    
    def delete_document_entities(self, document_id: str) -> bool:
        """Delete all entities and relationships for a specific document"""
        try:
            # Delete from Neo4j first
            if neo4j_service.is_connected():
                neo4j_service.delete_document_entities(document_id)
            
            # Delete from in-memory structures
            entities_to_remove = []
            relationships_to_remove = []
            
            # Find entities from this document
            for entity_id, entity in self.entities.items():
                if hasattr(entity, 'source_document_id') and entity.source_document_id == document_id:
                    entities_to_remove.append(entity_id)
            
            # Find relationships from this document
            for rel_id, rel in self.relationships.items():
                if rel.source_document_id == document_id:
                    relationships_to_remove.append(rel_id)
            
            # Remove from NetworkX graph
            for entity_id in entities_to_remove:
                if entity_id in self.graph:
                    self.graph.remove_node(entity_id)
                del self.entities[entity_id]
                if entity_id in self.nodes:
                    del self.nodes[entity_id]
            
            # Remove relationships
            for rel_id in relationships_to_remove:
                del self.relationships[rel_id]
            
            logger.info(f"Deleted {len(entities_to_remove)} entities and {len(relationships_to_remove)} relationships for document {document_id}")
            return True
            
        except Exception as e:
            logger.error("Failed to delete document entities", error=e)
            return False
    
    def export_graph(self, format: str = "json") -> Dict[str, Any]:
        """Export knowledge graph in various formats"""
        try:
            if format == "json":
                # Get data from Neo4j if available, otherwise use in-memory data
                if neo4j_service.is_connected():
                    stats = neo4j_service.get_graph_statistics()
                else:
                    stats = self._get_graph_statistics()
                
                return {
                    "entities": [entity.to_dict() for entity in self.entities.values()],
                    "relationships": [rel.to_dict() for rel in self.relationships.values()],
                    "clusters": [cluster.to_dict() for cluster in self.clusters.values()],
                    "statistics": stats,
                    "exported_at": datetime.utcnow().isoformat(),
                    "neo4j_connected": neo4j_service.is_connected()
                }
            else:
                return {"error": f"Unsupported format: {format}"}
                
        except Exception as e:
            logger.error("Failed to export graph", error=e)
            return {"error": str(e)}


# Global knowledge graph builder instance
knowledge_graph = KnowledgeGraphBuilder()