"""
Neo4j Knowledge Graph Database Service for Kurachi AI
Provides persistent graph storage and advanced graph operations
"""
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError, ConfigurationError

from config import config
from services.entity_extraction import Entity
from utils.logger import get_logger

logger = get_logger("neo4j_service")


@dataclass
class GraphNode:
    """Represents a node in the Neo4j graph"""
    id: str
    labels: List[str]
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "labels": self.labels,
            "properties": self.properties
        }


@dataclass
class GraphRelationship:
    """Represents a relationship in the Neo4j graph"""
    id: str
    start_node_id: str
    end_node_id: str
    type: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "type": self.type,
            "properties": self.properties
        }


@dataclass
class GraphPath:
    """Represents a path in the graph"""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    length: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "length": self.length
        }


class Neo4jService:
    """Neo4j database service for knowledge graph operations"""
    
    def __init__(self):
        self.driver: Optional[Driver] = None
        self.connected = False
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                config.database.neo4j_uri,
                auth=(config.database.neo4j_username, config.database.neo4j_password)
            )
            
            # Test connection
            with self.driver.session(database=config.database.neo4j_database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            self.connected = True
            logger.info("Successfully connected to Neo4j database")
            
            # Initialize schema
            self._initialize_schema()
            
        except (ServiceUnavailable, AuthError, ConfigurationError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.connected = False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            self.connected = False
    
    def _initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes"""
        try:
            with self.driver.session(database=config.database.neo4j_database) as session:
                # Create constraints for unique entity IDs
                constraints = [
                    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                    "CREATE CONSTRAINT relationship_id_unique IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() REQUIRE r.id IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        # Constraint might already exist
                        logger.debug(f"Constraint creation result: {e}")
                
                # Create indexes for better performance
                indexes = [
                    "CREATE INDEX entity_text_index IF NOT EXISTS FOR (e:Entity) ON (e.text)",
                    "CREATE INDEX entity_label_index IF NOT EXISTS FOR (e:Entity) ON (e.label)",
                    "CREATE INDEX document_filename_index IF NOT EXISTS FOR (d:Document) ON (d.filename)",
                    "CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.type)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.debug(f"Index creation result: {e}")
                
                logger.info("Neo4j schema initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j schema: {e}")
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Neo4j connection closed")
    
    def is_connected(self) -> bool:
        """Check if connected to Neo4j"""
        return self.connected
    
    # Node operations
    def create_entity_node(self, entity: Entity, document_id: str) -> bool:
        """Create an entity node in Neo4j"""
        if not self.connected:
            logger.warning("Neo4j not connected, cannot create entity node")
            return False
        
        try:
            with self.driver.session(database=config.database.neo4j_database) as session:
                query = """
                MERGE (e:Entity {id: $entity_id})
                SET e.text = $text,
                    e.label = $label,
                    e.confidence = $confidence,
                    e.start_char = $start_char,
                    e.end_char = $end_char,
                    e.context = $context,
                    e.created_at = $created_at,
                    e.updated_at = $updated_at
                WITH e
                MERGE (d:Document {id: $document_id})
                MERGE (e)-[:EXTRACTED_FROM]->(d)
                RETURN e.id as entity_id
                """
                
                result = session.run(query, {
                    "entity_id": entity.id,
                    "text": entity.text,
                    "label": entity.label,
                    "confidence": entity.confidence,
                    "start_char": entity.start_char,
                    "end_char": entity.end_char,
                    "context": entity.context or "",
                    "document_id": document_id,
                    "created_at": entity.created_at.isoformat() if entity.created_at else datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                })
                
                record = result.single()
                if record:
                    logger.debug(f"Created entity node: {record['entity_id']}")
                    return True
                
        except Exception as e:
            logger.error(f"Failed to create entity node: {e}")
        
        return False
    
    def create_relationship(self, source_entity_id: str, target_entity_id: str, 
                          relationship_type: str, confidence: float, 
                          evidence: Dict[str, Any], document_id: str) -> bool:
        """Create a relationship between two entities"""
        if not self.connected:
            logger.warning("Neo4j not connected, cannot create relationship")
            return False
        
        try:
            with self.driver.session(database=config.database.neo4j_database) as session:
                relationship_id = str(uuid.uuid4())
                
                query = """
                MATCH (source:Entity {id: $source_id})
                MATCH (target:Entity {id: $target_id})
                MERGE (source)-[r:RELATIONSHIP {
                    id: $rel_id,
                    type: $rel_type,
                    confidence: $confidence,
                    evidence: $evidence,
                    document_id: $document_id,
                    created_at: $created_at
                }]->(target)
                RETURN r.id as relationship_id
                """
                
                result = session.run(query, {
                    "source_id": source_entity_id,
                    "target_id": target_entity_id,
                    "rel_id": relationship_id,
                    "rel_type": relationship_type,
                    "confidence": confidence,
                    "evidence": json.dumps(evidence),
                    "document_id": document_id,
                    "created_at": datetime.utcnow().isoformat()
                })
                
                record = result.single()
                if record:
                    logger.debug(f"Created relationship: {record['relationship_id']}")
                    return True
                
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
        
        return False
    
    # Query operations
    def get_entity_by_id(self, entity_id: str) -> Optional[GraphNode]:
        """Get entity by ID"""
        if not self.connected:
            return None
        
        try:
            with self.driver.session(database=config.database.neo4j_database) as session:
                query = "MATCH (e:Entity {id: $entity_id}) RETURN e"
                result = session.run(query, {"entity_id": entity_id})
                record = result.single()
                
                if record:
                    node = record["e"]
                    return GraphNode(
                        id=node["id"],
                        labels=list(node.labels),
                        properties=dict(node)
                    )
                
        except Exception as e:
            logger.error(f"Failed to get entity by ID: {e}")
        
        return None
    
    def search_entities(self, query: str, limit: int = 20) -> List[GraphNode]:
        """Search entities by text or label"""
        if not self.connected:
            return []
        
        try:
            with self.driver.session(database=config.database.neo4j_database) as session:
                cypher_query = """
                MATCH (e:Entity)
                WHERE toLower(e.text) CONTAINS toLower($query) 
                   OR toLower(e.label) CONTAINS toLower($query)
                   OR toLower(e.context) CONTAINS toLower($query)
                RETURN e
                ORDER BY e.confidence DESC, e.text
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {"query": query, "limit": limit})
                
                entities = []
                for record in result:
                    node = record["e"]
                    entities.append(GraphNode(
                        id=node["id"],
                        labels=list(node.labels),
                        properties=dict(node)
                    ))
                
                return entities
                
        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
        
        return []
    
    def get_entity_neighbors(self, entity_id: str, max_distance: int = 2) -> Dict[str, Any]:
        """Get neighboring entities within max_distance"""
        if not self.connected:
            return {"entities": [], "relationships": [], "paths": []}
        
        try:
            with self.driver.session(database=config.database.neo4j_database) as session:
                query = """
                MATCH path = (start:Entity {id: $entity_id})-[*1..$max_distance]-(neighbor:Entity)
                WHERE start <> neighbor
                RETURN DISTINCT neighbor, 
                       relationships(path) as rels,
                       length(path) as distance
                ORDER BY distance, neighbor.confidence DESC
                LIMIT 100
                """
                
                result = session.run(query, {
                    "entity_id": entity_id, 
                    "max_distance": max_distance
                })
                
                neighbors = []
                relationships = []
                
                for record in result:
                    neighbor = record["neighbor"]
                    neighbors.append(GraphNode(
                        id=neighbor["id"],
                        labels=list(neighbor.labels),
                        properties=dict(neighbor)
                    ))
                    
                    # Extract relationships
                    for rel in record["rels"]:
                        relationships.append(GraphRelationship(
                            id=rel["id"],
                            start_node_id=rel.start_node["id"],
                            end_node_id=rel.end_node["id"],
                            type=rel["type"],
                            properties=dict(rel)
                        ))
                
                return {
                    "entities": [n.to_dict() for n in neighbors],
                    "relationships": [r.to_dict() for r in relationships],
                    "neighbor_count": len(neighbors)
                }
                
        except Exception as e:
            logger.error(f"Failed to get entity neighbors: {e}")
        
        return {"entities": [], "relationships": [], "neighbor_count": 0}
    
    # Graph traversal algorithms
    def find_shortest_path(self, source_id: str, target_id: str, max_length: int = 6) -> Optional[GraphPath]:
        """Find shortest path between two entities"""
        if not self.connected:
            return None
        
        try:
            with self.driver.session(database=config.database.neo4j_database) as session:
                query = """
                MATCH path = shortestPath((source:Entity {id: $source_id})-[*1..$max_length]-(target:Entity {id: $target_id}))
                RETURN nodes(path) as nodes, relationships(path) as rels, length(path) as length
                """
                
                result = session.run(query, {
                    "source_id": source_id,
                    "target_id": target_id,
                    "max_length": max_length
                })
                
                record = result.single()
                if record:
                    nodes = [GraphNode(
                        id=node["id"],
                        labels=list(node.labels),
                        properties=dict(node)
                    ) for node in record["nodes"]]
                    
                    relationships = [GraphRelationship(
                        id=rel["id"],
                        start_node_id=rel.start_node["id"],
                        end_node_id=rel.end_node["id"],
                        type=rel["type"],
                        properties=dict(rel)
                    ) for rel in record["rels"]]
                    
                    return GraphPath(
                        nodes=nodes,
                        relationships=relationships,
                        length=record["length"]
                    )
                
        except Exception as e:
            logger.error(f"Failed to find shortest path: {e}")
        
        return None
    
    def find_common_neighbors(self, entity1_id: str, entity2_id: str) -> List[GraphNode]:
        """Find entities that are connected to both given entities"""
        if not self.connected:
            return []
        
        try:
            with self.driver.session(database=config.database.neo4j_database) as session:
                query = """
                MATCH (e1:Entity {id: $entity1_id})-[]-(common:Entity)-[]-(e2:Entity {id: $entity2_id})
                WHERE e1 <> common AND e2 <> common AND e1 <> e2
                RETURN DISTINCT common
                ORDER BY common.confidence DESC
                LIMIT 50
                """
                
                result = session.run(query, {
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id
                })
                
                common_neighbors = []
                for record in result:
                    node = record["common"]
                    common_neighbors.append(GraphNode(
                        id=node["id"],
                        labels=list(node.labels),
                        properties=dict(node)
                    ))
                
                return common_neighbors
                
        except Exception as e:
            logger.error(f"Failed to find common neighbors: {e}")
        
        return []
    
    def get_most_connected_entities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get entities with the highest degree centrality"""
        if not self.connected:
            return []
        
        try:
            with self.driver.session(database=config.database.neo4j_database) as session:
                query = """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r:RELATIONSHIP]-()
                WITH e, count(r) as degree
                RETURN e, degree
                ORDER BY degree DESC, e.confidence DESC
                LIMIT $limit
                """
                
                result = session.run(query, {"limit": limit})
                
                entities = []
                for record in result:
                    node = record["e"]
                    entities.append({
                        "entity": GraphNode(
                            id=node["id"],
                            labels=list(node.labels),
                            properties=dict(node)
                        ).to_dict(),
                        "degree": record["degree"]
                    })
                
                return entities
                
        except Exception as e:
            logger.error(f"Failed to get most connected entities: {e}")
        
        return []
    
    def detect_communities(self, min_community_size: int = 3) -> List[Dict[str, Any]]:
        """Detect communities in the knowledge graph using Louvain algorithm"""
        if not self.connected:
            return []
        
        try:
            with self.driver.session(database=config.database.neo4j_database) as session:
                # First, create a graph projection for community detection
                projection_query = """
                CALL gds.graph.project(
                    'entityGraph',
                    'Entity',
                    'RELATIONSHIP'
                )
                """
                
                # Run Louvain community detection
                louvain_query = """
                CALL gds.louvain.stream('entityGraph')
                YIELD nodeId, communityId
                RETURN gds.util.asNode(nodeId) as entity, communityId
                """
                
                # Clean up projection
                cleanup_query = "CALL gds.graph.drop('entityGraph')"
                
                try:
                    session.run(projection_query)
                    result = session.run(louvain_query)
                    
                    communities = {}
                    for record in result:
                        entity = record["entity"]
                        community_id = record["communityId"]
                        
                        if community_id not in communities:
                            communities[community_id] = []
                        
                        communities[community_id].append(GraphNode(
                            id=entity["id"],
                            labels=list(entity.labels),
                            properties=dict(entity)
                        ))
                    
                    # Filter communities by minimum size
                    filtered_communities = []
                    for comm_id, entities in communities.items():
                        if len(entities) >= min_community_size:
                            filtered_communities.append({
                                "community_id": comm_id,
                                "entities": [e.to_dict() for e in entities],
                                "size": len(entities)
                            })
                    
                    session.run(cleanup_query)
                    return filtered_communities
                    
                except Exception as e:
                    # Try cleanup even if detection failed
                    try:
                        session.run(cleanup_query)
                    except:
                        pass
                    raise e
                
        except Exception as e:
            logger.error(f"Failed to detect communities: {e}")
        
        return []
    
    # Graph statistics and analytics
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        if not self.connected:
            return {}
        
        try:
            with self.driver.session(database=config.database.neo4j_database) as session:
                stats_query = """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r:RELATIONSHIP]-()
                OPTIONAL MATCH (d:Document)
                RETURN 
                    count(DISTINCT e) as total_entities,
                    count(DISTINCT r) as total_relationships,
                    count(DISTINCT d) as total_documents,
                    avg(r.confidence) as avg_relationship_confidence,
                    max(r.confidence) as max_relationship_confidence,
                    min(r.confidence) as min_relationship_confidence
                """
                
                result = session.run(stats_query)
                record = result.single()
                
                if record:
                    return {
                        "total_entities": record["total_entities"],
                        "total_relationships": record["total_relationships"],
                        "total_documents": record["total_documents"],
                        "avg_relationship_confidence": record["avg_relationship_confidence"],
                        "max_relationship_confidence": record["max_relationship_confidence"],
                        "min_relationship_confidence": record["min_relationship_confidence"]
                    }
                
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
        
        return {}
    
    def delete_document_entities(self, document_id: str) -> bool:
        """Delete all entities and relationships for a specific document"""
        if not self.connected:
            return False
        
        try:
            with self.driver.session(database=config.database.neo4j_database) as session:
                query = """
                MATCH (d:Document {id: $document_id})
                MATCH (e:Entity)-[:EXTRACTED_FROM]->(d)
                DETACH DELETE e, d
                """
                
                session.run(query, {"document_id": document_id})
                logger.info(f"Deleted entities for document: {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete document entities: {e}")
        
        return False


# Global Neo4j service instance
neo4j_service = Neo4jService()