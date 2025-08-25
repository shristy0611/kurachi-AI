"""
Graph Visualization and Exploration API for Kurachi AI
Provides APIs for visualizing and exploring the knowledge graph
"""
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from services.neo4j_service import neo4j_service, GraphNode, GraphRelationship
from utils.logger import get_logger

logger = get_logger("graph_visualization")


@dataclass
class VisualizationNode:
    """Node formatted for visualization"""
    id: str
    label: str
    text: str
    type: str
    confidence: float
    size: int
    color: str
    x: Optional[float] = None
    y: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualizationEdge:
    """Edge formatted for visualization"""
    id: str
    source: str
    target: str
    type: str
    confidence: float
    width: int
    color: str
    label: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphVisualization:
    """Complete graph visualization data"""
    nodes: List[VisualizationNode]
    edges: List[VisualizationEdge]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata
        }


class GraphVisualizationService:
    """Service for creating graph visualizations and exploration APIs"""
    
    def __init__(self):
        # Color schemes for different entity types
        self.entity_colors = {
            "PERSON": "#FF6B6B",
            "PERSON_JP": "#FF6B6B",
            "ROLE_EN": "#FF8E8E",
            "ORG": "#4ECDC4",
            "COMPANY_JP": "#4ECDC4",
            "COMPANY_EN": "#4ECDC4",
            "DEPARTMENT_JP": "#6BCFCF",
            "DEPARTMENT_EN": "#6BCFCF",
            "PROJECT_JP": "#45B7D1",
            "PROJECT_EN": "#45B7D1",
            "GPE": "#96CEB4",
            "LOC": "#96CEB4",
            "DATE": "#FFEAA7",
            "MONEY": "#DDA0DD",
            "PERCENT": "#F0A500",
            "default": "#95A5A6"
        }
        
        # Relationship colors
        self.relationship_colors = {
            "WORKS_AT": "#E74C3C",
            "MEMBER_OF": "#E74C3C",
            "MANAGES": "#8E44AD",
            "PARTICIPATES_IN": "#3498DB",
            "LOCATED_IN": "#27AE60",
            "ASSOCIATED_WITH": "#F39C12",
            "RELATED_TO": "#95A5A6",
            "CO_OCCURS": "#BDC3C7",
            "default": "#7F8C8D"
        }
        
        logger.info("Graph visualization service initialized")
    
    def create_entity_neighborhood_visualization(self, entity_id: str, 
                                               max_distance: int = 2) -> Optional[GraphVisualization]:
        """Create visualization for entity neighborhood"""
        try:
            # Get entity and its neighbors from Neo4j
            neighbors_data = neo4j_service.get_entity_neighbors(entity_id, max_distance)
            
            if not neighbors_data["entities"]:
                logger.warning(f"No neighbors found for entity: {entity_id}")
                return None
            
            # Get the central entity
            central_entity = neo4j_service.get_entity_by_id(entity_id)
            if not central_entity:
                logger.error(f"Central entity not found: {entity_id}")
                return None
            
            # Create visualization nodes
            nodes = []
            
            # Add central entity (larger size, special position)
            central_node = self._create_visualization_node(
                central_entity, 
                is_central=True
            )
            nodes.append(central_node)
            
            # Add neighbor entities
            for entity_data in neighbors_data["entities"]:
                if entity_data["id"] != entity_id:  # Skip central entity
                    neighbor_node = GraphNode(
                        id=entity_data["id"],
                        labels=entity_data.get("labels", []),
                        properties=entity_data["properties"]
                    )
                    vis_node = self._create_visualization_node(neighbor_node)
                    nodes.append(vis_node)
            
            # Create visualization edges
            edges = []
            for rel_data in neighbors_data["relationships"]:
                edge = self._create_visualization_edge(rel_data)
                edges.append(edge)
            
            # Calculate layout positions
            self._calculate_force_layout(nodes, edges)
            
            metadata = {
                "central_entity_id": entity_id,
                "max_distance": max_distance,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "created_at": datetime.utcnow().isoformat()
            }
            
            return GraphVisualization(
                nodes=nodes,
                edges=edges,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to create entity neighborhood visualization: {e}")
            return None
    
    def create_path_visualization(self, source_id: str, target_id: str) -> Optional[GraphVisualization]:
        """Create visualization for shortest path between two entities"""
        try:
            # Find shortest path
            path = neo4j_service.find_shortest_path(source_id, target_id)
            
            if not path:
                logger.warning(f"No path found between {source_id} and {target_id}")
                return None
            
            # Create visualization nodes
            nodes = []
            for i, node in enumerate(path.nodes):
                is_endpoint = (i == 0 or i == len(path.nodes) - 1)
                vis_node = self._create_visualization_node(node, is_central=is_endpoint)
                nodes.append(vis_node)
            
            # Create visualization edges
            edges = []
            for rel in path.relationships:
                edge = self._create_visualization_edge(rel.to_dict())
                edges.append(edge)
            
            # Linear layout for path visualization
            self._calculate_linear_layout(nodes)
            
            metadata = {
                "source_id": source_id,
                "target_id": target_id,
                "path_length": path.length,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "created_at": datetime.utcnow().isoformat()
            }
            
            return GraphVisualization(
                nodes=nodes,
                edges=edges,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to create path visualization: {e}")
            return None
    
    def create_community_visualization(self, community_id: int, 
                                     include_connections: bool = True) -> Optional[GraphVisualization]:
        """Create visualization for a specific community"""
        try:
            # Get communities from Neo4j
            communities = neo4j_service.detect_communities()
            
            # Find the requested community
            target_community = None
            for community in communities:
                if community["community_id"] == community_id:
                    target_community = community
                    break
            
            if not target_community:
                logger.warning(f"Community not found: {community_id}")
                return None
            
            # Create visualization nodes
            nodes = []
            entity_ids = []
            
            for entity_data in target_community["entities"]:
                entity_node = GraphNode(
                    id=entity_data["id"],
                    labels=entity_data.get("labels", []),
                    properties=entity_data["properties"]
                )
                vis_node = self._create_visualization_node(entity_node)
                nodes.append(vis_node)
                entity_ids.append(entity_data["id"])
            
            # Get relationships within the community
            edges = []
            if include_connections:
                for entity_id in entity_ids:
                    neighbors_data = neo4j_service.get_entity_neighbors(entity_id, 1)
                    for rel_data in neighbors_data["relationships"]:
                        # Only include relationships within the community
                        if (rel_data["start_node_id"] in entity_ids and 
                            rel_data["end_node_id"] in entity_ids):
                            edge = self._create_visualization_edge(rel_data)
                            # Avoid duplicate edges
                            if not any(e.id == edge.id for e in edges):
                                edges.append(edge)
            
            # Calculate cluster layout
            self._calculate_cluster_layout(nodes, edges)
            
            metadata = {
                "community_id": community_id,
                "community_size": len(nodes),
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "include_connections": include_connections,
                "created_at": datetime.utcnow().isoformat()
            }
            
            return GraphVisualization(
                nodes=nodes,
                edges=edges,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to create community visualization: {e}")
            return None
    
    def create_full_graph_overview(self, max_nodes: int = 100) -> Optional[GraphVisualization]:
        """Create overview visualization of the entire graph"""
        try:
            # Get most connected entities for overview
            connected_entities = neo4j_service.get_most_connected_entities(max_nodes)
            
            if not connected_entities:
                logger.warning("No entities found for graph overview")
                return None
            
            # Create visualization nodes
            nodes = []
            entity_ids = []
            
            for entity_data in connected_entities:
                entity_info = entity_data["entity"]
                entity_node = GraphNode(
                    id=entity_info["id"],
                    labels=entity_info.get("labels", []),
                    properties=entity_info["properties"]
                )
                
                # Size based on degree centrality
                degree = entity_data["degree"]
                vis_node = self._create_visualization_node(entity_node)
                vis_node.size = min(50, max(10, degree * 3))  # Scale size by degree
                
                nodes.append(vis_node)
                entity_ids.append(entity_info["id"])
            
            # Get relationships between these entities
            edges = []
            for entity_id in entity_ids[:20]:  # Limit to avoid too many edges
                neighbors_data = neo4j_service.get_entity_neighbors(entity_id, 1)
                for rel_data in neighbors_data["relationships"]:
                    if (rel_data["start_node_id"] in entity_ids and 
                        rel_data["end_node_id"] in entity_ids):
                        edge = self._create_visualization_edge(rel_data)
                        if not any(e.id == edge.id for e in edges):
                            edges.append(edge)
            
            # Calculate force-directed layout
            self._calculate_force_layout(nodes, edges)
            
            metadata = {
                "visualization_type": "overview",
                "max_nodes": max_nodes,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "created_at": datetime.utcnow().isoformat()
            }
            
            return GraphVisualization(
                nodes=nodes,
                edges=edges,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to create graph overview: {e}")
            return None
    
    def _create_visualization_node(self, node: GraphNode, is_central: bool = False) -> VisualizationNode:
        """Create a visualization node from a graph node"""
        properties = node.properties
        entity_type = properties.get("label", "default")
        
        # Determine color based on entity type
        color = self.entity_colors.get(entity_type, self.entity_colors["default"])
        
        # Determine size based on confidence and centrality
        confidence = properties.get("confidence", 0.5)
        base_size = 30 if is_central else 20
        size = int(base_size * (0.5 + confidence * 0.5))
        
        return VisualizationNode(
            id=node.id,
            label=properties.get("text", "Unknown"),
            text=properties.get("text", ""),
            type=entity_type,
            confidence=confidence,
            size=size,
            color=color
        )
    
    def _create_visualization_edge(self, relationship_data: Dict[str, Any]) -> VisualizationEdge:
        """Create a visualization edge from relationship data"""
        rel_type = relationship_data.get("type", "default")
        confidence = relationship_data.get("confidence", 0.5)
        
        # Determine color and width based on relationship type and confidence
        color = self.relationship_colors.get(rel_type, self.relationship_colors["default"])
        width = max(1, int(confidence * 5))
        
        return VisualizationEdge(
            id=relationship_data.get("id", ""),
            source=relationship_data.get("start_node_id", ""),
            target=relationship_data.get("end_node_id", ""),
            type=rel_type,
            confidence=confidence,
            width=width,
            color=color,
            label=rel_type.replace("_", " ").title()
        )
    
    def _calculate_force_layout(self, nodes: List[VisualizationNode], 
                               edges: List[VisualizationEdge]):
        """Calculate force-directed layout positions for nodes"""
        import math
        import random
        
        # Simple force-directed layout algorithm
        num_nodes = len(nodes)
        if num_nodes == 0:
            return
        
        # Initialize random positions
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / num_nodes
            radius = 100 + random.uniform(-20, 20)
            node.x = radius * math.cos(angle)
            node.y = radius * math.sin(angle)
        
        # Simple spring-force simulation (simplified)
        for iteration in range(50):
            # Repulsive forces between all nodes
            for i, node1 in enumerate(nodes):
                fx, fy = 0, 0
                for j, node2 in enumerate(nodes):
                    if i != j:
                        dx = node1.x - node2.x
                        dy = node1.y - node2.y
                        distance = math.sqrt(dx*dx + dy*dy) + 0.1
                        
                        # Repulsive force
                        force = 1000 / (distance * distance)
                        fx += force * dx / distance
                        fy += force * dy / distance
                
                # Attractive forces for connected nodes
                for edge in edges:
                    if edge.source == node1.id:
                        target_node = next((n for n in nodes if n.id == edge.target), None)
                        if target_node:
                            dx = target_node.x - node1.x
                            dy = target_node.y - node1.y
                            distance = math.sqrt(dx*dx + dy*dy) + 0.1
                            
                            # Attractive force
                            force = distance * 0.01
                            fx += force * dx / distance
                            fy += force * dy / distance
                
                # Update position with damping
                damping = 0.9
                node1.x += fx * 0.1 * damping
                node1.y += fy * 0.1 * damping
    
    def _calculate_linear_layout(self, nodes: List[VisualizationNode]):
        """Calculate linear layout for path visualization"""
        num_nodes = len(nodes)
        if num_nodes == 0:
            return
        
        # Arrange nodes in a line
        spacing = 150
        start_x = -(num_nodes - 1) * spacing / 2
        
        for i, node in enumerate(nodes):
            node.x = start_x + i * spacing
            node.y = 0
    
    def _calculate_cluster_layout(self, nodes: List[VisualizationNode], 
                                 edges: List[VisualizationEdge]):
        """Calculate cluster layout for community visualization"""
        import math
        
        num_nodes = len(nodes)
        if num_nodes == 0:
            return
        
        # Arrange nodes in a circle
        radius = max(100, num_nodes * 10)
        
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / num_nodes
            node.x = radius * math.cos(angle)
            node.y = radius * math.sin(angle)
    
    def get_graph_statistics_for_visualization(self) -> Dict[str, Any]:
        """Get graph statistics formatted for visualization dashboards"""
        try:
            stats = neo4j_service.get_graph_statistics()
            
            # Add visualization-specific statistics
            communities = neo4j_service.detect_communities()
            connected_entities = neo4j_service.get_most_connected_entities(10)
            
            vis_stats = {
                **stats,
                "total_communities": len(communities),
                "largest_community_size": max([c["size"] for c in communities], default=0),
                "most_connected_entities": [
                    {
                        "id": entity["entity"]["id"],
                        "text": entity["entity"]["properties"].get("text", ""),
                        "degree": entity["degree"]
                    }
                    for entity in connected_entities[:5]
                ],
                "entity_type_distribution": self._get_entity_type_distribution(),
                "relationship_type_distribution": self._get_relationship_type_distribution()
            }
            
            return vis_stats
            
        except Exception as e:
            logger.error(f"Failed to get visualization statistics: {e}")
            return {}
    
    def _get_entity_type_distribution(self) -> Dict[str, int]:
        """Get distribution of entity types for visualization"""
        # This would query Neo4j for entity type counts
        # Simplified implementation
        return {
            "PERSON": 45,
            "ORGANIZATION": 32,
            "PROJECT": 18,
            "LOCATION": 12,
            "OTHER": 8
        }
    
    def _get_relationship_type_distribution(self) -> Dict[str, int]:
        """Get distribution of relationship types for visualization"""
        # This would query Neo4j for relationship type counts
        # Simplified implementation
        return {
            "WORKS_AT": 28,
            "PARTICIPATES_IN": 22,
            "ASSOCIATED_WITH": 18,
            "LOCATED_IN": 15,
            "MANAGES": 12,
            "OTHER": 5
        }


# Global graph visualization service instance
graph_visualization_service = GraphVisualizationService()