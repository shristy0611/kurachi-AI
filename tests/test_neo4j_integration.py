"""
Test Neo4j Knowledge Graph Integration for Kurachi AI
Tests Neo4j database operations, graph traversal, and visualization APIs
"""
import pytest
import uuid
from datetime import datetime
from typing import Dict, List, Any

from services.neo4j_service import neo4j_service, GraphNode, GraphRelationship
from services.graph_visualization import graph_visualization_service
from services.entity_extraction import Entity
from services.knowledge_graph import knowledge_graph
from utils.logger import get_logger

logger = get_logger("test_neo4j")


class TestNeo4jService:
    """Test Neo4j database service"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment"""
        # Skip tests if Neo4j is not available
        if not neo4j_service.is_connected():
            pytest.skip("Neo4j not available for testing")
    
    def test_connection(self):
        """Test Neo4j connection"""
        assert neo4j_service.is_connected()
        logger.info("Neo4j connection test passed")
    
    def test_create_entity_node(self):
        """Test creating entity nodes"""
        # Create test entity
        entity = Entity(
            id=str(uuid.uuid4()),
            text="Test Company",
            label="COMPANY_EN",
            confidence=0.9,
            start_char=0,
            end_char=12,
            context="Test Company is a leading technology firm",
            source_document_id="test_doc_1",
            created_at=datetime.utcnow()
        )
        
        # Create entity node
        success = neo4j_service.create_entity_node(entity, "test_doc_1")
        assert success
        
        # Verify entity was created
        retrieved_entity = neo4j_service.get_entity_by_id(entity.id)
        assert retrieved_entity is not None
        assert retrieved_entity.properties["text"] == "Test Company"
        assert retrieved_entity.properties["label"] == "COMPANY_EN"
        
        logger.info("Entity node creation test passed")
    
    def test_create_relationship(self):
        """Test creating relationships between entities"""
        # Create two test entities
        entity1 = Entity(
            id=str(uuid.uuid4()),
            text="John Doe",
            label="PERSON",
            confidence=0.95,
            start_char=0,
            end_char=8,
            source_document_id="test_doc_2"
        )
        
        entity2 = Entity(
            id=str(uuid.uuid4()),
            text="Acme Corp",
            label="COMPANY_EN",
            confidence=0.9,
            start_char=20,
            end_char=29,
            source_document_id="test_doc_2"
        )
        
        # Create entity nodes
        neo4j_service.create_entity_node(entity1, "test_doc_2")
        neo4j_service.create_entity_node(entity2, "test_doc_2")
        
        # Create relationship
        evidence = {
            "method": "pattern_matching",
            "context": "John Doe works at Acme Corp"
        }
        
        success = neo4j_service.create_relationship(
            entity1.id, entity2.id, "WORKS_AT", 0.8, evidence, "test_doc_2"
        )
        assert success
        
        # Verify relationship exists
        neighbors = neo4j_service.get_entity_neighbors(entity1.id, 1)
        assert len(neighbors["entities"]) > 0
        assert len(neighbors["relationships"]) > 0
        
        logger.info("Relationship creation test passed")
    
    def test_search_entities(self):
        """Test entity search functionality"""
        # Create test entities with searchable text
        test_entities = [
            ("Apple Inc", "COMPANY_EN"),
            ("Apple Pie", "PRODUCT"),
            ("Steve Jobs", "PERSON"),
            ("iPhone", "PRODUCT")
        ]
        
        entity_ids = []
        for text, label in test_entities:
            entity = Entity(
                id=str(uuid.uuid4()),
                text=text,
                label=label,
                confidence=0.9,
                start_char=0,
                end_char=len(text),
                source_document_id="test_doc_search"
            )
            neo4j_service.create_entity_node(entity, "test_doc_search")
            entity_ids.append(entity.id)
        
        # Test search
        results = neo4j_service.search_entities("Apple", 10)
        assert len(results) >= 2  # Should find "Apple Inc" and "Apple Pie"
        
        # Test specific search
        results = neo4j_service.search_entities("Steve", 10)
        assert len(results) >= 1  # Should find "Steve Jobs"
        
        logger.info("Entity search test passed")
    
    def test_graph_traversal(self):
        """Test graph traversal algorithms"""
        # Create a small test graph
        entities = []
        for i, name in enumerate(["Alice", "Bob", "Charlie", "David"]):
            entity = Entity(
                id=f"person_{i}",
                text=name,
                label="PERSON",
                confidence=0.9,
                start_char=0,
                end_char=len(name),
                source_document_id="test_doc_traversal"
            )
            entities.append(entity)
            neo4j_service.create_entity_node(entity, "test_doc_traversal")
        
        # Create relationships: Alice -> Bob -> Charlie -> David
        relationships = [
            ("person_0", "person_1", "KNOWS"),
            ("person_1", "person_2", "WORKS_WITH"),
            ("person_2", "person_3", "MANAGES")
        ]
        
        for source, target, rel_type in relationships:
            neo4j_service.create_relationship(
                source, target, rel_type, 0.8, 
                {"method": "test"}, "test_doc_traversal"
            )
        
        # Test shortest path
        path = neo4j_service.find_shortest_path("person_0", "person_3", 5)
        assert path is not None
        assert path.length == 3  # Alice -> Bob -> Charlie -> David
        
        # Test common neighbors
        # Add a connection: Alice -> Charlie (so Bob is common neighbor)
        neo4j_service.create_relationship(
            "person_0", "person_2", "COLLABORATES", 0.7,
            {"method": "test"}, "test_doc_traversal"
        )
        
        common = neo4j_service.find_common_neighbors("person_1", "person_2")
        # Bob should be connected to both Alice and Charlie
        
        logger.info("Graph traversal test passed")
    
    def test_graph_statistics(self):
        """Test graph statistics collection"""
        stats = neo4j_service.get_graph_statistics()
        
        assert isinstance(stats, dict)
        assert "total_entities" in stats
        assert "total_relationships" in stats
        assert "total_documents" in stats
        
        # Statistics should be non-negative
        assert stats["total_entities"] >= 0
        assert stats["total_relationships"] >= 0
        assert stats["total_documents"] >= 0
        
        logger.info("Graph statistics test passed")


class TestGraphVisualization:
    """Test graph visualization service"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment"""
        if not neo4j_service.is_connected():
            pytest.skip("Neo4j not available for testing")
    
    def test_entity_neighborhood_visualization(self):
        """Test entity neighborhood visualization"""
        # Create test entity with neighbors
        central_entity = Entity(
            id="central_test",
            text="Central Entity",
            label="ORGANIZATION",
            confidence=0.9,
            start_char=0,
            end_char=14,
            source_document_id="test_doc_viz"
        )
        
        neo4j_service.create_entity_node(central_entity, "test_doc_viz")
        
        # Create neighbor entities
        neighbors = []
        for i in range(3):
            neighbor = Entity(
                id=f"neighbor_{i}",
                text=f"Neighbor {i}",
                label="PERSON",
                confidence=0.8,
                start_char=0,
                end_char=10,
                source_document_id="test_doc_viz"
            )
            neighbors.append(neighbor)
            neo4j_service.create_entity_node(neighbor, "test_doc_viz")
            
            # Create relationship to central entity
            neo4j_service.create_relationship(
                neighbor.id, central_entity.id, "ASSOCIATED_WITH", 0.7,
                {"method": "test"}, "test_doc_viz"
            )
        
        # Test visualization creation
        viz = graph_visualization_service.create_entity_neighborhood_visualization(
            "central_test", max_distance=2
        )
        
        assert viz is not None
        assert len(viz.nodes) >= 4  # Central + 3 neighbors
        assert len(viz.edges) >= 3  # 3 relationships
        assert viz.metadata["central_entity_id"] == "central_test"
        
        # Check that nodes have positions
        for node in viz.nodes:
            assert node.x is not None
            assert node.y is not None
        
        logger.info("Entity neighborhood visualization test passed")
    
    def test_path_visualization(self):
        """Test path visualization between entities"""
        # Create a path: Entity A -> Entity B -> Entity C
        entities = []
        for i, name in enumerate(["Entity A", "Entity B", "Entity C"]):
            entity = Entity(
                id=f"path_entity_{i}",
                text=name,
                label="CONCEPT",
                confidence=0.9,
                start_char=0,
                end_char=len(name),
                source_document_id="test_doc_path"
            )
            entities.append(entity)
            neo4j_service.create_entity_node(entity, "test_doc_path")
        
        # Create path relationships
        neo4j_service.create_relationship(
            "path_entity_0", "path_entity_1", "LEADS_TO", 0.8,
            {"method": "test"}, "test_doc_path"
        )
        neo4j_service.create_relationship(
            "path_entity_1", "path_entity_2", "RESULTS_IN", 0.8,
            {"method": "test"}, "test_doc_path"
        )
        
        # Test path visualization
        viz = graph_visualization_service.create_path_visualization(
            "path_entity_0", "path_entity_2"
        )
        
        assert viz is not None
        assert len(viz.nodes) == 3  # A, B, C
        assert len(viz.edges) == 2  # A->B, B->C
        assert viz.metadata["path_length"] == 2
        
        # Check linear layout
        x_positions = [node.x for node in viz.nodes]
        assert len(set(x_positions)) == 3  # All different x positions
        
        logger.info("Path visualization test passed")
    
    def test_visualization_statistics(self):
        """Test visualization statistics"""
        stats = graph_visualization_service.get_graph_statistics_for_visualization()
        
        assert isinstance(stats, dict)
        assert "total_entities" in stats or "total_communities" in stats
        
        logger.info("Visualization statistics test passed")


class TestKnowledgeGraphIntegration:
    """Test integration between knowledge graph and Neo4j"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment"""
        if not neo4j_service.is_connected():
            pytest.skip("Neo4j not available for testing")
    
    def test_document_processing_with_neo4j(self):
        """Test document processing with Neo4j persistence"""
        # Create a test document
        test_doc_id = str(uuid.uuid4())
        
        # Process document (this should create entities and relationships in Neo4j)
        result = knowledge_graph.process_document(test_doc_id)
        
        # The process_document method should work even if document doesn't exist
        # (it uses placeholder text for testing)
        assert isinstance(result, dict)
        assert "success" in result
        
        logger.info("Document processing with Neo4j test passed")
    
    def test_hybrid_search(self):
        """Test hybrid search using both in-memory and Neo4j data"""
        # Create test entity
        entity = Entity(
            id="hybrid_test",
            text="Hybrid Search Test",
            label="CONCEPT",
            confidence=0.9,
            start_char=0,
            end_char=18,
            source_document_id="test_doc_hybrid"
        )
        
        # Add to both in-memory and Neo4j
        knowledge_graph._add_entities_to_graph([entity])
        
        # Test search
        results = knowledge_graph.search_entities("Hybrid", 10)
        assert len(results) > 0
        
        # Should find the entity we created
        found = any(r.get("id") == "hybrid_test" or 
                   r.get("properties", {}).get("text") == "Hybrid Search Test" 
                   for r in results)
        assert found
        
        logger.info("Hybrid search test passed")
    
    def test_graph_operations_fallback(self):
        """Test that graph operations work with Neo4j unavailable"""
        # Temporarily disconnect Neo4j
        original_connected = neo4j_service.connected
        neo4j_service.connected = False
        
        try:
            # Test operations should still work with in-memory fallback
            results = knowledge_graph.search_entities("test", 5)
            assert isinstance(results, list)
            
            neighbors = knowledge_graph.get_entity_neighbors("nonexistent", 1)
            assert isinstance(neighbors, dict)
            assert "entities" in neighbors
            
        finally:
            # Restore connection status
            neo4j_service.connected = original_connected
        
        logger.info("Graph operations fallback test passed")


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_data():
    """Clean up test data after all tests"""
    yield
    
    if neo4j_service.is_connected():
        try:
            # Clean up test documents and their entities
            test_docs = [
                "test_doc_1", "test_doc_2", "test_doc_search", 
                "test_doc_traversal", "test_doc_viz", "test_doc_path",
                "test_doc_hybrid"
            ]
            
            for doc_id in test_docs:
                neo4j_service.delete_document_entities(doc_id)
            
            logger.info("Test data cleanup completed")
            
        except Exception as e:
            logger.warning(f"Test cleanup failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])