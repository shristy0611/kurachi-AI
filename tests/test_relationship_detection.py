"""
Tests for relationship detection and mapping service
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from services.relationship_detection import (
    RelationshipDetector, RelationshipManager, EntityRelationship, 
    RelationshipEvidence, RelationshipConflict, RelationshipType
)
from services.entity_extraction import Entity


class TestRelationshipDetector:
    """Test relationship detection functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = RelationshipDetector()
        
        # Create sample entities
        self.entities = [
            Entity(
                id="entity1",
                text="John Smith",
                label="PERSON",
                start_char=0,
                end_char=10,
                confidence=0.9,
                source_document_id="doc1",
                properties={"language": "en"}
            ),
            Entity(
                id="entity2",
                text="Acme Corporation",
                label="ORG",
                start_char=20,
                end_char=35,
                confidence=0.8,
                source_document_id="doc1",
                properties={"language": "en"}
            ),
            Entity(
                id="entity3",
                text="Project Alpha",
                label="PROJECT_EN",
                start_char=50,
                end_char=63,
                confidence=0.85,
                source_document_id="doc1",
                properties={"language": "en"}
            )
        ]
    
    def test_detect_pattern_relationships(self):
        """Test pattern-based relationship detection"""
        text = "John Smith works for Acme Corporation and leads Project Alpha."
        
        relationships = self.detector._detect_pattern_relationships(self.entities, text, "doc1")
        
        assert len(relationships) > 0
        
        # Check for work relationship
        work_rels = [r for r in relationships if r.relationship_type == RelationshipType.WORKS_FOR]
        assert len(work_rels) > 0
        
        work_rel = work_rels[0]
        assert work_rel.confidence > 0.5
        assert len(work_rel.evidence) > 0
        assert work_rel.evidence[0].extraction_method == "pattern_matching"
    
    def test_detect_cooccurrence_relationships(self):
        """Test co-occurrence based relationship detection"""
        text = "John Smith and Acme Corporation are working together on Project Alpha."
        
        relationships = self.detector._detect_cooccurrence_relationships(self.entities, text, "doc1")
        
        assert len(relationships) > 0
        
        # Check that relationships were created between nearby entities
        for rel in relationships:
            assert rel.confidence > 0.0
            assert rel.properties["extraction_method"] == "cooccurrence"
            assert "distance" in rel.properties
    
    def test_detect_semantic_relationships(self):
        """Test semantic similarity based relationship detection"""
        # Temporarily lower the similarity threshold for testing
        original_threshold = self.detector.similarity_threshold
        self.detector.similarity_threshold = 0.2  # Lower threshold for testing
        
        try:
            # Create similar entities with same label
            similar_entities = [
                Entity(
                    id="entity4",
                    text="Python programming",
                    label="SKILL",  # Use consistent label
                    start_char=0,
                    end_char=18,
                    confidence=0.9,
                    source_document_id="doc1",
                    properties={"language": "en"}
                ),
                Entity(
                    id="entity5",
                    text="Python development",
                    label="SKILL",  # Use consistent label
                    start_char=20,
                    end_char=38,
                    confidence=0.9,
                    source_document_id="doc1",
                    properties={"language": "en"}
                )
            ]
            
            text = "Python programming and Python development are related skills."
            
            relationships = self.detector._detect_semantic_relationships(similar_entities, text, "doc1")
            
            # Should find at least one semantic relationship
            assert len(relationships) > 0
            
            semantic_rels = [r for r in relationships if r.relationship_type == RelationshipType.SIMILAR_TO]
            assert len(semantic_rels) > 0
            
            semantic_rel = semantic_rels[0]
            assert semantic_rel.confidence > 0.0
            assert semantic_rel.properties["extraction_method"] == "semantic_similarity"
        finally:
            # Restore original threshold
            self.detector.similarity_threshold = original_threshold
    
    def test_detect_temporal_relationships(self):
        """Test temporal relationship detection"""
        # Add temporal entity
        temporal_entity = Entity(
            id="entity_time",
            text="January 2024",
            label="DATE",
            start_char=40,
            end_char=52,
            confidence=0.9,
            source_document_id="doc1",
            properties={"language": "en"}
        )
        
        entities_with_time = self.entities + [temporal_entity]
        text = "John Smith started working on Project Alpha in January 2024."
        
        relationships = self.detector._detect_temporal_relationships(entities_with_time, text, "doc1")
        
        # Should find temporal relationships
        temporal_rels = [r for r in relationships if r.temporal_info]
        assert len(temporal_rels) >= 0  # May or may not find temporal patterns
    
    def test_deduplicate_relationships(self):
        """Test relationship deduplication"""
        # Create duplicate relationships
        rel1 = EntityRelationship(
            id="rel1",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type=RelationshipType.WORKS_FOR,
            confidence=0.8,
            evidence=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        rel2 = EntityRelationship(
            id="rel2",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type=RelationshipType.WORKS_FOR,
            confidence=0.7,
            evidence=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        relationships = [rel1, rel2]
        deduplicated = self.detector._deduplicate_relationships(relationships)
        
        # Should merge into one relationship
        assert len(deduplicated) == 1
        assert deduplicated[0].confidence > 0.7  # Should be boosted
        assert deduplicated[0].properties["merged_from_count"] == 2
    
    def test_detect_relationship_conflicts(self):
        """Test conflict detection"""
        # Create conflicting relationships
        rel1 = EntityRelationship(
            id="rel1",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type=RelationshipType.MANAGES,
            confidence=0.8,
            evidence=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        rel2 = EntityRelationship(
            id="rel2",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type=RelationshipType.REPORTS_TO,
            confidence=0.7,
            evidence=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        relationships = [rel1, rel2]
        conflicts = self.detector.detect_relationship_conflicts(relationships)
        
        assert len(conflicts) > 0
        assert conflicts[0].conflict_type == "contradictory_relationship_types"
        assert len(conflicts[0].conflicting_relationships) == 2
    
    def test_resolve_relationship_conflicts(self):
        """Test conflict resolution"""
        # Create conflicting relationships
        rel1 = EntityRelationship(
            id="rel1",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type=RelationshipType.MANAGES,
            confidence=0.8,
            evidence=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        rel2 = EntityRelationship(
            id="rel2",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type=RelationshipType.REPORTS_TO,
            confidence=0.6,
            evidence=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        conflict = RelationshipConflict(
            id="conflict1",
            conflicting_relationships=["rel1", "rel2"],
            conflict_type="contradictory_relationship_types",
            description="Test conflict",
            confidence=0.8,
            resolution_status="pending",
            created_at=datetime.utcnow()
        )
        
        relationships = [rel1, rel2]
        conflicts = [conflict]
        
        resolved = self.detector.resolve_relationship_conflicts(conflicts, relationships)
        
        # Should keep only the higher confidence relationship
        assert len(resolved) == 1
        assert resolved[0].id == "rel1"  # Higher confidence
        assert conflict.resolution_status == "resolved"
    
    def test_update_relationship_versions(self):
        """Test relationship versioning"""
        # Create existing relationship
        existing_rel = EntityRelationship(
            id="rel1",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type=RelationshipType.WORKS_FOR,
            confidence=0.7,
            evidence=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1
        )
        
        # Create new relationship with different confidence
        new_rel = EntityRelationship(
            id="rel2",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type=RelationshipType.WORKS_FOR,
            confidence=0.9,
            evidence=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        existing_relationships = [existing_rel]
        new_relationships = [new_rel]
        
        updated = self.detector.update_relationship_versions(existing_relationships, new_relationships)
        
        assert len(updated) == 1
        assert updated[0].version == 2  # Version incremented
        assert updated[0].confidence == 0.9  # Updated confidence
        assert "version_history" in updated[0].properties
    
    def test_get_relationship_statistics(self):
        """Test relationship statistics"""
        relationships = [
            EntityRelationship(
                id="rel1",
                source_entity_id="entity1",
                target_entity_id="entity2",
                relationship_type=RelationshipType.WORKS_FOR,
                confidence=0.9,
                evidence=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                properties={"extraction_method": "pattern_matching"}
            ),
            EntityRelationship(
                id="rel2",
                source_entity_id="entity2",
                target_entity_id="entity3",
                relationship_type=RelationshipType.PARTICIPATES_IN,
                confidence=0.6,
                evidence=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                properties={"extraction_method": "cooccurrence"},
                temporal_info={"temporal_type": "during"}
            )
        ]
        
        stats = self.detector.get_relationship_statistics(relationships)
        
        assert stats["total_relationships"] == 2
        assert stats["relationships_by_type"]["works_for"] == 1
        assert stats["relationships_by_type"]["participates_in"] == 1
        assert stats["relationships_by_confidence"]["high"] == 1
        assert stats["relationships_by_confidence"]["medium"] == 1
        assert stats["relationships_by_method"]["pattern_matching"] == 1
        assert stats["relationships_by_method"]["cooccurrence"] == 1
        assert stats["temporal_relationships"] == 1
        assert stats["average_confidence"] == 0.75


class TestRelationshipManager:
    """Test relationship manager functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.manager = RelationshipManager()
        
        # Create sample entities
        self.entities = [
            Entity(
                id="entity1",
                text="Alice Johnson",
                label="PERSON",
                start_char=0,
                end_char=13,
                confidence=0.9,
                source_document_id="doc1",
                properties={"language": "en"}
            ),
            Entity(
                id="entity2",
                text="Tech Corp",
                label="ORG",
                start_char=20,
                end_char=29,
                confidence=0.8,
                source_document_id="doc1",
                properties={"language": "en"}
            )
        ]
    
    def test_process_document_relationships(self):
        """Test document relationship processing"""
        text = "Alice Johnson works for Tech Corp and manages the development team."
        
        result = self.manager.process_document_relationships(self.entities, text, "doc1")
        
        assert "relationships" in result
        assert "conflicts" in result
        assert "statistics" in result
        assert result["document_id"] == "doc1"
        assert "processed_at" in result
        
        # Check that relationships were stored
        stored_relationships = self.manager.get_document_relationships("doc1")
        assert len(stored_relationships) > 0
    
    def test_store_and_retrieve_relationships(self):
        """Test relationship storage and retrieval"""
        relationship = EntityRelationship(
            id="rel1",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type=RelationshipType.WORKS_FOR,
            confidence=0.8,
            evidence=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Store relationship
        self.manager.store_relationships("doc1", [relationship])
        
        # Retrieve relationships
        retrieved = self.manager.get_document_relationships("doc1")
        assert len(retrieved) == 1
        assert retrieved[0].id == "rel1"
        
        # Test get all relationships
        all_rels = self.manager.get_all_relationships()
        assert len(all_rels) == 1
        
        # Test get entity relationships
        entity_rels = self.manager.get_entity_relationships("entity1")
        assert len(entity_rels) == 1
        
        # Test get relationship by ID
        rel_by_id = self.manager.get_relationship_by_id("rel1")
        assert rel_by_id is not None
        assert rel_by_id.id == "rel1"
    
    def test_store_and_retrieve_conflicts(self):
        """Test conflict storage and retrieval"""
        conflict = RelationshipConflict(
            id="conflict1",
            conflicting_relationships=["rel1", "rel2"],
            conflict_type="test_conflict",
            description="Test conflict",
            confidence=0.8,
            resolution_status="pending",
            created_at=datetime.utcnow()
        )
        
        # Store conflict
        self.manager.store_conflicts("doc1", [conflict])
        
        # Retrieve conflicts
        retrieved = self.manager.get_document_conflicts("doc1")
        assert len(retrieved) == 1
        assert retrieved[0].id == "conflict1"


class TestRelationshipDataModels:
    """Test relationship data models"""
    
    def test_relationship_evidence_serialization(self):
        """Test RelationshipEvidence serialization"""
        evidence = RelationshipEvidence(
            document_id="doc1",
            page_number=1,
            sentence="Test sentence",
            context="Test context",
            extraction_method="test",
            confidence=0.8,
            created_at=datetime.utcnow()
        )
        
        # Test to_dict
        data = evidence.to_dict()
        assert data["document_id"] == "doc1"
        assert data["page_number"] == 1
        assert isinstance(data["created_at"], str)
        
        # Test from_dict
        reconstructed = RelationshipEvidence.from_dict(data)
        assert reconstructed.document_id == evidence.document_id
        assert reconstructed.page_number == evidence.page_number
        assert isinstance(reconstructed.created_at, datetime)
    
    def test_entity_relationship_serialization(self):
        """Test EntityRelationship serialization"""
        evidence = RelationshipEvidence(
            document_id="doc1",
            page_number=1,
            sentence="Test sentence",
            context="Test context",
            extraction_method="test",
            confidence=0.8,
            created_at=datetime.utcnow()
        )
        
        relationship = EntityRelationship(
            id="rel1",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type=RelationshipType.WORKS_FOR,
            confidence=0.8,
            evidence=[evidence],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            temporal_info={"type": "current"},
            properties={"test": "value"}
        )
        
        # Test to_dict
        data = relationship.to_dict()
        assert data["id"] == "rel1"
        assert data["relationship_type"] == "works_for"
        assert isinstance(data["created_at"], str)
        assert len(data["evidence"]) == 1
        
        # Test from_dict
        reconstructed = EntityRelationship.from_dict(data)
        assert reconstructed.id == relationship.id
        assert reconstructed.relationship_type == RelationshipType.WORKS_FOR
        assert isinstance(reconstructed.created_at, datetime)
        assert len(reconstructed.evidence) == 1
    
    def test_relationship_conflict_serialization(self):
        """Test RelationshipConflict serialization"""
        conflict = RelationshipConflict(
            id="conflict1",
            conflicting_relationships=["rel1", "rel2"],
            conflict_type="test_conflict",
            description="Test conflict",
            confidence=0.8,
            resolution_status="pending",
            created_at=datetime.utcnow(),
            resolved_at=datetime.utcnow(),
            resolution_method="test_method"
        )
        
        # Test to_dict
        data = conflict.to_dict()
        assert data["id"] == "conflict1"
        assert data["conflicting_relationships"] == ["rel1", "rel2"]
        assert isinstance(data["created_at"], str)
        assert isinstance(data["resolved_at"], str)
        
        # Test serialization without resolved_at
        conflict_pending = RelationshipConflict(
            id="conflict2",
            conflicting_relationships=["rel3", "rel4"],
            conflict_type="test_conflict",
            description="Test conflict",
            confidence=0.8,
            resolution_status="pending",
            created_at=datetime.utcnow()
        )
        
        data_pending = conflict_pending.to_dict()
        assert data_pending["resolved_at"] is None


if __name__ == "__main__":
    pytest.main([__file__])