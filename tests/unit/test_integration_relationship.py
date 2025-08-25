"""
Simple integration test for relationship detection
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.entity_extraction import EntityExtractor
from services.relationship_detection import RelationshipManager


def test_integration():
    """Test integration between entity extraction and relationship detection"""
    
    # Initialize services
    entity_extractor = EntityExtractor()
    relationship_manager = RelationshipManager()
    
    # Sample text
    text = "John Smith works for Acme Corporation and leads Project Alpha."
    document_id = "test_doc"
    
    print("Testing relationship detection integration...")
    print(f"Text: {text}")
    
    # Extract entities
    entities = entity_extractor.extract_entities(text, document_id)
    print(f"\nExtracted {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity.text} ({entity.label}) - confidence: {entity.confidence:.2f}")
    
    # Process relationships
    result = relationship_manager.process_document_relationships(entities, text, document_id)
    relationships = result['relationships']
    conflicts = result['conflicts']
    stats = result['statistics']
    
    print(f"\nDetected {len(relationships)} relationships:")
    for rel in relationships:
        source_entity = next((e for e in entities if e.id == rel.source_entity_id), None)
        target_entity = next((e for e in entities if e.id == rel.target_entity_id), None)
        
        if source_entity and target_entity:
            print(f"  - {source_entity.text} --[{rel.relationship_type.value}]--> {target_entity.text}")
            print(f"    Confidence: {rel.confidence:.2f}")
            print(f"    Method: {rel.properties.get('extraction_method', 'unknown')}")
    
    print(f"\nConflicts: {len(conflicts)}")
    print(f"Statistics: {stats}")
    
    print("\nIntegration test completed successfully!")
    return True


if __name__ == "__main__":
    test_integration()