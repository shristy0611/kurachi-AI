"""
Demonstration of relationship detection and mapping functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from services.entity_extraction import EntityExtractor
from services.relationship_detection import RelationshipManager, RelationshipType


def demo_relationship_detection():
    """Demonstrate relationship detection capabilities"""
    
    # Initialize services
    entity_extractor = EntityExtractor()
    relationship_manager = RelationshipManager()
    
    # Sample business document text
    sample_texts = [
        {
            "id": "doc1",
            "text": """
            John Smith works for Acme Corporation as a Senior Software Engineer. 
            He leads the Project Alpha development team and collaborates with Sarah Johnson 
            from the Marketing Department. John has expertise in Python programming and 
            machine learning. The project started in January 2024 and is scheduled to 
            complete by December 2024.
            """
        },
        {
            "id": "doc2", 
            "text": """
            Sarah Johnson is the Marketing Manager at Acme Corporation. She manages 
            the product marketing team and works closely with the engineering department 
            on Project Alpha. Sarah has experience in digital marketing and customer 
            analytics. She reports to the VP of Marketing.
            """
        },
        {
            "id": "doc3",
            "text": """
            Project Alpha is a machine learning initiative at Acme Corporation. 
            The project uses Python programming and TensorFlow for model development. 
            It involves collaboration between engineering and marketing teams. 
            The project aims to improve customer analytics capabilities.
            """
        }
    ]
    
    print("=== Relationship Detection Demo ===\n")
    
    all_relationships = []
    all_entities = []
    
    # Process each document
    for doc in sample_texts:
        print(f"Processing Document: {doc['id']}")
        print(f"Text: {doc['text'][:100]}...")
        
        # Extract entities
        entities = entity_extractor.extract_entities(doc['text'], doc['id'])
        all_entities.extend(entities)
        
        print(f"Found {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity.text} ({entity.label}) - confidence: {entity.confidence:.2f}")
        
        # Process relationships
        result = relationship_manager.process_document_relationships(entities, doc['text'], doc['id'])
        relationships = result['relationships']
        conflicts = result['conflicts']
        stats = result['statistics']
        
        all_relationships.extend(relationships)
        
        print(f"\nFound {len(relationships)} relationships:")
        for rel in relationships:
            source_entity = next((e for e in entities if e.id == rel.source_entity_id), None)
            target_entity = next((e for e in entities if e.id == rel.target_entity_id), None)
            
            if source_entity and target_entity:
                print(f"  - {source_entity.text} --[{rel.relationship_type.value}]--> {target_entity.text}")
                print(f"    Confidence: {rel.confidence:.2f}, Method: {rel.properties.get('extraction_method', 'unknown')}")
        
        if conflicts:
            print(f"\nFound {len(conflicts)} conflicts:")
            for conflict in conflicts:
                print(f"  - {conflict.conflict_type}: {conflict.description}")
        
        print(f"\nStatistics: {json.dumps(stats, indent=2)}")
        print("-" * 80)
    
    # Cross-document relationship analysis
    print("\n=== Cross-Document Analysis ===")
    
    # Find entities that appear in multiple documents
    entity_texts = {}
    for entity in all_entities:
        text_key = entity.text.lower()
        if text_key not in entity_texts:
            entity_texts[text_key] = []
        entity_texts[text_key].append(entity)
    
    cross_doc_entities = {text: entities for text, entities in entity_texts.items() if len(entities) > 1}
    
    print(f"Found {len(cross_doc_entities)} entities appearing in multiple documents:")
    for text, entities in cross_doc_entities.items():
        doc_ids = [e.source_document_id for e in entities]
        print(f"  - '{text}' appears in documents: {', '.join(doc_ids)}")
    
    # Relationship type analysis
    print("\n=== Relationship Type Analysis ===")
    relationship_types = {}
    for rel in all_relationships:
        rel_type = rel.relationship_type.value
        if rel_type not in relationship_types:
            relationship_types[rel_type] = []
        relationship_types[rel_type].append(rel)
    
    for rel_type, rels in relationship_types.items():
        print(f"{rel_type}: {len(rels)} relationships")
        avg_confidence = sum(r.confidence for r in rels) / len(rels)
        print(f"  Average confidence: {avg_confidence:.2f}")
    
    # High-confidence relationships
    print("\n=== High-Confidence Relationships (>0.8) ===")
    high_conf_rels = [r for r in all_relationships if r.confidence > 0.8]
    
    for rel in high_conf_rels:
        source_entity = next((e for e in all_entities if e.id == rel.source_entity_id), None)
        target_entity = next((e for e in all_entities if e.id == rel.target_entity_id), None)
        
        if source_entity and target_entity:
            print(f"  - {source_entity.text} --[{rel.relationship_type.value}]--> {target_entity.text}")
            print(f"    Confidence: {rel.confidence:.2f}")
            print(f"    Evidence: {len(rel.evidence)} pieces")
            if rel.evidence:
                print(f"    Sample evidence: {rel.evidence[0].sentence[:100]}...")
    
    # Temporal relationships
    print("\n=== Temporal Relationships ===")
    temporal_rels = [r for r in all_relationships if r.temporal_info]
    
    if temporal_rels:
        for rel in temporal_rels:
            source_entity = next((e for e in all_entities if e.id == rel.source_entity_id), None)
            target_entity = next((e for e in all_entities if e.id == rel.target_entity_id), None)
            
            if source_entity and target_entity:
                print(f"  - {source_entity.text} --[{rel.relationship_type.value}]--> {target_entity.text}")
                print(f"    Temporal info: {rel.temporal_info}")
    else:
        print("  No temporal relationships detected")
    
    # Knowledge graph summary
    print("\n=== Knowledge Graph Summary ===")
    print(f"Total entities: {len(all_entities)}")
    print(f"Total relationships: {len(all_relationships)}")
    print(f"Unique entity types: {len(set(e.label for e in all_entities))}")
    print(f"Unique relationship types: {len(set(r.relationship_type.value for r in all_relationships))}")
    
    entity_types = {}
    for entity in all_entities:
        if entity.label not in entity_types:
            entity_types[entity.label] = 0
        entity_types[entity.label] += 1
    
    print("\nEntity distribution:")
    for entity_type, count in sorted(entity_types.items()):
        print(f"  {entity_type}: {count}")
    
    print("\nRelationship distribution:")
    for rel_type, count in sorted(relationship_types.items()):
        print(f"  {rel_type}: {len(count)}")


def demo_japanese_relationships():
    """Demonstrate Japanese relationship detection"""
    
    print("\n=== Japanese Relationship Detection Demo ===\n")
    
    entity_extractor = EntityExtractor()
    relationship_manager = RelationshipManager()
    
    # Japanese business document
    japanese_text = """
    田中太郎は株式会社アクメで上級ソフトウェアエンジニアとして働いています。
    彼はプロジェクトアルファの開発チームをリードし、マーケティング部の佐藤花子と
    協力しています。田中さんはPythonプログラミングと機械学習の専門知識を持っています。
    このプロジェクトは2024年1月に開始され、2024年12月に完了予定です。
    """
    
    print("Processing Japanese document:")
    print(f"Text: {japanese_text[:100]}...")
    
    # Extract entities
    entities = entity_extractor.extract_entities(japanese_text, "doc_jp")
    
    print(f"\nFound {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity.text} ({entity.label}) - confidence: {entity.confidence:.2f}")
    
    # Process relationships
    result = relationship_manager.process_document_relationships(entities, japanese_text, "doc_jp")
    relationships = result['relationships']
    
    print(f"\nFound {len(relationships)} relationships:")
    for rel in relationships:
        source_entity = next((e for e in entities if e.id == rel.source_entity_id), None)
        target_entity = next((e for e in entities if e.id == rel.target_entity_id), None)
        
        if source_entity and target_entity:
            print(f"  - {source_entity.text} --[{rel.relationship_type.value}]--> {target_entity.text}")
            print(f"    Confidence: {rel.confidence:.2f}")


if __name__ == "__main__":
    # Run English demo
    demo_relationship_detection()
    
    # Run Japanese demo
    demo_japanese_relationships()
    
    print("\n=== Demo Complete ===")