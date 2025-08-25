#!/usr/bin/env python3
"""
Neo4j Setup Script for Kurachi AI
Sets up Neo4j database with proper schema, constraints, and indexes
"""
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.neo4j_service import neo4j_service
from utils.logger import get_logger

logger = get_logger("neo4j_setup")


def wait_for_neo4j(max_attempts: int = 30, delay: int = 2):
    """Wait for Neo4j to be available"""
    print("Waiting for Neo4j to be available...")
    
    for attempt in range(max_attempts):
        if neo4j_service.is_connected():
            print("✅ Neo4j is available!")
            return True
        
        print(f"⏳ Attempt {attempt + 1}/{max_attempts} - Neo4j not ready, waiting {delay}s...")
        time.sleep(delay)
    
    print("❌ Neo4j is not available after maximum attempts")
    return False


def setup_database_schema():
    """Set up Neo4j database schema"""
    print("Setting up Neo4j database schema...")
    
    try:
        with neo4j_service.driver.session(database=neo4j_service.driver.default_database) as session:
            # Create kurachi database if it doesn't exist
            try:
                session.run("CREATE DATABASE kurachi IF NOT EXISTS")
                print("✅ Database 'kurachi' created/verified")
            except Exception as e:
                print(f"⚠️  Database creation info: {e}")
        
        # Switch to kurachi database for schema setup
        with neo4j_service.driver.session(database="kurachi") as session:
            
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT relationship_id_unique IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() REQUIRE r.id IS UNIQUE"
            ]
            
            print("Creating constraints...")
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"✅ {constraint.split()[2]} constraint created")
                except Exception as e:
                    print(f"⚠️  Constraint already exists or error: {e}")
            
            # Create indexes
            indexes = [
                "CREATE INDEX entity_text_index IF NOT EXISTS FOR (e:Entity) ON (e.text)",
                "CREATE INDEX entity_label_index IF NOT EXISTS FOR (e:Entity) ON (e.label)",
                "CREATE INDEX entity_confidence_index IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
                "CREATE INDEX document_filename_index IF NOT EXISTS FOR (d:Document) ON (d.filename)",
                "CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.type)",
                "CREATE INDEX relationship_confidence_index IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.confidence)"
            ]
            
            print("Creating indexes...")
            for index in indexes:
                try:
                    session.run(index)
                    print(f"✅ {index.split()[2]} index created")
                except Exception as e:
                    print(f"⚠️  Index already exists or error: {e}")
            
            print("✅ Schema setup completed successfully!")
            
    except Exception as e:
        print(f"❌ Schema setup failed: {e}")
        return False
    
    return True


def create_sample_data():
    """Create sample data for testing"""
    print("Creating sample data...")
    
    try:
        with neo4j_service.driver.session(database="kurachi") as session:
            # Sample entities
            sample_entities = [
                {
                    "id": "sample_person_1",
                    "text": "田中太郎",
                    "label": "PERSON_JP",
                    "confidence": 0.95,
                    "context": "田中太郎は営業部の部長です"
                },
                {
                    "id": "sample_company_1", 
                    "text": "株式会社サンプル",
                    "label": "COMPANY_JP",
                    "confidence": 0.9,
                    "context": "株式会社サンプルは東京に本社があります"
                },
                {
                    "id": "sample_project_1",
                    "text": "AIプロジェクト",
                    "label": "PROJECT_JP", 
                    "confidence": 0.85,
                    "context": "AIプロジェクトは来年開始予定です"
                }
            ]
            
            # Create sample entities
            for entity in sample_entities:
                query = """
                MERGE (e:Entity {id: $id})
                SET e.text = $text,
                    e.label = $label,
                    e.confidence = $confidence,
                    e.context = $context,
                    e.created_at = datetime(),
                    e.updated_at = datetime()
                WITH e
                MERGE (d:Document {id: 'sample_document'})
                SET d.filename = 'sample.txt',
                    d.created_at = datetime()
                MERGE (e)-[:EXTRACTED_FROM]->(d)
                """
                
                session.run(query, entity)
                print(f"✅ Created entity: {entity['text']}")
            
            # Create sample relationships
            relationships = [
                {
                    "source": "sample_person_1",
                    "target": "sample_company_1", 
                    "type": "WORKS_AT",
                    "confidence": 0.9
                },
                {
                    "source": "sample_person_1",
                    "target": "sample_project_1",
                    "type": "MANAGES", 
                    "confidence": 0.85
                }
            ]
            
            for rel in relationships:
                query = """
                MATCH (source:Entity {id: $source})
                MATCH (target:Entity {id: $target})
                MERGE (source)-[r:RELATIONSHIP {
                    id: randomUUID(),
                    type: $type,
                    confidence: $confidence,
                    created_at: datetime()
                }]->(target)
                """
                
                session.run(query, rel)
                print(f"✅ Created relationship: {rel['type']}")
            
            print("✅ Sample data created successfully!")
            
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False
    
    return True


def verify_setup():
    """Verify the setup is working correctly"""
    print("Verifying setup...")
    
    try:
        with neo4j_service.driver.session(database="kurachi") as session:
            # Count entities
            result = session.run("MATCH (e:Entity) RETURN count(e) as entity_count")
            entity_count = result.single()["entity_count"]
            print(f"✅ Found {entity_count} entities")
            
            # Count relationships
            result = session.run("MATCH ()-[r:RELATIONSHIP]-() RETURN count(r) as rel_count")
            rel_count = result.single()["rel_count"]
            print(f"✅ Found {rel_count} relationships")
            
            # Test search
            result = session.run("""
                MATCH (e:Entity) 
                WHERE toLower(e.text) CONTAINS '田中'
                RETURN e.text as text
                LIMIT 1
            """)
            
            record = result.single()
            if record:
                print(f"✅ Search test passed: Found '{record['text']}'")
            else:
                print("⚠️  Search test: No results found")
            
            print("✅ Setup verification completed!")
            
    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        return False
    
    return True


def main():
    """Main setup function"""
    print("🚀 Starting Neo4j setup for Kurachi AI...")
    print("=" * 50)
    
    # Wait for Neo4j to be available
    if not wait_for_neo4j():
        print("❌ Setup failed: Neo4j not available")
        sys.exit(1)
    
    # Setup database schema
    if not setup_database_schema():
        print("❌ Setup failed: Schema setup error")
        sys.exit(1)
    
    # Create sample data
    if not create_sample_data():
        print("❌ Setup failed: Sample data creation error")
        sys.exit(1)
    
    # Verify setup
    if not verify_setup():
        print("❌ Setup failed: Verification error")
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 Neo4j setup completed successfully!")
    print()
    print("Next steps:")
    print("1. Access Neo4j Browser at: http://localhost:7474")
    print("2. Login with username: neo4j, password: password")
    print("3. Switch to 'kurachi' database")
    print("4. Run test queries to explore the knowledge graph")
    print()
    print("Example queries:")
    print("- MATCH (e:Entity) RETURN e LIMIT 10")
    print("- MATCH (e:Entity)-[r:RELATIONSHIP]->(t:Entity) RETURN e, r, t LIMIT 5")
    print("- CALL db.schema.visualization()")


if __name__ == "__main__":
    main()