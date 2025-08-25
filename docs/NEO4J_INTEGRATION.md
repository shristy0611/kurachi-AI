# Neo4j Knowledge Graph Integration

This document describes the Neo4j integration for Kurachi AI's knowledge graph functionality.

## Overview

Kurachi AI uses Neo4j as the persistent storage backend for its knowledge graph, providing:

- **Persistent Graph Storage**: Entities and relationships are stored in Neo4j for durability
- **Advanced Graph Queries**: Leverage Cypher query language for complex graph operations
- **Graph Algorithms**: Use Neo4j's Graph Data Science library for community detection and centrality analysis
- **Scalable Performance**: Handle large knowledge graphs with optimized graph database performance
- **Visualization Support**: Generate graph visualizations for exploration and analysis

## Architecture

### Hybrid Approach

Kurachi AI uses a hybrid approach combining:

1. **Neo4j**: Persistent storage, complex queries, graph algorithms
2. **NetworkX**: In-memory operations, fallback when Neo4j unavailable
3. **ChromaDB**: Vector embeddings for semantic search

### Components

- `services/neo4j_service.py`: Core Neo4j database operations
- `services/graph_visualization.py`: Graph visualization and exploration APIs
- `services/knowledge_graph.py`: Enhanced with Neo4j integration
- `tests/test_neo4j_integration.py`: Comprehensive test suite

## Setup

### 1. Start Neo4j Database

Using Docker Compose:

```bash
# Start Neo4j
docker-compose -f docker-compose.neo4j.yml up -d

# Check status
docker-compose -f docker-compose.neo4j.yml ps
```

### 2. Configure Environment

Update your `.env` file:

```bash
# Neo4j Knowledge Graph Database
KURACHI_NEO4J_URI=bolt://localhost:7687
KURACHI_NEO4J_USERNAME=neo4j
KURACHI_NEO4J_PASSWORD=password
KURACHI_NEO4J_DATABASE=kurachi
```

### 3. Initialize Database

Run the setup script:

```bash
python scripts/setup_neo4j.py
```

This will:
- Wait for Neo4j to be available
- Create database schema (constraints and indexes)
- Create sample data for testing
- Verify the setup

### 4. Access Neo4j Browser

Open http://localhost:7474 in your browser:
- Username: `neo4j`
- Password: `password`
- Database: `kurachi`

## Usage

### Basic Operations

```python
from services.neo4j_service import neo4j_service
from services.entity_extraction import Entity

# Create entity
entity = Entity(
    id="example_entity",
    text="Example Company",
    label="COMPANY_EN",
    confidence=0.9,
    start_char=0,
    end_char=15,
    source_document_id="doc_1"
)

# Store in Neo4j
neo4j_service.create_entity_node(entity, "doc_1")

# Create relationship
neo4j_service.create_relationship(
    source_entity_id="entity_1",
    target_entity_id="entity_2", 
    relationship_type="WORKS_AT",
    confidence=0.8,
    evidence={"method": "pattern_matching"},
    document_id="doc_1"
)

# Search entities
results = neo4j_service.search_entities("Company", limit=10)

# Get neighbors
neighbors = neo4j_service.get_entity_neighbors("entity_1", max_distance=2)
```

### Graph Traversal

```python
# Find shortest path
path = neo4j_service.find_shortest_path("entity_1", "entity_2", max_length=6)

# Find common neighbors
common = neo4j_service.find_common_neighbors("entity_1", "entity_2")

# Get most connected entities
top_entities = neo4j_service.get_most_connected_entities(limit=20)

# Detect communities
communities = neo4j_service.detect_communities(min_community_size=3)
```

### Visualization

```python
from services.graph_visualization import graph_visualization_service

# Entity neighborhood visualization
viz = graph_visualization_service.create_entity_neighborhood_visualization(
    entity_id="entity_1",
    max_distance=2
)

# Path visualization
path_viz = graph_visualization_service.create_path_visualization(
    source_id="entity_1",
    target_id="entity_2"
)

# Community visualization
community_viz = graph_visualization_service.create_community_visualization(
    community_id=1,
    include_connections=True
)

# Full graph overview
overview = graph_visualization_service.create_full_graph_overview(max_nodes=100)
```

## Schema

### Node Types

#### Entity
- **Labels**: `Entity`
- **Properties**:
  - `id` (string, unique): Entity identifier
  - `text` (string): Entity text content
  - `label` (string): Entity type (PERSON, COMPANY_JP, etc.)
  - `confidence` (float): Extraction confidence score
  - `start_char` (int): Start position in source text
  - `end_char` (int): End position in source text
  - `context` (string): Surrounding context
  - `created_at` (datetime): Creation timestamp
  - `updated_at` (datetime): Last update timestamp

#### Document
- **Labels**: `Document`
- **Properties**:
  - `id` (string, unique): Document identifier
  - `filename` (string): Original filename
  - `created_at` (datetime): Creation timestamp

### Relationship Types

#### RELATIONSHIP
- **Properties**:
  - `id` (string, unique): Relationship identifier
  - `type` (string): Relationship type (WORKS_AT, MANAGES, etc.)
  - `confidence` (float): Relationship confidence score
  - `evidence` (string): JSON evidence data
  - `document_id` (string): Source document
  - `created_at` (datetime): Creation timestamp

#### EXTRACTED_FROM
- Connects entities to their source documents
- No additional properties

## Cypher Query Examples

### Basic Queries

```cypher
-- Find all entities
MATCH (e:Entity) RETURN e LIMIT 10

-- Find entities by type
MATCH (e:Entity {label: 'PERSON_JP'}) RETURN e

-- Find relationships
MATCH (e1:Entity)-[r:RELATIONSHIP]->(e2:Entity) 
RETURN e1.text, r.type, e2.text LIMIT 10

-- Search entities by text
MATCH (e:Entity) 
WHERE toLower(e.text) CONTAINS '田中'
RETURN e
```

### Advanced Queries

```cypher
-- Find shortest path between entities
MATCH path = shortestPath((a:Entity {id: 'entity_1'})-[*1..6]-(b:Entity {id: 'entity_2'}))
RETURN path

-- Find most connected entities
MATCH (e:Entity)
OPTIONAL MATCH (e)-[r:RELATIONSHIP]-()
WITH e, count(r) as degree
RETURN e, degree
ORDER BY degree DESC
LIMIT 10

-- Find entities in same document
MATCH (e1:Entity)-[:EXTRACTED_FROM]->(d:Document)<-[:EXTRACTED_FROM]-(e2:Entity)
WHERE e1 <> e2
RETURN e1, e2, d
LIMIT 20

-- Find relationship patterns
MATCH (person:Entity {label: 'PERSON_JP'})-[r1:RELATIONSHIP {type: 'WORKS_AT'}]->(company:Entity {label: 'COMPANY_JP'})
MATCH (person)-[r2:RELATIONSHIP {type: 'MANAGES'}]->(project:Entity {label: 'PROJECT_JP'})
RETURN person.text, company.text, project.text
```

## Performance Optimization

### Indexes

The setup script creates these indexes:
- `entity_text_index`: For text search
- `entity_label_index`: For filtering by entity type
- `entity_confidence_index`: For confidence-based queries
- `relationship_type_index`: For relationship type filtering

### Query Optimization

1. **Use LIMIT**: Always limit result sets for large graphs
2. **Index Hints**: Use index hints for complex queries
3. **Profile Queries**: Use `PROFILE` to analyze query performance
4. **Batch Operations**: Use batch processing for large data imports

### Memory Configuration

Adjust Neo4j memory settings in `docker-compose.neo4j.yml`:

```yaml
environment:
  NEO4J_dbms_memory_heap_initial__size: 512m
  NEO4J_dbms_memory_heap_max__size: 1G
  NEO4J_dbms_memory_pagecache_size: 512m
```

## Testing

Run the test suite:

```bash
# Run all Neo4j tests
python -m pytest tests/test_neo4j_integration.py -v

# Run specific test
python -m pytest tests/test_neo4j_integration.py::TestNeo4jService::test_connection -v
```

## Troubleshooting

### Connection Issues

1. **Check Neo4j Status**:
   ```bash
   docker-compose -f docker-compose.neo4j.yml ps
   ```

2. **Check Logs**:
   ```bash
   docker-compose -f docker-compose.neo4j.yml logs neo4j
   ```

3. **Verify Configuration**:
   - Check `.env` file settings
   - Verify Neo4j is running on correct ports
   - Test connection with Neo4j Browser

### Performance Issues

1. **Monitor Query Performance**:
   ```cypher
   CALL dbms.listQueries() YIELD query, elapsedTimeMillis
   WHERE elapsedTimeMillis > 1000
   RETURN query, elapsedTimeMillis
   ```

2. **Check Index Usage**:
   ```cypher
   CALL db.indexes() YIELD name, state, populationPercent
   RETURN name, state, populationPercent
   ```

3. **Analyze Query Plans**:
   ```cypher
   PROFILE MATCH (e:Entity) WHERE e.text CONTAINS 'search_term' RETURN e
   ```

### Data Issues

1. **Check Data Integrity**:
   ```cypher
   -- Find orphaned relationships
   MATCH ()-[r:RELATIONSHIP]-()
   WHERE NOT EXISTS((r)-[:EXTRACTED_FROM]->(:Document))
   RETURN count(r)
   ```

2. **Clean Up Test Data**:
   ```cypher
   -- Remove test entities
   MATCH (e:Entity)-[:EXTRACTED_FROM]->(d:Document {id: 'test_doc'})
   DETACH DELETE e, d
   ```

## Migration

### From NetworkX Only

If upgrading from NetworkX-only implementation:

1. Start Neo4j database
2. Run setup script
3. Re-process existing documents to populate Neo4j
4. Verify data consistency

### Backup and Restore

```bash
# Backup
docker exec kurachi-neo4j neo4j-admin database dump kurachi

# Restore
docker exec kurachi-neo4j neo4j-admin database load kurachi
```

## Future Enhancements

- **Graph ML**: Integrate Neo4j's Graph Data Science library
- **Real-time Updates**: Implement change data capture
- **Multi-tenancy**: Support multiple knowledge graphs
- **Federation**: Connect multiple Neo4j instances
- **Advanced Visualization**: 3D graph visualization