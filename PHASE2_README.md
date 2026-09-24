# Phase 2: Query Orchestrator - Quick Start Guide

## Overview

Phase 2 provides a sophisticated Query Orchestrator that maps Natural Language queries to your SQL knowledge graph using Gemini AI and Milvus vector search.

---

## 🚀 Quick Start

### 1. Initialize Database and Ingest into Milvus

```bash
source ~/anaconda3/bin/activate ml-env
python init_and_ingest.py
```

**This will:**
- ✓ Ensure SQL tables are created from `sql/schema.sql`
- ✓ Verify entities exist in the database
- ✓ Generate embeddings for all entities
- ✓ Store embeddings in Milvus vector database

**Output:**
```
✓ INITIALIZATION AND INGESTION COMPLETE
Database ready at: knowledge_graph.db
Milvus database at: ./outputs/milvus_orchestrator.db
```

---

### 2. Test the Orchestrator

```bash
source ~/anaconda3/bin/activate ml-env
python test_orchestrator.py
```

This runs sample queries to verify everything works correctly.

---

## 📚 Usage

### Python API

```python
from src.orchestrator import QueryOrchestrator

# Initialize
orchestrator = QueryOrchestrator(
    db_path="knowledge_graph.db",
    milvus_path="./outputs/milvus_orchestrator.db"
)

# Process a query
result = orchestrator.process_query(
    user_query="Show me all people working at Acme Corp",
    user_tags=["HR", "FINANCE"],  # User's access tags
    max_depth=2  # Graph traversal depth
)

# Use the results
if result['status'] == 'success':
    print(f"Found {len(result['mini_graph'])} relationships")

    # Access entities
    for candidate in result['candidates']:
        print(f"{candidate['canonical_name']} ({candidate['entity_type']})")

    # Access node details with attributes
    for node_id, node in result['nodes'].items():
        print(f"{node['name']} ({node['type']})")
        # Access entity attributes like CIN, PAN, etc.
        for key, value in node['attributes'].items():
            print(f"  {key}: {value}")

    # Access graph
    for edge in result['mini_graph']:
        print(f"{edge['source_name']} --[{edge['relation']}]--> {edge['target_name']}")

    # Access citations
    for doc, pages in result['citation_context'].items():
        print(f"Source: {doc}, Pages: {pages}")

# Cleanup
orchestrator.close()
```

---

## 🏗️ Architecture

### 4-Step Pipeline

#### **Step 1: Schema-Aware Intent Extraction**
- Uses Gemini AI to analyze the query
- Dynamically fetches available entity_types and relationship_types
- Extracts:
  - Named entities (e.g., "John Smith", "Acme Corp")
  - Entity types (e.g., "Person", "Company")
  - Relationship types (e.g., "WORKS_FOR")

#### **Step 2: Candidate Resolution**
- **Named Entities:** Milvus vector search (cosine similarity >= `similarity_threshold`, default 0.1; results are ranked)
- **Entity Types:** SQL query for all entities of that type
- Returns consolidated list of candidate IDs

#### **Step 3: Security Filtering**
- Applies ACL enforcement via `documents.access_tags`
- Filters candidates against user's access tags
- Only returns entities in accessible documents

#### **Step 4: Execution**
- Calls `SecureGraphTraverser.get_context()` (Walker)
- Returns:
  - `nodes`: Full node details with attributes (CIN, PAN, etc.)
  - `mini_graph`: List of relationship edges
  - `citation_context`: Document → page numbers mapping

---

## 📊 Response Format

```json
{
  "status": "success|ambiguous|no_access|error",
  "intent": {
    "entities": ["John Smith"],
    "entity_types": ["Person"],
    "relationship_types": ["WORKS_FOR"]
  },
  "candidates": [
    {
      "entity_id": "md5_hash",
      "entity_type": "Person",
      "canonical_name": "John Smith"
    }
  ],
  "nodes": {
    "entity_001": {
      "name": "John Smith",
      "type": "Person",
      "attributes": {
        "email": "john@example.com",
        "role": "Manager",
        "pan": "ABCDE1234F"
      },
      "name_variants": ["J. Smith", "John A. Smith"],
      "cluster_size": 3
    },
    "entity_002": {
      "name": "Acme Corp",
      "type": "Company",
      "attributes": {
        "cin": "U00000XX0000PTC000000",
        "status": "Active",
        "registered_office": "Mumbai, India"
      },
      "name_variants": ["ACME Corporation"],
      "cluster_size": 1
    }
  },
  "mini_graph": [
    {
      "source": "entity_001",
      "source_name": "John Smith",
      "target": "entity_002",
      "target_name": "Acme Corp",
      "relation": "WORKS_FOR",
      "properties": {...}
    }
  ],
  "citation_context": {
    "annual_report_2023.pdf": [1, 5, 10]
  },
  "message": "Found 5 entities, 2 nodes, and 12 relationships."
}
```

---

## 🔧 Configuration

### Adjust Search Parameters

```python
result = orchestrator.process_query(
    user_query="...",
    user_tags=["UNCLASSIFIED"],
    max_depth=3,  # Increase for deeper graph traversal
    similarity_threshold=0.2,  # Cosine similarity; higher = stricter matching
    max_candidates_per_entity=10,  # More candidates per entity
    max_type_results=50  # More entities per type
)
```

---

## 🗂️ Files Created

| File | Purpose |
|------|---------|
| `src/orchestrator.py` | Main QueryOrchestrator class |
| `src/milvus_ingestion.py` | Milvus ingestion engine |
| `init_and_ingest.py` | Database initialization script |
| `test_orchestrator.py` | Test suite |

---

## 🔍 Example Queries

```python
# Find specific people
"Show me John Smith"

# Find by entity type
"List all companies"
"Find all projects"

# Complex queries
"Who works for Acme Corporation?"
"Show me directors of tech companies"
"Find all auditors working on Project Alpha"
```

---

## ⚠️ Troubleshooting

### No entities found
```bash
# Re-run ingestion
python init_and_ingest.py
```

### Milvus errors
```bash
# Clear Milvus database and re-ingest
rm -rf outputs/milvus_orchestrator.db
python -m src.milvus_ingestion
```

### Empty results
- Check user has correct access tags
- Verify entities exist in SQL database
- Lower `similarity_threshold` for broader matches

---

## 📞 Next Steps

1. **Integrate with API:** Build REST endpoints around `QueryOrchestrator`
2. **Add UI:** Create web interface for natural language queries
3. **Expand Schema:** Add more entity types and relationship types
4. **Fine-tune:** Adjust similarity thresholds based on your data

---

**Phase 2 Complete!** 🎉

The Query Orchestrator is now ready to process natural language queries with ACL-enforced graph traversal and citation tracking.
