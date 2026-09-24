# Bakasur

**GraphRAG++ Knowledge Graph System with Multi-Modal Entity Extraction**

A sophisticated knowledge graph construction and query system that combines multimodal document processing, advanced entity deduplication, hierarchical community detection, and secure graph traversal with access control.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## 📚 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Features](#core-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Phase 1: Entity Extraction & Knowledge Graph](#phase-1-entity-extraction--knowledge-graph)
- [Phase 2: Query Orchestrator](#phase-2-query-orchestrator)
- [Phase 3A: Leiden Community Detection](#phase-3a-leiden-community-detection)
- [Document Processing](#document-processing)
- [Configuration Reference](#configuration-reference)
- [Database Schema](#database-schema)
- [API Reference](#api-reference)
- [Complete Workflows](#complete-workflows)
- [CLI Reference](#cli-reference)
- [Examples](#examples)
- [Testing](#testing)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)
- [Project Structure](#project-structure)

---

## Overview

**Bakasur** is a production-ready knowledge graph system designed for extracting, deduplicating, and querying entities and relationships from multi-format documents. It leverages Google's Gemini 2.5 Flash for multimodal understanding and implements a GraphRAG++ architecture with hierarchical community detection.

### Key Capabilities

🔍 **Multi-Format Document Processing**: PDF, Images (JPG/PNG/TIFF), Word (DOCX), Excel (XLSX)
🤖 **Multimodal Entity Extraction**: Combines Gemini Vision + RapidOCR for superior accuracy
🔄 **5 Deduplication Methodologies**: R-Swoosh, Probabilistic (Splink), Topological, Semantic, LLM Full Context
📖 **Provenance Tracking**: Document-level and page-level citation context
🌐 **Hierarchical Community Detection**: Leiden algorithm with auto-tuned resolution parameters
🔒 **Secure Query System**: ACL-enforced graph traversal with vector search (Milvus)
💾 **SQL-Based Storage**: Optimized 5-table schema with junction tables for provenance

### Use Cases

- **Financial Document Analysis**: Extract entities from annual reports, filings, contracts
- **Corporate Intelligence**: Map relationships between companies, directors, and projects
- **Research Literature Mining**: Build knowledge graphs from academic papers
- **Legal Document Processing**: Track entities and relationships across case files
- **Multi-Source Entity Resolution**: Deduplicate entities mentioned across multiple documents

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BAKASUR SYSTEM PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: ENTITY EXTRACTION & KNOWLEDGE GRAPH CONSTRUCTION                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Documents (PDF, DOCX, Images, XLSX)                                        │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────┐                                          │
│  │  Universal Document Parser   │  RapidOCR + PyMuPDF + python-docx        │
│  │  (RapidOCRParser)            │  PIL for Excel rendering                 │
│  └──────────┬───────────────────┘                                          │
│             │                                                                │
│             │ (page_image, ocr_text, metadata) tuples                       │
│             ▼                                                                │
│  ┌──────────────────────────────┐                                          │
│  │  Multimodal Entity Extractor │  Gemini 2.5 Flash                        │
│  │  (DocLens Interleaved)       │  Text PRIMARY + Image SECONDARY          │
│  └──────────┬───────────────────┘                                          │
│             │                                                                │
│             │ {entities: [...], relationships: [...]}                       │
│             ▼                                                                │
│  ┌──────────────────────────────┐                                          │
│  │  Knowledge Graph Builder     │  NetworkX graph                          │
│  │  (All entities as nodes)     │  Duplicates kept intentionally           │
│  └──────────┬───────────────────┘                                          │
│             │                                                                │
│             │ Raw Knowledge Graph (non-deduplicated)                        │
│             ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │         5 DEDUPLICATION METHODOLOGIES (Parallel Comparison)       │      │
│  ├───────────────────────────────────────────────────────────────────┤      │
│  │ 1. R-Swoosh (Rule-based)           4. Semantic (Embeddings)      │      │
│  │ 2. Probabilistic (Splink)          5. LLM Full Context (Gemini)  │      │
│  │ 3. Topological (Graph-based)                                      │      │
│  └──────────┬────────────────────────────────────────────────────────┘      │
│             │                                                                │
│             │ Deduplicated Knowledge Graphs (one per method)                │
│             ▼                                                                │
│  ┌──────────────────────────────┐                                          │
│  │  Best Methodology Selected   │  Based on metrics & use case             │
│  │  Convert to SQL              │  5-table normalized schema                │
│  └──────────┬───────────────────┘                                          │
│             │                                                                │
└─────────────┼────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: QUERY ORCHESTRATOR & VECTOR SEARCH                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐          ┌────────────────────┐                     │
│  │   SQLite Database  │          │   Milvus Vector DB │                     │
│  │                    │          │                    │                     │
│  │  • documents       │          │  • Entity embeddings│                    │
│  │  • entities        │◄─────────┤  • Semantic search │                     │
│  │  • relationships   │          │  • Top-K retrieval │                     │
│  │  • occurrences     │          │                    │                     │
│  │  • ACL tags        │          └────────────────────┘                     │
│  └────────┬───────────┘                                                      │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────────────────┐                                          │
│  │  Query Orchestrator          │  1. Semantic search (Milvus)             │
│  │                              │  2. LLM query planning (Gemini)          │
│  │  Natural Language → Graph    │  3. ACL enforcement                      │
│  └──────────┬───────────────────┘  4. Secure graph traversal               │
│             │                                                                │
│             ▼                                                                │
│  ┌──────────────────────────────┐                                          │
│  │  Secure Graph Traverser      │  ACL-filtered graph walks                │
│  │  (SecureGraphTraverser)      │  Citation tracking                       │
│  └──────────┬───────────────────┘  Bidirectional traversal                 │
│             │                                                                │
│             │ {nodes, edges, sources, citations}                            │
│             ▼                                                                │
│  ┌──────────────────────────────┐                                          │
│  │  Mini-Graph Result           │  Subgraph with provenance                │
│  │  + Citation Context          │  Ready for downstream analysis           │
│  └──────────────────────────────┘                                          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3A: HIERARCHICAL COMMUNITY DETECTION                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Knowledge Graph (from SQL)                                                 │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────┐                                          │
│  │  Graph Metrics Calculator    │  Density, avg degree, clustering         │
│  │  Auto-tune γ parameters      │  → {micro: 2.0, meso: 1.0, macro: 0.5}   │
│  └──────────┬───────────────────┘                                          │
│             │                                                                │
│             ▼                                                                │
│  ┌──────────────────────────────┐                                          │
│  │  Leiden Algorithm (3 levels) │  Micro: teams (~5-10 entities)           │
│  │  python-igraph + leidenalg   │  Meso: departments (~50-100)             │
│  └──────────┬───────────────────┘  Macro: divisions (~1000+)               │
│             │                                                                │
│             │ {entity_id → community_id} per level                          │
│             ▼                                                                │
│  ┌──────────────────────────────┐                                          │
│  │  Community Summarizer        │  Gemini-powered natural language         │
│  │  (Gemini 2.5 Flash)          │  summaries for each community            │
│  └──────────┬───────────────────┘                                          │
│             │                                                                │
│             ▼                                                                │
│  ┌──────────────────────────────┐                                          │
│  │  SQL: leiden_communities     │  Stored in database for                  │
│  │  SQL: community_summaries    │  query orchestrator use                  │
│  └──────────────────────────────┘                                          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Google Gemini 2.5 Flash, Gemini 2.0 Flash Lite |
| **OCR** | RapidOCR (ONNX Runtime) |
| **Document Parsing** | PyMuPDF, python-docx, openpyxl, PIL |
| **Vector Database** | Milvus (lite mode) |
| **Graph Processing** | NetworkX, python-igraph, leidenalg |
| **Storage** | SQLite (normalized 5-table schema) |
| **Embeddings** | sentence-transformers (all-mpnet-base-v2) |
| **Entity Resolution** | Splink, fuzzy matching (rapidfuzz) |
| **Language** | Python 3.8+ |

---

## Core Features

### 1. Universal Document Processing

- **Supported Formats**: PDF (text/scanned), JPG, PNG, TIFF, DOCX, XLSX
- **OCR Engine**: RapidOCR with multi-language support (80+ languages)
- **DocLens Interleaved Approach**: Text as PRIMARY source, images as SECONDARY for structure
- **Automatic Chunking**: Handles documents up to millions of tokens via intelligent page-level chunking
- **Quality Assessment**: Automatic detection of low-quality OCR

### 2. Multimodal Entity Extraction

- **Model**: Gemini 2.5 Flash with 250K token context window
- **Extraction Strategy**: Text-Image pairs sent interleaved per page (prevents "Lost in the Middle")
- **Schema-Free**: Dynamic entity and relationship type discovery
- **Provenance Tracking**: Document name + page numbers for every entity
- **Batch Processing**: Iterative file processing to avoid memory overflow

### 3. Entity Deduplication (5 Methodologies)

| Method | Type | Key Technology | Best For |
|--------|------|----------------|----------|
| **R-Swoosh** | Rule-based | Transitive closure with match/merge rules | Exact matches, deterministic merging |
| **Probabilistic** | ML-based | Splink (EM algorithm) | Fuzzy matching, uncertain links |
| **Topological** | Graph-based | NetworkX Jaccard similarity | Entities with shared relationships |
| **Semantic** | Embedding | sentence-transformers + cosine similarity | Semantic equivalence, name variants |
| **LLM Full Context** | LLM-powered | Gemini 2.5 Pro, whole graph in one prompt (`GEMINI_MODEL_DEDUP`) | Complex disambiguation, reasoning |

### 4. SQL-Based Knowledge Graph

- **5-Table Normalized Schema**: documents, entities, relationships, entity_occurrences, relationship_occurrences
- **Junction Tables**: Full provenance tracking with document-level granularity
- **Access Control**: ACL tags on documents, access policies on relationships
- **Views**: Pre-built views for common queries
- **Indexing**: Optimized indexes for fast entity/relationship lookups

### 5. Hierarchical Community Detection

- **Algorithm**: Leiden (CPM optimization, better than Louvain)
- **3 Levels**: Micro (teams), Meso (departments), Macro (divisions)
- **Auto-Tuning**: Resolution parameters (γ) calculated based on graph density
- **Summaries**: Gemini-powered natural language descriptions per community
- **Recomputation Logic**: Automatic rebuild when graph changes >10%

### 6. Secure Query Orchestrator

- **Natural Language Queries**: Powered by Gemini for intent extraction
- **Semantic Search**: Milvus vector database for candidate retrieval
- **ACL Enforcement**: Multi-layer security (document tags, edge policies, ghost node elimination)
- **Graph Traversal**: Bidirectional BFS with cycle prevention
- **Citation Tracking**: Source documents and page numbers in every result

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA (optional, for GPU acceleration with RapidOCR)
- SQLite 3.37+
- 4GB+ RAM recommended

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Bakasur
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
```
google-genai>=0.2.0           # Gemini API
rapidocr-onnxruntime>=1.3.0   # Fast OCR
pymupdf>=1.23.0               # PDF processing
sentence-transformers>=2.2.0   # Embeddings
pymilvus>=2.3.0               # Vector database
splink>=3.9.0                 # Probabilistic matching
leidenalg                     # Community detection
python-igraph                 # Graph algorithms
```

### Step 4: Configure Environment

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Required: Google Gemini API Key
GOOGLE_API_KEY=your_api_key_here

# Optional: Logging
LOG_LEVEL=INFO

# Optional: model overrides (useful if a model is retired or has no quota
# on your plan, e.g. gemini-2.5-pro on the free tier)
# GEMINI_MODEL_HEAVY=gemini-2.5-flash
# GEMINI_MODEL_DEDUP=gemini-2.5-pro
```

**Get your API key from**: https://aistudio.google.com/app/apikey

The key is only needed for the Gemini stages (entity extraction, LLM
deduplication, community summaries, query planning). SQL conversion, Milvus
ingestion and `scripts/run_leiden.py --skip-summaries` run without it.

Milvus runs as **Milvus Lite** (a local `.db` file under `outputs/`); no
Milvus server is required. LibreOffice is optional: when present, DOCX pages
are rendered to images, otherwise DOCX files are processed as text only.

### Step 5: Initialize Database

```bash
python init_and_ingest.py
```

This will:
- Create SQL schema (documents, entities, relationships, etc.)
- Initialize Milvus vector database
- Verify setup

---

## Quick Start

### Process Your First Document

**1. Place documents in `dataset/` folder**:
```bash
cp your_document.pdf dataset/
```

**2. Run full pipeline**:
```bash
python main.py --dataset dataset/
```

This will:
- Parse all documents (PDF, DOCX, images, etc.)
- Extract entities using Gemini Vision + OCR
- Build knowledge graph
- Run 5 deduplication methodologies
- Save results to `outputs/`

**3. Convert to SQL** (select best method):
```bash
python kg_to_sql.py \
  --graph outputs/knowledge_graphs/dedup_llm_full_context_kg.gpickle \
  --output knowledge_graph.db
```

**4. Initialize vector search**:
```bash
python init_and_ingest.py
```

**5. Run Leiden community detection**:
```bash
python scripts/run_leiden.py
```

**6. Query your knowledge graph**:
```python
from src.orchestrator import QueryOrchestrator

orchestrator = QueryOrchestrator(db_path="knowledge_graph.db")

result = orchestrator.process_query(
    user_query="Who are the directors of Acme Corporation?",
    user_tags=["FINANCE", "HR"],
    max_depth=2
)

print(result)
```

### Expected Output Structure

```
Bakasur/
├── dataset/
│   └── your_document.pdf
├── outputs/
│   ├── parsed_documents/
│   │   └── your_document/
│   │       ├── images/
│   │       └── metadata.json
│   ├── entities_extracted/
│   │   └── your_document_entities.json
│   ├── knowledge_graphs/
│   │   ├── non_dedup_kg.gpickle / .graphml
│   │   ├── dedup_rswoosh_kg.gpickle / .graphml
│   │   ├── dedup_probabilistic_kg.gpickle / .graphml
│   │   ├── dedup_topological_kg.gpickle / .graphml
│   │   ├── dedup_semantic_kg.gpickle / .graphml
│   │   └── dedup_llm_full_context_kg.gpickle / .graphml
│   ├── reasoning/                 # LLM merge reasoning (JSON)
│   ├── benchmark_report_<timestamp>.json
│   └── milvus_orchestrator.db     # Milvus Lite file (no server needed)
└── knowledge_graph.db  # SQLite database
```

---

## Phase 1: Entity Extraction & Knowledge Graph

### Overview

Phase 1 implements the complete pipeline from raw documents to a deduplicated knowledge graph with full provenance tracking.

### 1.1 Document Parsing

**Component**: `src/rapidocr_parser.py`

**Supported Formats**:

| Format | Extensions | Parser | OCR Support |
|--------|-----------|--------|-------------|
| PDF (Text-based) | `.pdf` | PyMuPDF | No (direct extraction) |
| PDF (Scanned) | `.pdf` | PyMuPDF + RapidOCR | Yes |
| Images | `.jpg`, `.jpeg`, `.png`, `.tiff` | PIL + RapidOCR | Yes |
| Word Documents | `.docx` | python-docx + LibreOffice | Optional |
| Excel Spreadsheets | `.xlsx` | pandas + PIL (custom renderer) | No (native extraction) |

**Usage**:

```python
from src.rapidocr_parser import RapidOCRParser

parser = RapidOCRParser()
page_pairs = parser.parse(Path("document.pdf"))

# Each page_pair: (PIL.Image, ocr_text, metadata)
for image, text, metadata in page_pairs:
    print(f"Page {metadata['page_number']}")
    print(f"  OCR Confidence: {metadata.get('ocr_confidence', 'N/A')}")
    print(f"  Text Quality: {metadata.get('text_quality', 'N/A')}")
    print(f"  Text length: {len(text)} chars")
```

**Key Features**:
- Auto-detection of PDF type (text vs. scanned)
- OCR quality assessment with confidence scores
- Multi-language OCR support (English, Chinese, Hindi, etc.)
- PIL-based Excel rendering (~10-20x faster than matplotlib)
- Automatic column width calculation for spreadsheets

### 1.2 Multimodal Entity Extraction (DocLens Interleaved)

**Component**: `src/entity_extractor.py`

**Extraction Strategy (DocLens Phase 1)**:

The system sends text-image pairs **interleaved** per page:

```
[Instruction Prompt]
[Page 1 OCR Text (PRIMARY)]
[Page 1 Image (SECONDARY)]
[Page 2 OCR Text (PRIMARY)]
[Page 2 Image (SECONDARY)]
...
```

**Philosophy**:
- **Text is PRIMARY**: Ground truth for entity values (names, dates, IDs)
- **Images are SECONDARY**: Provide structure/layout context
- **Interleaved**: Text-image pairs stay together (prevents "Lost in the Middle")
- **NO summarization**: Full text preserved for maximum recall

**Configuration**:
- Model: `gemini-2.5-flash`
- Temperature: `0.1` (factual extraction)
- Max Output Tokens: `65,536`
- Max Input Tokens: `250,000` (auto-chunks if exceeded)

**Token Management**:
- Automatic chunking at page boundaries
- ~3,500 tokens per PDF page image
- Text adds ~1,000 tokens per page (inline)
- Entity ID remapping across chunks to avoid conflicts

**Response Format**:
```json
{
  "entities": [
    {
      "id": "PERSON_001",
      "type": "Person",
      "attributes": {
        "name": "John Smith",
        "role": "Director",
        "DIN": "12345678"
      }
    }
  ],
  "relationships": [
    {
      "from_id": "PERSON_001",
      "to_id": "COMPANY_001",
      "type": "DIRECTOR_OF",
      "attributes": {"appointment_date": "2020-01-01"}
    }
  ]
}
```

### 1.3 Knowledge Graph Construction

**Component**: `src/kg_builder.py`

**Strategy**: Every entity mention becomes a separate node (duplicates kept intentionally).

**Node Structure**:
```python
{
    'original_id': 'PERSON_001',           # From Gemini
    'type': 'Person',
    'source_doc': 'annual_report_2023',   # Which document
    'page_numbers': [5, 12, 18],          # Which pages
    'name': 'John Smith',                  # Attributes
    'role': 'Director',
    'DIN': '12345678'
}
```

**Internal Node ID**: `node_{counter}_{original_id}`
Example: `node_123_PERSON_001`

### 1.4 Entity Deduplication

**Component**: `src/methodologies/`

All 5 methodologies work on the raw knowledge graph and produce deduplicated versions.

#### 1.4.1 R-Swoosh (Rule-Based)

**File**: `src/methodologies/rswoosh.py`

- Transitive closure with match/merge rules
- String similarity (Levenshtein distance)
- Threshold: 0.85 (configurable)

#### 1.4.2 Probabilistic (Splink)

**File**: `src/methodologies/probabilistic.py`

- Expectation-Maximization (EM) algorithm
- Probabilistic record linkage
- Match probability threshold: 0.8

#### 1.4.3 Topological (NetworkX Jaccard)

**File**: `src/methodologies/topological.py`

- Jaccard similarity of node neighborhoods
- Graph structure-based matching
- Threshold: 0.7

#### 1.4.4 Semantic (Embeddings)

**File**: `src/methodologies/semantic.py`

- sentence-transformers (all-mpnet-base-v2)
- Cosine similarity on entity embeddings
- Threshold: 0.9

#### 1.4.5 LLM Full Context (Gemini)

**File**: `src/methodologies/llm_full_context.py`

1. The whole raw graph (entities, attributes, relationships) is sent in one prompt
2. The model returns duplicate clusters with an analysis and a MERGE decision
3. Model: `gemini-2.5-pro` by default, override with `GEMINI_MODEL_DEDUP`
4. Reasoning is saved to `outputs/reasoning/` for transparency

### 1.5 Running the Full Pipeline

```bash
python main.py --dataset dataset/
```

**Output**:
```
outputs/knowledge_graphs/
├── non_dedup_kg.gpickle                 # raw graph, one node per mention
├── dedup_rswoosh_kg.gpickle            # e.g. 60 nodes → 45 nodes
├── dedup_probabilistic_kg.gpickle
├── dedup_topological_kg.gpickle
├── dedup_semantic_kg.gpickle
└── dedup_llm_full_context_kg.gpickle   # (.graphml copies alongside)
```

---

## Phase 2: Query Orchestrator

### 2.1 SQL Schema

**File**: `sql/schema.sql`

**5-Table Normalized Design**:

1. **documents** - Source document metadata
2. **entities** - Deduplicated entities
3. **relationships** - Entity relationships
4. **entity_occurrences** - Provenance (which documents, which pages)
5. **relationship_occurrences** - Relationship provenance

**Key Features**:
- Junction tables for many-to-many provenance tracking
- ACL tags on documents (e.g., `["HR", "FINANCE"]`)
- Access policies on relationships
- Pre-built views for common queries

### 2.2 Converting Graph to SQL

```bash
python kg_to_sql.py \
  --graph outputs/knowledge_graphs/dedup_llm_full_context_kg.gpickle \
  --output knowledge_graph.db
```

### 2.3 Milvus Vector Database

**Component**: `src/milvus_ingestion.py`

```bash
python init_and_ingest.py
```

- Generates embeddings using `sentence-transformers/all-mpnet-base-v2`
- Ingests into Milvus (lite mode)
- Creates index for fast similarity search

### 2.4 Query Orchestrator

**Component**: `src/orchestrator.py`

**Pipeline**:
1. **Semantic Search** (Milvus) → Candidate entities
2. **LLM Query Planning** (Gemini) → Intent extraction
3. **ACL Enforcement** → Filter by user access tags
4. **Graph Traversal** (SecureGraphTraverser) → Build mini-graph
5. **Citation Tracking** → Attach source documents and page numbers

```python
from src.orchestrator import QueryOrchestrator

orchestrator = QueryOrchestrator(db_path="knowledge_graph.db")

result = orchestrator.process_query(
    user_query="Show me all directors of companies audited by KPMG",
    user_tags=["FINANCE", "AUDIT"],
    max_depth=2
)
```

### 2.5 Secure Graph Traversal

**Component**: `src/inference_engine.py` (`SecureGraphTraverser`)

**Security Layers**:
1. Document ACL filtering
2. Ghost node elimination
3. Edge policy checks
4. Node grounding verification

---

## Phase 3A: Leiden Community Detection

### Overview

Phase 3A implements hierarchical community detection using the Leiden algorithm with auto-tuned resolution parameters.

### 3.1 Graph Metrics & Auto-Tuning

**Component**: `src/graph_metrics.py`

**Auto-Tuned Resolution Parameters (γ)**:

| Graph Type | Density | Avg Degree | γ (Micro) | γ (Meso) | γ (Macro) |
|------------|---------|-----------|-----------|----------|-----------|
| **Dense** | >0.1 | >20 | 2.0 | 1.5 | 1.0 |
| **Medium** | 0.01-0.1 | 8-20 | 1.5 | 1.0 | 0.5 |
| **Sparse** | <0.01 | <8 | 1.0 | 0.5 | 0.3 |

### 3.2 Leiden Algorithm

**Component**: `src/leiden_builder.py`

**3-Level Hierarchy**:
- **Micro**: Fine-grained (~5-10 entities) - Teams, working groups
- **Meso**: Mid-level (~50-100 entities) - Departments, functions
- **Macro**: Coarse (~1000+ entities) - Divisions, organizations

```bash
# Full pipeline (auto-tuned)
python scripts/run_leiden.py

# Force rebuild
python scripts/run_leiden.py --force
```

### 3.3 Community Summarization

**Component**: `src/community_summarizer.py`

Uses Gemini (`MODEL_HEAVY`, gemini-2.5-flash by default) to generate natural language summaries for each community.

```python
from src.community_summarizer import CommunitySummarizer

summarizer = CommunitySummarizer(db_path="knowledge_graph.db")
counts = summarizer.generate_all_summaries()
```

---

## Document Processing

### DocLens Interleaved Approach

**Rationale**:
- Text provides precise values (names, dates, IDs)
- Images provide structure (tables, forms, layouts)
- Interleaving prevents "Lost in the Middle" problem
- No summarization = maximum recall

**Content Structure Sent to Gemini**:
```
[Instruction Prompt]
[Page 1 OCR Text] ← PRIMARY
[Page 1 Image]    ← SECONDARY
[Page 2 OCR Text] ← PRIMARY
[Page 2 Image]    ← SECONDARY
```

### RapidOCR Configuration

**Languages**:
```python
RAPIDOCR_LANGUAGES = ['en', 'ch']  # English, Chinese
# Available: 80+ languages
```

**Performance**:
- CPU: ~0.1s per page
- GPU: ~0.05s per page
- Accuracy: ~95% for printed text, ~60-70% for handwritten

---

## Configuration Reference

**File**: `src/config.py`

### Key Settings

```python
# Models
MODEL_HEAVY = "gemini-2.5-flash"           # Entity extraction
MODEL_LIGHT = "gemini-2.5-flash-lite"      # Comparisons
MODEL_FEATHER = "gemini-2.0-flash-lite"    # Summaries

# Token Limits
MAX_TOKENS_PER_REQUEST = 250_000

# Document Formats
SUPPORTED_FORMATS = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.docx', '.xlsx']

# OCR
RAPIDOCR_LANGUAGES = ['en', 'ch']
OCR_CONFIDENCE_THRESHOLD = 0.7
TEXT_QUALITY_THRESHOLD = 0.6

# Deduplication Thresholds
RSWOOSH_SIMILARITY_THRESHOLD = 0.85
SPLINK_MATCH_PROBABILITY_THRESHOLD = 0.8
TOPOLOGICAL_JACCARD_THRESHOLD = 0.7
SEMANTIC_SIMILARITY_THRESHOLD = 0.9
```

---

## Database Schema

### Core Tables

**1. documents** - Source document metadata
```sql
document_id, document_name, file_path, access_tags, created_at
```

**2. entities** - Deduplicated entities
```sql
unique_entity_id, entity_type, canonical_name, name_variants,
cluster_size, attributes, merge_reasoning, created_at
```

**3. relationships** - Entity relationships
```sql
relationship_id, relationship_type, from_entity_id, to_entity_id,
attributes, access_policy, created_at
```

**4. entity_occurrences** - Provenance
```sql
entity_id, document_id, page_numbers
```

**5. relationship_occurrences** - Relationship provenance
```sql
relationship_id, document_id, page_numbers
```

### Community Tables (Phase 3A)

**6. leiden_communities**
```sql
entity_id, level, resolution, community_id, modularity
```

**7. community_summaries**
```sql
community_key, level, community_id, summary, entity_count,
relationship_count, top_entities, density, created_at
```

---

## API Reference

### RapidOCRParser

```python
from src.rapidocr_parser import RapidOCRParser

parser = RapidOCRParser()
page_pairs = parser.parse(file_path: Path)
```

### MultimodalEntityExtractor

```python
from src.entity_extractor import MultimodalEntityExtractor

result = extractor.extract_entities(
    page_pairs: List[Tuple],
    doc_name: str,
    save_path: Path = None
)
```

### QueryOrchestrator

```python
from src.orchestrator import QueryOrchestrator

result = orchestrator.process_query(
    user_query: str,
    user_tags: List[str],
    max_depth: int = 2
)
```

### LeidenCommunityBuilder

```python
from src.leiden_builder import LeidenCommunityBuilder
￼Filtering based on entity access, documents, relationships
builder = LeidenCommunityBuilder()
stats = builder.build_communities()
```

---

## Complete Workflows

### Workflow 1: Full Pipeline

```bash
# 1. Place documents in dataset/
cp *.pdf dataset/

# 2. Run entity extraction & deduplication
python main.py --dataset dataset/

# 3. Convert to SQL (pick one deduplicated graph)
python kg_to_sql.py --graph outputs/knowledge_graphs/dedup_llm_full_context_kg.gpickle --output knowledge_graph.db

# 4. Initialize Milvus
python init_and_ingest.py

# 5. Build communities
python scripts/run_leiden.py

# 6. Query (runs a demo query; see the Python API above for your own)
python -m src.orchestrator
```

### Workflow 2: Benchmark Methods

```bash
python benchmark_deduplication.py outputs/entities_extracted/document_entities.json
```

---

## CLI Reference

### main.py - Full Pipeline

```bash
python main.py [--dataset PATH] [--config] [--log-level LEVEL]
```

### kg_to_sql.py - Convert Graph to SQL

```bash
python kg_to_sql.py --graph PATH_TO_GPICKLE [--output knowledge_graph.db]
```

### scripts/run_leiden.py - Community Detection

```bash
python scripts/run_leiden.py [--force] [--skip-summaries] [--check-only]
```

---

## Testing

```bash
pip install -r requirements.txt

# Offline suite: no API key or network needed (Gemini and the embedding
# model are replaced by fakes; Milvus Lite and SQLite use temp files)
python -m pytest

# Live end-to-end test (two Gemini calls); skipped when GOOGLE_API_KEY is unset
python -m pytest -m e2e
```

The tests use small synthetic documents for a fictional company in
`tests/fixtures/` (PDF, DOCX, XLSX, PNG). Rebuild them with
`python tests/fixtures/make_fixtures.py`.

---

## Performance

| Operation | Speed |
|-----------|-------|
| **OCR (RapidOCR)** | 0.1s/page (CPU), 0.05s/page (GPU) |
| **Entity Extraction** | 5-10s/page |
| **Deduplication (LLM)** | 30-60s (100 entities) |
| **Leiden (3 levels)** | 2-5s (100 entities) |

---

## Troubleshooting

### ImportError: leidenalg not found

```bash
pip install leidenalg python-igraph
```

### Gemini API quota exceeded

- Check quota: https://aistudio.google.com
- Reduce batch size
- Use `MODEL_LIGHT` for non-critical tasks

### Low OCR quality

**Expected** - RapidOCR has ~60-70% on handwriting. Gemini Vision handles it better!

---

## Advanced Topics

### Custom Entity Types

Edit `src/config.py`:

```python
ENTITY_EXTRACTION_PROMPT_TEMPLATE = """
Extract entities including:
- Custom Type 1 (e.g., "Product", "Patent")
- Custom Type 2 (e.g., "Location", "Event")
"""
```

### ACL Configuration

```sql
-- Set document tags
UPDATE documents SET access_tags = '["EXECUTIVE"]' WHERE ...;

-- Set relationship policies
UPDATE relationships SET access_policy = 'RESTRICTED' WHERE ...;
```

---

## Project Structure

```
Bakasur/
├── src/                           # Core modules
│   ├── rapidocr_parser.py         # Document parser
│   ├── entity_extractor.py        # Gemini extraction
│   ├── kg_builder.py              # Graph construction
│   ├── orchestrator.py            # Query orchestrator
│   ├── inference_engine.py        # Graph traversal
│   ├── leiden_builder.py          # Communities
│   └── methodologies/             # 5 dedup methods
├── sql/                           # Database schemas
├── dataset/                       # Input documents
├── outputs/                       # Generated results
├── main.py                        # Full pipeline
├── kg_to_sql.py                   # Graph → SQL
└── README.md                      # This file
```

---

## References

- **Google Gemini**: Multimodal LLM
- **RapidOCR**: https://github.com/RapidAI/RapidOCR
- **Milvus**: https://milvus.io
- **Leiden Algorithm**: https://github.com/vtraag/leidenalg
- **Splink**: https://github.com/moj-analytical-services/splink

---

**Built for knowledge graph practitioners** | Last Updated: 2024-12-22
