# Bakasur: Entity Resolution & Knowledge Graph Construction

Production-ready entity resolution system for financial documents with 5 deduplication methodologies, explainable AI reasoning, and SQL export capabilities.

---

## 🎯 Overview

**Bakasur** extracts entities from financial PDFs, builds knowledge graphs, and deduplicates them using multiple methodologies including LLM-based approaches with full reasoning transparency.

### Key Features

- **Multimodal Processing**: Gemini 2.5 Flash analyzes page images + text for superior entity extraction
- **Dynamic Schema**: No hardcoded schemas - LLM discovers entity types automatically
- **5 Deduplication Methods**: Compare traditional (R-Swoosh, Probabilistic, Topological, Semantic) vs. LLM-based approaches
- **Explainable AI**: LLM Full Context provides natural language reasoning for every merge decision
- **SQL Export**: Convert knowledge graphs to normalized SQL databases for fast lookups
- **Token-Safe**: Automatic chunking at 250k tokens, prevents API errors

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ FULL PIPELINE (main.py)                                     │
│ PDF Files → PyMuPDF → Entity Extractor → KG Builder  │
│                                   ↓                          │
│                        Non-Deduplicated KG                   │
│                                   ↓                          │
│   ┌──────────┬─────────────┬────────────┬──────────────┐   │
│  Rswoosh  Probabilistic  Topological  Semantic  LLM Full │
│                                                  Context  │
│   └─────────┴─────────────┴────────────┴──────────────┘   │
│                          5 Deduplicated KGs                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PARTIAL PIPELINE (benchmark_deduplication.py)               │
│ Pre-extracted Entities JSON → KG Builder → Deduplication   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ SQL CONVERSION (kg_to_sql.py)                               │
│ Deduplicated KG → 5-Table SQLite Database                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 2. Validate Setup

```bash
python test_setup.py
```

### 3. Run Full Pipeline

```bash
# Process PDFs end-to-end
python main.py --dataset dataset/

# Output: 5 deduplicated graphs + benchmark report
```

### 4. Inspect Results

```bash
# View LLM reasoning
python inspect_llm_reasoning.py --all

# Convert to SQL
python kg_to_sql.py --graph outputs/knowledge_graphs/dedup_llm_full_context_*.gpickle

# Query SQL database
sqlite3 knowledge_graph.db "SELECT * FROM entities WHERE canonical_name LIKE '%EXAMPLECO%'"
```

---

## 📦 Components

### A. PDF Parsing (`src/mineru_parser.py`)

Extracts structured data from PDFs using Magic-PDF (Mineru):
- Layout-aware extraction (tables, images, text)
- GPU-accelerated OCR
- Outputs: `(page_image, page_text, metadata)` per page

### B. Entity Extraction (`src/entity_extractor.py`)

Multimodal LLM extraction with Gemini 2.5 Flash:
- Processes image + text together for better context
- Dynamic schema - discovers entity types on-the-fly
- Automatic chunking at 250k tokens
- Outputs: `{entities: [...], relationships: [...]}`

### C. Knowledge Graph Builder (`src/kg_builder.py`)

NetworkX-based graph construction:
- **Non-deduplicated**: Every entity mention = separate node
- **Deduplicated**: Clusters merged with reasoning attached
- Formats: `.gpickle` (Python), `.graphml` (universal)

### D. Deduplication Methodologies

| Method | Approach | Speed | Accuracy | Cost | Reasoning |
|--------|----------|-------|----------|------|-----------|
| **R-Swoosh** | String matching + transitive closure | ⚡ Fast | Medium | Free | None |
| **Probabilistic** | Fellegi-Sunter EM (Splink) | Medium | High | Free | Statistical |
| **Topological** | Graph neighbor Jaccard similarity | ⚡ Fast | Medium | Free | None |
| **Semantic** | Embeddings (all-mpnet-base-v2) + cosine | 🐌 Slow | High | Free | Similarity scores |
| **LLM Full Context** ⭐ | Gemini 2.5 Pro holistic analysis | Medium | **Highest** | $ | **Full natural language** |

**Why LLM Full Context is Recommended:**
- ✅ Only 1 API call (efficient, not wasteful)
- ✅ Natural language explanation for every merge
- ✅ Handles complex cases (typos, abbreviations, name variations)
- ✅ Transparent and auditable decisions

### E. SQL Conversion (`kg_to_sql.py`)

Converts knowledge graphs to normalized SQLite database:
- 5-table schema: `documents`, `entities`, `relationships`, `entity_occurrences`, `relationship_occurrences`
- Hash-based unique IDs (MD5 of canonical name + type)
- Indexed for fast lookups (<1ms)

### F. Reasoning Inspection (`inspect_llm_reasoning.py`)

Interactive CLI to browse LLM merge decisions:
- View which entities were merged and why
- Filter by cluster, entity type
- Access reasoning from graph nodes or JSON files

---

## 🔄 Workflows

### Workflow 1: Full Pipeline (PDF → Deduplicated KG)

```bash
python main.py --dataset dataset/

# Steps performed:
# 1. Parse PDFs with Mineru
# 2. Extract entities with Gemini 2.5
# 3. Build non-deduplicated KG
# 4. Run 5 deduplication methodologies
# 5. Save 5 deduplicated KGs + reasoning
# 6. Generate benchmark report
```

**Outputs:**
- `outputs/mineru_parsed/` - Parsed PDFs (images + text)
- `outputs/entities_extracted/` - Entity JSON files
- `outputs/knowledge_graphs/` - 5 deduplicated KGs
- `outputs/reasoning/` - LLM reasoning JSON
- `outputs/dedup_benchmark_*.json` - Comparison report

### Workflow 2: Partial Pipeline (Pre-Extracted Entities)

```bash
python benchmark_deduplication.py "outputs/entities_extracted/XBRL_CFS_entities.json"

# Skips PDF parsing/extraction, runs only deduplication
```

**Use when:** You already have entities extracted and want to re-run deduplication with different thresholds.

### Workflow 3: Inspect LLM Reasoning

```bash
# Summary
python inspect_llm_reasoning.py

# All clusters
python inspect_llm_reasoning.py --all

# Specific cluster
python inspect_llm_reasoning.py --cluster 5

# Reasoning in graph nodes
python inspect_llm_reasoning.py --graph outputs/knowledge_graphs/dedup_llm_full_context_*.gpickle
```

**Example Output:**
```
Cluster ID: 1
Entities Merged: 3
Entity IDs: node_1_PERSON_001, node_45_PERSON_045, node_89_PERSON_089

Analysis:
  All three entities represent the same person 'John Smith' (Director). 
  Evidence: (1) PERSON_001 and PERSON_045 share identical DIN 12345678;
  (2) PERSON_089 has same role and company; (3) 'J. Smith' is common 
  abbreviation of 'John Smith'.

Decision:
  MERGE - These 3 entities represent John Smith (DIN 12345678), Director at ABC Corp
```

### Workflow 4: SQL Export & Querying

```bash
# Convert
python kg_to_sql.py \
  --graph outputs/knowledge_graphs/dedup_llm_full_context_*.gpickle \
  --output kg.db

# Query
sqlite3 kg.db

# Example queries:
SELECT * FROM entities WHERE entity_type = 'Company';

SELECT canonical_name, cluster_size 
FROM entities 
WHERE cluster_size > 1 
ORDER BY cluster_size DESC;

SELECT json_extract(attributes, '$.corporate_identity_number') AS cin, 
       canonical_name 
FROM entities 
WHERE cin IS NOT NULL;

-- Relationships for specific entity
SELECT r.relationship_type, e2.canonical_name AS related_to
FROM relationships r
JOIN entities e2 ON r.to_entity_id = e2.unique_entity_id
WHERE r.from_entity_id = '<entity_hash>';
```

---

## 📂 Output Structure

```
outputs/
├── mineru_parsed/              # Parsed PDFs (from main.py)
│   └── XBRL_CFS/
│       ├── page_001.png
│       └── page_001.txt
├── entities_extracted/         # Entity JSON files (from main.py)
│   └── XBRL_CFS_entities.json
├── knowledge_graphs/           # Deduplicated KGs
│   ├── non_dedup_kg.gpickle        # Raw graph (main.py only)
│   ├── dedup_rswoosh_kg.gpickle
│   ├── dedup_probabilistic_kg.gpickle
│   ├── dedup_topological_kg.gpickle
│   ├── dedup_semantic_kg.gpickle
│   └── dedup_llm_full_context_kg.gpickle  ← With reasoning!
├── reasoning/                  # LLM reasoning JSON
│   └── llm_full_context_reasoning_*.json
└── dedup_benchmark_*.json     # Benchmark report
```

---

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Gemini Models
MODEL_HEAVY = "gemini-2.5-flash"      # Entity extraction, LLM dedup
MODEL_LIGHT = "gemini-2.5-flash-lite"  # Token counting

# Token Management
MAX_TOKENS_PER_REQUEST = 250_000  # Auto-chunk if exceeded

# Deduplication Thresholds
RSWOOSH_SIMILARITY_THRESHOLD = 0.85
SPLINK_MATCH_PROBABILITY_THRESHOLD = 0.8
TOPOLOGICAL_JACCARD_THRESHOLD = 0.7
SEMANTIC_SIMILARITY_THRESHOLD = 0.9

# Directories
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REASONING_DIR = OUTPUTS_DIR / "reasoning"
```

Or set via environment variables in `.env`:
```bash
GOOGLE_API_KEY=your_api_key_here
RSWOOSH_SIMILARITY_THRESHOLD=0.90
```

---

## 🗃️ SQL Schema

5-table normalized design:

```sql
-- Unique documents
documents (document_id, document_name, file_path)

-- Deduplicated entities with golden truth names
entities (
    unique_entity_id,      -- MD5 hash(canonical_name|type)
    entity_type,           -- Person, Company, etc.
    canonical_name,        -- Golden truth name
    cluster_size,          -- Number of entities merged
    attributes,            -- JSON blob (CIN, PAN, role, etc.)
    merge_reasoning        -- JSON (LLM reasoning if available)
)

-- Relationships between entities
relationships (
    relationship_id,       -- MD5 hash(from|to|type)
    relationship_type,     -- DIRECTOR_OF, AUDITED_BY, etc.
    from_entity_id,
    to_entity_id,
    attributes             -- JSON blob
)

-- Which entities appear in which documents/pages
entity_occurrences (entity_id, document_id, page_numbers)

-- Which relationships appear in which documents/pages
relationship_occurrences (relationship_id, document_id, page_numbers)
```

**Example Queries:** See `sql/sample_queries.sql`

---

## 🛠️ Troubleshooting

### Magic-PDF Installation Issues

```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils libgl1

# Then retry
pip install magic-pdf
```

### CUDA/GPU Issues

If you don't have CUDA or want CPU-only mode:
```python
# Edit src/config.py
MINERU_USE_GPU = False
```

### Memory Issues

For large documents, reduce batch sizes:
```python
# In src/entity_extractor.py or src/methodologies/semantic.py
batch_size = 8  # Reduce from default
```

### API Rate Limits

- Scripts automatically retry on rate limits
- Process fewer PDFs at once
- Add delays between calls if needed

---

## 📊 Performance

- **Single PDF**: ~5-10 minutes
- **10 PDFs**: ~1-2 hours (depends on size)
- **Database size**: ~500KB per 60 entities
- **SQL query time**: <1ms (indexed lookups)

---

## 📁 Project Structure

```
Bakasur/
├── main.py                          # Full pipeline entry point
├── benchmark_deduplication.py       # Partial pipeline (pre-extracted entities)
├── inspect_llm_reasoning.py         # Reasoning browser
├── kg_to_sql.py                     # SQL conversion
├── test_setup.py                    # Setup validation
├── test_iterative.py                # PDF parsing test
├── test_token_counting.py           # Token management test
├── requirements.txt                 # Dependencies
├── .env                             # API keys (gitignored)
├── .env.example                     # Template
├── README.md                        # This file
├── src/
│   ├── config.py                    # Configuration
│   ├── mineru_parser.py             # PDF parsing
│   ├── entity_extractor.py          # Entity extraction
│   ├── token_manager.py             # Token counting
│   ├── kg_builder.py                # Graph construction
│   ├── benchmark_harness.py         # Full pipeline orchestration
│   └── methodologies/
│       ├── rswoosh.py               # R-Swoosh deduplication
│       ├── probabilistic.py         # Splink/Fellegi-Sunter
│       ├── topological.py           # NetworkX Jaccard
│       ├── semantic.py              # Embedding-based
│       └── llm_full_context.py      # LLM Full Context ⭐
├── sql/
│   ├── schema.sql                   # Database schema
│   └── sample_queries.sql           # Example SQL queries
├── dataset/                         # PDF files (user-provided)
└── outputs/                         # Generated results
```

---

## 📚 Advanced Usage

### Custom Deduplication Thresholds

```python
# src/config.py
RSWOOSH_SIMILARITY_THRESHOLD = 0.90  # More conservative (fewer merges)
SEMANTIC_SIMILARITY_THRESHOLD = 0.85  # More aggressive (more merges)
```

### Programmatic Access

```python
import pickle
import networkx as nx

# Load graph
with open("outputs/knowledge_graphs/dedup_llm_full_context_kg.gpickle", "rb") as f:
    kg = pickle.load(f)

# Query nodes
for node, attrs in kg.nodes(data=True):
    if attrs.get('entity_type') == 'Person':
        print(f"{attrs['canonical_name']}: {attrs.get('merge_reasoning')}")

# Query relationships
for u, v, attrs in kg.edges(data=True):
    print(f"{kg.nodes[u]['canonical_name']} --[{attrs['relationship_type']}]--> {kg.nodes[v]['canonical_name']}")
```

### Batch Processing

```python
from pathlib import Path
import subprocess

pdf_dir = Path("dataset")
for pdf in pdf_dir.glob("*.pdf"):
    subprocess.run(["python", "main.py", "--dataset", str(pdf.parent)])
```

---

## 🤝 Contributing

For bugs and feature requests, please open an issue on GitHub.

---

## 📝 License

MIT License

---

## 🎓 Citation

If you use Bakasur in your research:

```bibtex
@software{bakasur2024,
  title={Bakasur: Entity Resolution and Knowledge Graph Construction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Bakasur}
}
```

---

## 📞 Support

- **Documentation**: This README
- **Configuration**: `src/config.py`
- **Logs**: `outputs/benchmark.log`
- **Issues**: GitHub Issues

---

**Built with:** Python 3.9+, NetworkX, Gemini 2.5, Splink, Sentence Transformers, SQLite
