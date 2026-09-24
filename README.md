# GraphRAG++

Corporate and financial documents mention the same people and companies many times, spelled differently across annual reports, board minutes, spreadsheets and scanned letters. GraphRAG++ builds a knowledge graph from such documents with Gemini and keeps every mention as its own node. It then runs five entity-resolution methods side by side, from string rules to an LLM that sees the whole graph at once, to decide which mentions refer to the same entity. You pick one resulting graph. It is loaded into SQLite and Milvus Lite, and a query orchestrator returns a cited, access-filtered subgraph for a natural-language question. The orchestrator does not write prose answers.

This is a personal proof of concept, not a finished product. No real documents ship with the repository. You supply your own, and `tests/fixtures/` has four small synthetic ones (a fictional "Acme Widgets Pvt Ltd") to try it on.

## How it works

```
dataset/  (PDF, JPG/PNG/TIFF, DOCX, XLSX)
   │  RapidOCRParser: one (page image, page text, metadata) triple per page or sheet
   ▼
Gemini extraction: text + image per page, interleaved          → outputs/entities_extracted/*.json
   ▼
Mention graph (NetworkX): one node per mention                → outputs/knowledge_graphs/non_dedup_kg.gpickle
   ▼
Five dedup methods, side by side                              → dedup_<method>_kg.gpickle, benchmark_report_*.json
   R-Swoosh · Splink · topological · semantic · LLM full-context
   ▼  (pick one graph)
kg_to_sql.py → SQLite: documents, entities, relationships, per-document occurrences with page numbers
   ├─► init_and_ingest.py → Milvus Lite entity vectors (all-mpnet-base-v2, local file, no server)
   └─► scripts/run_leiden.py → Leiden communities at three levels + Gemini summaries
   ▼
QueryOrchestrator: vector search → Gemini query plan → access filter → SQL graph walk
                   → matched entities, nodes, edges, {document: [pages]}
```

**Parsing.** PDFs with a text layer are read with PyMuPDF. Scanned PDFs and images go through RapidOCR. A PDF counts as scanned when none of its first three pages has more than 100 characters of text, and the choice then applies to the whole file. DOCX text comes from python-docx, and pages are rendered to images when LibreOffice is installed (text only otherwise). Each XLSX sheet becomes one "page": the sheet as text plus a rendered image of it.

**Extraction.** Pages are sent to Gemini (`gemini-2.5-flash` by default) as text-then-image pairs. The text is the source for names and numbers, and the image supplies layout. The prompt is schema-free, so entity and relationship types are whatever the model finds. Documents larger than 250k tokens are split at page boundaries, and entity IDs are remapped across the chunks.

**Deduplication.** Every method works on the same mention graph:

| Method | How it decides | Default threshold |
|---|---|---|
| R-Swoosh | Iterative merge-closure. Same type only; weighted similarity of name and attributes (rapidfuzz); conflicting IDs such as DIN or PAN count against a match. | 0.85 |
| Probabilistic | Splink 3 (Fellegi–Sunter model trained with EM, DuckDB backend) | match probability 0.8 |
| Topological | Same type only; 0.7 × Jaccard similarity of neighbours + 0.3 × name-token overlap | 0.7 |
| Semantic | `all-mpnet-base-v2` embeddings of each entity's attributes in Milvus Lite | cosine 0.9 |
| LLM full-context | The whole graph goes into one prompt (`gemini-2.5-pro` by default). The model returns merge groups with its reasoning, which is saved to `outputs/reasoning/`. | — |

**Querying.** Milvus first retrieves candidate entities. Gemini then plans the query in one of two modes:

- *Flashlight* explores up to `max_depth` hops around the named entities or entity types.
- *Laser* finds the shortest path between two entities.

The graph walk is a recursive SQL query. It only reaches entities that occur in documents the caller may see, and only follows relationships whose policy is `OPEN`. Citations are filtered the same way.

**Communities.** Leiden (`leidenalg`, RB-configuration partition) runs at three resolutions: micro, meso and macro. The resolutions are chosen from the graph's density, average degree and clustering. Gemini summarises each community. Results go to `leiden_communities` and `community_summaries`. `run_leiden.py` only recomputes when entity or relationship counts have changed by more than 10 % or 15 % since the last run, or when given `--force`.

## Inputs and questions

**Inputs:** `.pdf` (text layer or scanned), `.jpg`, `.jpeg`, `.png`, `.tiff`, `.docx` and `.xlsx`, placed in a folder. The prompt is written for financial and corporate documents: annual reports, board minutes, related-party schedules, letterheads.

**Questions the orchestrator handles:**

- "Who is Director B?" or "Show me all companies": flashlight mode. You get the matching entities with their attributes, name variants and how many mentions were merged, plus the surrounding subgraph.
- "How is Director A connected to Beta Supplies LLP?": laser mode, which returns the shortest path.
- Every answer includes `citation_context`, which lists each document and the pages the returned entities appear on, limited to documents the caller can see.

Other outputs you can inspect:

- How many merge groups each method found and how far it shrank the graph: `benchmark_report_*.json`. The merged graphs are in `dedup_<method>_kg.gpickle` and `.graphml`.
- Why the LLM merged what it did: `outputs/reasoning/`, or `python inspect_llm_reasoning.py <file>`.
- Community membership and summaries in SQLite.

## Quick start

Tested with Python 3.12. Run everything from the repository root, because the scripts read and write `knowledge_graph.db` and `outputs/` there.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                  # set GOOGLE_API_KEY

mkdir -p dataset && cp tests/fixtures/acme_* dataset/ # or your own documents
python main.py --dataset dataset/                     # parse → extract → mention graph → 5 dedup methods
python kg_to_sql.py --graph outputs/knowledge_graphs/dedup_llm_full_context_kg.gpickle   # → knowledge_graph.db
python init_and_ingest.py                             # entity vectors → outputs/milvus_orchestrator.db
python scripts/run_leiden.py                          # add --skip-summaries to avoid API calls
```

```python
from src.orchestrator import QueryOrchestrator

orch = QueryOrchestrator(db_path="knowledge_graph.db")
result = orch.process_query("How is Director A connected to Beta Supplies LLP?",
                            user_tags=["UNCLASSIFIED"], max_depth=2)
for edge in result["mini_graph"]:
    print(edge["source_name"], edge["relation"], edge["target_name"])
print(result["citation_context"])                     # {document: [page numbers]}
orch.close()
```

`kg_to_sql.py` asks before it overwrites an existing `knowledge_graph.db`. The embedding model is downloaded the first time it is used.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Needed for extraction, LLM dedup, community summaries and query planning. `kg_to_sql.py`, `init_and_ingest.py` and `run_leiden.py --skip-summaries` run without it. |
| `GEMINI_MODEL_HEAVY` | `gemini-2.5-flash` | Extraction, query planning, community summaries |
| `GEMINI_MODEL_DEDUP` | `gemini-2.5-pro` | LLM full-context dedup. A free-tier key has no quota for this model, so override it there. |
| `GEMINI_MODEL_LIGHT` | `gemini-2.5-flash-lite` | Only recorded in the benchmark report |
| `RSWOOSH_SIMILARITY_THRESHOLD`, `SPLINK_MATCH_PROBABILITY_THRESHOLD`, `TOPOLOGICAL_JACCARD_THRESHOLD`, `SEMANTIC_SIMILARITY_THRESHOLD` | 0.85 / 0.8 / 0.7 / 0.9 | Dedup thresholds |
| `LOG_LEVEL` | `INFO` | Logging; `main.py` also writes `outputs/benchmark.log` |

Other settings are in `src/config.py`, including the extraction prompt, the 250k-token chunk limit and the embedding model.

**Access control.** `kg_to_sql.py` tags every document `["UNCLASSIFIED"]` and every relationship `OPEN`, so everything is visible by default. To restrict data, edit the database:

```sql
UPDATE documents SET access_tags = '["BOARD"]' WHERE document_name = 'acme_board_minutes';
UPDATE relationships SET access_policy = 'RESTRICTED' WHERE relationship_type = 'LOAN_PROVIDER';
```

A caller then sees a document only if it is `UNCLASSIFIED` or shares one of the caller's `user_tags`. Restricted relationships are never traversed.

## Tests

```bash
python -m pytest            # 45 offline tests, about 40 s
python -m pytest -m e2e     # 1 live test (two Gemini calls); skipped unless GOOGLE_API_KEY is set
```

The offline suite replaces the Gemini client and the embedding model with deterministic fakes and runs Milvus Lite and SQLite on temporary files. It covers the parsers on the four fixtures (PDF text layer, DOCX with and without LibreOffice, XLSX, OCR on a PNG), chunking and parsing of extraction responses, each dedup method, `kg_to_sql`, access-filtered traversal, vector search, the orchestrator's flashlight, laser, ambiguous and no-access paths, Leiden with summaries, a full offline run from `main` to a query, and the CLIs without an API key.

The live test parses the fixture PDF, extracts with Gemini, runs R-Swoosh, SQL, Milvus and Leiden, and answers a path question. It passes with `GEMINI_MODEL_HEAVY=gemini-2.5-flash-lite`.

## Observations on the synthetic fixtures

These come from one run of the Quick start above on the four fixtures, with `GEMINI_MODEL_HEAVY` and `GEMINI_MODEL_DEDUP` set to `gemini-2.5-flash-lite`. Extraction produced 17 mentions. The output varies between runs, so treat this as an illustration, not a benchmark.

| Method | Result |
|---|---|
| LLM full-context | 4 merge groups, all correct: the three spellings of Acme, Beta Supplies across the Company and Organization types, Director A, Director B. It left one of the five Director A mentions unmerged. |
| Semantic | Merged the Acme spellings and Beta Supplies, but also merged Director A with Director B: the near-identical names embed almost identically. |
| R-Swoosh | Kept the two directors apart, but only because their extracted roles differed. On name alone, "Director A" and "Director B" score 0.90, above the 0.85 threshold. It missed "Acme Widgets Private Limited", two Director A mentions with other roles, and the Beta mention typed as Organization. |
| Topological | No merges. Every mention is its own node, so mentions from different documents share no neighbours. |
| Splink | No merges on a graph this small. |

The laser query above returned `Director A -[DESIGNATED_PARTNER_OF]-> Beta Supplies LLP`, with page citations.

## Status and limitations

- This is a proof of concept. The pipeline stops at retrieval: it returns a subgraph and citations, not a written answer. `InferenceStage.pdf` is a draft design for a fuller answer stage (several sampled answers plus an adjudicator, with caching and disambiguation). That stage is not implemented here.
- Communities and their summaries are stored but the query orchestrator does not use them yet. On small graphs, the summaries can guess at an organisational role the documents do not state.
- API errors are logged and the run carries on. With an exhausted quota, `main.py` still printed "Benchmark completed successfully" over an empty graph. With the default `gemini-2.5-pro` on a free-tier key, the LLM method reports zero merges. Check `outputs/benchmark.log`.
- All documents are extracted in one run and the mention graph is held in memory. There is no incremental update of an existing database.
- The scripts use fixed relative paths (`knowledge_graph.db`, `outputs/`, `sql/`), so run them from the repository root.
- Milvus Lite runs on Linux and macOS.
- `src/mineru_parser.py` is an earlier MinerU-based PDF parser that the pipeline no longer uses.
- `main` is the current branch. `master`, `mihirm/access-level-filtering` and `mihirm/excel-optimised-ocr` are earlier points in the same history, already contained in `main`.

## Repository layout

```
main.py                   pipeline entry: parse → extract → mention graph → five dedup methods
kg_to_sql.py              chosen graph → SQLite
init_and_ingest.py        SQLite entities → Milvus Lite (init_and_ingest.sh does the same with the sqlite3 CLI)
scripts/run_leiden.py     communities and summaries
src/rapidocr_parser.py    PDF, image, DOCX and XLSX parsing
src/entity_extractor.py   Gemini extraction; src/token_manager.py splits large documents into chunks
src/kg_builder.py         mention graph and merged graphs
src/methodologies/        the five dedup methods
src/orchestrator.py       query planning; src/inference_engine.py access-filtered SQL graph walk;
                          src/milvus_ingestion.py entity vectors
src/graph_metrics.py, src/leiden_builder.py, src/community_summarizer.py   communities
src/benchmark_harness.py  runs the pipeline for main.py
sql/                      schema.sql, leiden_schema.sql
benchmark_deduplication.py, inspect_llm_reasoning.py, debug_milvus.py, load_minigraph.py   helper scripts
InferenceStage.pdf        draft design for an answer-generation stage
tests/                    offline and live tests; fixtures/ holds the synthetic documents and make_fixtures.py
```

Licence: not yet specified.
