# Benchmark results

Two questions, measured on a synthetic corpus with exact ground truth:

1. **Entity resolution.** How do the five deduplication methods compare with each other and
   with two plain string-matching baselines?
2. **Question answering.** Does GraphRAG++'s cited-subgraph retrieval answer better than a basic
   RAG pipeline, or than pasting every document into one prompt?

The QA benchmark was run twice on the same local model: once on the code as it stood, and
again after fixing three problems the first run exposed. Both are reported.

## Setup

- **Corpus** (`corpus/`, built by `generate_corpus.py` from structured facts): 8 documents for
  a fictional group of companies: two annual reports (PDF), two sets of board minutes (DOCX),
  a related-party schedule and a shareholding register (XLSX), a loan sanction letter and an
  auditor's engagement letter (PDF/DOCX). About 20k tokens in total. The same people and
  companies appear under different names across documents ("Director A", "A. Director",
  "Mr Director A", abbreviations), so resolution and cross-document linking matter.
- **Ground truth** (`ground_truth/`): 20 entities with their mentions, the true relations, and
  39 questions of six types: lookup (8), attribute (6), list (6), yes/no (4), multi-hop (8),
  cross-document (7). 11 questions need facts from two documents.
- **Scoring** (`scoring.py`, unit-tested): exact or normalised match for single answers,
  entity-set F1 for list answers; pairwise precision/recall/F1 and cluster purity for
  resolution. No LLM judge.
- **Systems compared for QA**, each answering with the same model within a run:

| System | What it is |
|---|---|
| `graphrag` | This repository: parse → LLM extraction → LLM dedup → SQLite graph + Milvus Lite → planner (flashlight / laser) → cited subgraph → one answer call over the serialised subgraph |
| `rag` | Baseline: fixed word chunks, dense retrieval with the pipeline's own embedding model, top-k, one answer call |
| `full_context` | Baseline: every document in one prompt, one answer call, no retrieval |

- **Models.** Run A: `gemini-3.5-flash-lite` (free tier) for extraction, dedup, planning and
  answering. Runs B and C: `gpt-oss:20b` on Ollama (16k context, `think: low`), on an RTX 4070
  SUPER with ~80% of the weights in VRAM. Temperature 0, one pass, no repeated seeds.

## Entity resolution

Pairwise F1 against the ground-truth clusters, for the gold mentions (clean input) and for
the mentions the extractor actually produced. Run A, Gemini:

| Method | F1, gold mentions | F1, extracted mentions | Time |
|---|---|---|---|
| **LLM full-context** | **0.902** | 0.718 | 11–22 s |
| Baseline: fuzzy match (token_sort_ratio ≥ 90) | 0.863 | **0.883** | < 0.1 s |
| Baseline: exact match after normalisation | 0.834 | 0.822 | < 0.1 s |
| R-Swoosh | 0.800 | 0.811 | < 0.1 s |
| Semantic (Milvus embeddings) | 0.623 | 0.490 | 5–19 s |
| Splink (probabilistic) | 0.000 | 0.000 | 2–3 s |
| Topological | 0.000 | 0.026 | < 0.1 s |

Findings:

- The LLM method is the best on clean mentions, but a two-line fuzzy-match baseline is close
  behind (0.86) and **beats it on the extractor's real output** (0.88 vs 0.72), where the LLM
  over-merges. R-Swoosh is precise (1.0) but misses a third of the pairs.
- Splink found no duplicates on this corpus: its model never trained useful match weights on
  66 records, which is too few for the probabilistic approach. Topological dedup found almost
  nothing, because mentions from different documents rarely share graph neighbours.
- On `gpt-oss:20b` (run B), the single-call LLM dedup **failed outright**: 73 mentions in one
  prompt produced a reply that hit the 8,192-token output cap, the JSON could not be parsed,
  and zero merges were made. The graph used for QA in run B is therefore unmerged (73 nodes).
  This is fixed in run C by batching (see below).

## Question answering

### Run A: `gemini-3.5-flash-lite`, all 39 questions

| Metric | graphrag | rag | full_context |
|---|---|---|---|
| Score | 0.581 | 0.658 | **0.846** |
| Exact match | 0.564 | 0.641 | 0.846 |
| Abstained (no answer given) | 15 | 11 | 6 |
| Answer present in retrieved context | 0.800 | 0.914 | 1.000 |
| LLM calls / question | 2.0 | 1.0 | 1.0 |
| Input tokens / question | 2,161 | 926 | 2,379 |
| Latency / question | 21.6 s | 12.9 s | 11.0 s |
| lookup (n=8) | 0.625 | 0.875 | 1.000 |
| attribute (n=6) | 0.667 | 0.833 | 1.000 |
| list (n=6) | 1.000 | 1.000 | 1.000 |
| yes/no (n=4) | 0.500 | 0.750 | 1.000 |
| **multi-hop (n=8)** | **0.458** | 0.083 | 0.375 |
| cross-document (n=7) | 0.286 | 0.571 | 0.857 |

Indexing for `graphrag`: 8 extraction calls (18.6k in / 9.4k out tokens) and 1 dedup call
(12.0k in / 2.8k out); the baselines index nothing. One extraction call returned a reply with
a JSON syntax slip that the parser rejected, so the largest document contributed no entities
to the graph in this run. That is one of the fixes below.

### Run B: `gpt-oss:20b`, before the fixes

| Metric | graphrag | rag | full_context |
|---|---|---|---|
| Score | 0.650 | 0.718 | **0.880** |
| Abstained | 10 | 8 | 2 |
| Answer present in retrieved context | 0.743 | 0.914 | 1.000 |
| LLM calls / question | 1.9 | 1.0 | 1.0 |
| Input tokens / question | 3,471 | 949 | 2,278 |
| Latency / question | 10.0 s | 2.8 s | 3.7 s |
| **multi-hop (n=8)** | **0.542** | 0.250 | 0.542 |
| cross-document (n=7) | 0.571 | 0.714 | 1.000 |

### What the first two runs exposed

1. **Extraction failed silently.** A reply that was not strict JSON (run A) or was empty because
   the model spent its output budget reasoning (run B, in a warm-up attempt) was treated as
   "no entities", with only a log line. **Fix:** repair non-strict JSON with `json_repair`,
   retry once with a "return only the JSON object" instruction, and record every document
   whose extraction still fails (`failed_documents`).
2. **The planner gave up instead of falling back.** When intent extraction found no usable
   entities, or the traversal returned nothing, the orchestrator answered `ambiguous` and the
   question went unanswered (10 of 39 in run B). **Fix:** a BM25 index over the parsed
   document text (`src/text_fallback.py`); when the graph has no answer the orchestrator
   returns the top passages with `status: text_fallback`, so the caller still gets cited
   context. Access-control refusals are never bypassed.
3. **Single-call dedup does not scale to a verbose model.** 73 mentions in one prompt exceeded
   `gpt-oss`'s output cap; the reply was truncated and every merge was lost. **Fix:** batch
   the mentions by entity type, reconcile across batches using one representative per
   cluster, union the results, and repair truncated replies.

A fourth, smaller bug surfaced on the way: a laser plan with a missing source or target
entity crashed `_is_intent_empty`; it is now tolerated.

### Run C: `gpt-oss:20b`, after the fixes

Same model and settings as run B. GraphRAG++ was queried over three graphs built from the
same extraction with different deduplication methods; the baselines are unchanged.

| Metric | graphrag (LLM dedup) | graphrag (R-Swoosh) | graphrag (fuzzy) | rag | full_context |
|---|---|---|---|---|---|
| Score | 0.542 | 0.615 | 0.624 | 0.718 | **0.880** |
| Exact match | 0.513 | 0.564 | 0.641 | 0.718 | 0.872 |
| Abstained | 11 | 11 | 8 | 8 | 2 |
| Answer present in retrieved context | 0.771 | 0.800 | 0.771 | 0.914 | 1.000 |
| LLM calls / question | 2.0 | 2.0 | 2.0 | 1.0 | 1.0 |
| Input tokens / question | 2,876 | 2,939 | 2,914 | 949 | 2,278 |
| Latency / question | 9.7 s | 9.2 s | 9.2 s | 2.8 s | 3.7 s |
| Graph size (nodes / edges) | 35 / 58 | 41 / 60 | 33 / 54 | – | – |
| lookup (n=8) | 0.750 | 0.625 | 0.750 | 0.875 | 1.000 |
| attribute (n=6) | 0.500 | 0.667 | 0.667 | 0.833 | 1.000 |
| list (n=6) | 0.967 | 0.944 | 0.944 | 1.000 | 1.000 |
| yes/no (n=4) | 0.000 | 0.500 | 0.500 | 0.750 | 0.750 |
| **multi-hop (n=8)** | 0.417 | **0.667** | 0.583 | 0.250 | 0.542 |
| cross-document (n=7) | 0.429 | 0.286 | 0.286 | 0.714 | 1.000 |

An intermediate run with only the extraction and fallback fixes, querying the *unmerged*
graph from run B (73 nodes, LLM dedup still truncated), is in
`results/local-gpt-oss-20b-fallback-only.json`: score 0.709, 7 abstentions, 5 questions
answered through the text fallback.

What changed, and what did not:

- **The fallback works as intended.** Abstentions fell from 10 to 7 on the unmerged graph,
  evidence recall rose from 0.83 to 0.95, and the questions the graph could not plan for
  were answered from passages instead of "Not found".
- **Batched dedup runs, but merging hurt QA on this model.** The LLM dedup now completes
  (12 calls, 15 clusters, 73 → 35 nodes) but with precision 0.65 on extracted mentions it
  merges entities that are not the same, and every merged graph scores *below* the unmerged
  one (0.54–0.62 vs 0.71). Wrong merges are worse than no merges: they produce confident
  wrong context, which the fallback cannot detect, whereas a missing entity at least
  triggers it. The fuzzy string method, the most precise resolver on this model (0.88), is
  the least damaging, and R-Swoosh's graph gives the best multi-hop result of any system
  (0.67).
- **The gap to the baselines is not closed.** GraphRAG++'s best configuration on gpt-oss
  (unmerged graph with fallback, 0.71) matches basic RAG (0.72) and trails the full-text
  paste (0.88) by a wide margin, at twice the calls and three times the input tokens per
  question. It leads only on multi-hop questions.

The fixes made the pipeline robust: no silent extraction loss, no abstaining without a
fallback, a dedup step that completes on a small model, and a planner that survives a
malformed plan. They did not change the underlying trade-off on this corpus.

## Reading the results

- **On a corpus that fits in one prompt, pasting everything wins.** Both baselines are one
  call; `full_context` scores highest on every question type except multi-hop and costs
  about the same input tokens as one GraphRAG++ query. GraphRAG++ pays for indexing (LLM
  extraction and dedup) that the baselines skip entirely.
- **Multi-hop is where the graph earns its cost.** It is the only type where GraphRAG++ beats
  basic RAG clearly (0.46 vs 0.08 on Gemini; 0.54 vs 0.25 on gpt-oss) and matches or beats
  the full-context paste. Path queries between two named entities are what the laser mode
  is for.
- **The graph is only as good as extraction and resolution.** Every GraphRAG++ loss traces
  back to a document that never entered the graph, a mention that was not merged, or a plan
  the model could not ground. The fixes above address the failure modes the benchmark found;
  they do not change the underlying trade-off.
- The approach would be expected to matter more on corpora too large to paste into one
  prompt, and on questions that need relationships across many documents. This benchmark
  does not test that regime.

## Caveats

- Synthetic corpus, 39 questions, one run per system, temperature 0. Differences of one or
  two questions within a type are noise; the totals and the per-type pattern are the signal.
- Two models, both small or free-tier. No claim is made about larger models.
- Automatic scoring rewards answers that contain the expected strings; it does not judge
  explanation quality or penalise extra content.
- Latencies were measured on a shared machine; the Gemini figures include free-tier pacing.

## Files

- `generate_corpus.py`, `corpus/`, `ground_truth/`: the corpus, its facts and the questions
- `run_benchmark.py`: the harness (`--report results/<run>.json` prints the tables; every LLM
  reply is cached under `.cache/` so reruns repeat no calls)
- `scoring.py`: metrics (tests in `tests/test_benchmark_scoring.py`)
- `results/gemini-3.5-flash-lite.json`, `results/local-gpt-oss-20b.json`,
  `results/local-gpt-oss-20b-fallback-only.json`, `results/local-gpt-oss-20b-fixed.json`:
  raw per-question outputs, including answers, retrieved context and the orchestrator's plan
