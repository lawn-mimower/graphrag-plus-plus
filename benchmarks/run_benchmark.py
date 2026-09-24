#!/usr/bin/env python
"""
Benchmark GraphRAG++ against conventional baselines on the synthetic corpus.

Two benchmarks:

  er  Entity resolution on the 66 gold mentions (what a perfect extractor
      would return). The five dedup methods in src/methodologies are compared
      with exact match after normalisation and rapidfuzz token_sort_ratio >= 90.
      Only the LLM method calls a model.

  qa  Question answering over the 8 documents (39 questions), three systems
      answered by the same LLM:
        graphrag      the GraphRAG++ pipeline (extraction, dedup, SQLite,
                      Milvus Lite, LLM query planner, graph walk) followed by
                      one answer call that sees the returned subgraph
        rag           basic RAG: 100-word chunks, all-mpnet-base-v2 top-5,
                      one answer call
        full_context  all document text in one prompt, one answer call
      The GraphRAG++ indexing run also yields an end-to-end entity-resolution
      score for the five methods on the extracted mentions.

Models are given as provider:model, e.g. gemini:gemini-2.5-flash or
ollama:llama3.2. LLM responses are cached in benchmarks/.cache/llm/, work
files go to benchmarks/.work/<run-name>/ and results to
benchmarks/results/<run-name>.json. When the live-call budget or the API quota
runs out, the partial results are saved together with the command that
resumes the run.

Examples:
    # fully local run
    python benchmarks/run_benchmark.py --run-name local-llama3.2 \\
        --index-model ollama:llama3.2 --plan-model ollama:llama3.2 --answer-model ollama:llama3.2

    # Gemini run: first 6 questions (one per type), at most 34 live calls, 10 s apart
    python benchmarks/run_benchmark.py --run-name gemini --tasks qa,er \\
        --index-model gemini:gemini-3-flash-preview --plan-model gemini:gemini-3-flash-preview \\
        --answer-model gemini:gemini-3.5-flash-lite --max-live-calls 34 --min-interval 10 \\
        --limit-questions 6 --resume

    # print markdown tables for finished result files
    python benchmarks/run_benchmark.py --report benchmarks/results/*.json
"""

import argparse
import contextlib
import json
import logging
import os
import pickle
import platform
import re
import shlex
import statistics
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from benchmarks import scoring  # noqa: E402
from benchmarks.llm import (BudgetExhausted, GenAIShim, LLMBackend, ModelSpec,  # noqa: E402
                            QuotaExhausted, record_to_dict)

log = logging.getLogger("benchmark")

SYSTEMS = ["graphrag", "rag", "full_context"]
DEDUP_METHODS = ["rswoosh", "probabilistic", "topological", "semantic", "llm_full_context"]
ANSWER_TEMPERATURE = 0.0
ANSWER_MAX_TOKENS = 2048
USER_TAGS = ["UNCLASSIFIED"]

ANSWER_PROMPT = """You answer questions about a small set of company documents: annual reports, board minutes, schedules and letters. Use only the context below.

Context:
{context}

Question: {question}

Reply with the answer only, without explanation: a name, a list of names separated by semicolons, a number, or Yes/No. For a question of the form "How is X connected to Y?", name the entity or entities that link them. If the context does not contain the answer, reply "Not found".
Answer:"""


# --------------------------------------------------------------------------
# Run context
# --------------------------------------------------------------------------
class Halt(Exception):
    """Stop the run and save partial results."""


class Run:
    def __init__(self, args):
        self.args = args
        self.work = BENCH_DIR / ".work" / args.run_name
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work / "state.json"
        self.results_path = BENCH_DIR / "results" / f"{args.run_name}.json"
        self.truth = load_truth()
        self.index = scoring.AliasIndex(self.truth["entities"])
        self.backend = LLMBackend(
            cache_dir=BENCH_DIR / ".cache" / "llm",
            max_live_calls=args.max_live_calls,
            min_interval_s=args.min_interval,
            ollama_url=args.ollama_url,
            ollama_num_ctx=args.num_ctx,
            ollama_max_predict=args.max_predict,
            ledger_path=self.work / "live_calls.tsv",
            gemini_attempts=args.gemini_attempts,
        )
        self.index_spec = ModelSpec.parse(args.index_model)
        self.plan_spec = ModelSpec.parse(args.plan_model)
        self.answer_spec = ModelSpec.parse(args.answer_model)
        self.er_spec = ModelSpec.parse(args.er_llm_model or args.index_model)
        self.state = {}
        if args.resume and self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            log.info("Resuming from %s", self.state_path.relative_to(REPO_ROOT))
        self.state.setdefault("units", {})
        self._parsed = None
        self._configure_outputs()

    def _configure_outputs(self):
        from src.config import Config

        outputs = self.work / "outputs"
        for name, sub in [("OUTPUTS_DIR", ""), ("PARSED_DOCS_DIR", "parsed_documents"),
                          ("ENTITIES_EXTRACTED_DIR", "entities_extracted"),
                          ("KNOWLEDGE_GRAPHS_DIR", "knowledge_graphs"), ("REASONING_DIR", "reasoning")]:
            path = outputs / sub if sub else outputs
            path.mkdir(parents=True, exist_ok=True)
            setattr(Config, name, path)

    def save_state(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(self.state_path)

    def check_halt(self):
        if self.backend.halted is not None:
            raise Halt(str(self.backend.halted))

    def records_since(self, start):
        return [record_to_dict(r) for r in self.backend.records[start:]]

    # -- corpus ---------------------------------------------------------------
    def corpus_files(self):
        return [BENCH_DIR / "corpus" / d["file"] for d in self.truth["documents"]]

    def parsed(self):
        """Parse every document once with the pipeline's own parser."""
        if self._parsed is None:
            from src.rapidocr_parser import RapidOCRParser

            parser = RapidOCRParser()
            start = time.monotonic()
            self._parsed = {f.stem: parser.parse(f) for f in self.corpus_files()}
            self.parse_seconds = time.monotonic() - start
        return self._parsed

    def baseline_texts(self):
        """Parsed page text per document, runs of spaces collapsed (XLSX tables are padded)."""
        texts = {}
        for doc, pages in self.parsed().items():
            lines = []
            for _, text, _ in pages:
                for line in (text or "").splitlines():
                    line = re.sub(r"[ \t]+", " ", line).strip()
                    if line:
                        lines.append(line)
            texts[doc] = "\n".join(lines)
        return texts


def load_truth():
    gt = BENCH_DIR / "ground_truth"

    def read(name):
        return json.loads((gt / name).read_text(encoding="utf-8"))

    return {
        "entities": read("entities.json")["entities"],
        "mentions": read("mentions.json")["mentions"],
        "mention_relations": read("mentions.json")["relations"],
        "relations": read("relations.json")["relations"],
        "questions": read("questions.json")["questions"],
        "documents": read("documents.json")["documents"],
    }


@contextlib.contextmanager
def patched_genai_client(shim):
    """QueryOrchestrator builds its own google-genai client; hand it the shim instead."""
    from google import genai

    original = genai.Client
    genai.Client = lambda *a, **k: shim
    try:
        yield
    finally:
        genai.Client = original


# --------------------------------------------------------------------------
# Entity resolution
# --------------------------------------------------------------------------
def build_gold_mention_graph(truth):
    """Mention graph as a perfect extractor would produce it: one node per gold mention."""
    from src.kg_builder import KnowledgeGraphBuilder

    builder = KnowledgeGraphBuilder()
    node_to_mention = {}
    for doc in truth["documents"]:
        name = doc["name"]
        gold = gold_extraction(truth, name)
        before = set(builder.graph.nodes)
        builder.add_entities(gold["entities"], gold["relationships"], source_doc=name)
        for node in set(builder.graph.nodes) - before:
            node_to_mention[node] = f"{name}:{builder.graph.nodes[node]['original_id']}"
    return builder, node_to_mention


def gold_extraction(truth, doc):
    """Extraction result a perfect extractor would return for one document."""
    mentions = [m for m in truth["mentions"] if m["document"] == doc]
    local = {m["mention_id"]: m["extractor_id"] for m in mentions}
    return {
        "entities": [{"id": m["extractor_id"], "type": m["type"], "attributes": dict(m["attributes"])}
                     for m in mentions],
        "relationships": [{"from_id": local[r["from"]], "to_id": local[r["to"]], "type": r["type"],
                           "attributes": dict(r["attributes"])}
                          for r in truth["mention_relations"] if r["document"] == doc],
    }


def run_dedup_methods(graph, run, llm_spec, purpose, methods=DEDUP_METHODS):
    """Run the pipeline's dedup methods the way BenchmarkHarness does. Returns clusters of node IDs."""
    from src.methodologies.llm_full_context import LLMFullContextDeduplicator
    from src.methodologies.probabilistic import ProbabilisticDeduplicator
    from src.methodologies.rswoosh import RSwooshDeduplicator
    from src.methodologies.semantic import SemanticDeduplicator
    from src.methodologies.topological import TopologicalDeduplicator

    out = {}
    for name in methods:
        start = time.monotonic()
        rec_start = len(run.backend.records)
        entry = {}
        try:
            if name == "llm_full_context":
                dedup = LLMFullContextDeduplicator(GenAIShim(run.backend, llm_spec, purpose))
                dedup.model_name = llm_spec.model
                details = dedup.deduplicate_with_details(graph)
                clusters = [set(g.get("entities", [])) for g in details.get("duplicates", [])
                            if len(g.get("entities", [])) > 1]
                entry["reasoning"] = details
            else:
                cls = {"rswoosh": RSwooshDeduplicator, "probabilistic": ProbabilisticDeduplicator,
                       "topological": TopologicalDeduplicator, "semantic": SemanticDeduplicator}[name]
                clusters = cls().deduplicate(graph)
        except Exception as exc:  # the harness also carries on after a failed method
            log.exception("dedup method %s failed", name)
            clusters, entry["error"] = [], str(exc)
        run.check_halt()
        unknown = sorted({n for c in clusters for n in c if n not in graph.nodes})
        entry.update({
            "clusters": [sorted(c) for c in clusters],
            "seconds": time.monotonic() - start,
            "unknown_ids": unknown,
            "llm": run.records_since(rec_start),
        })
        out[name] = entry
    return out


def baseline_clusters(names):
    return {
        "exact_normalised": scoring.exact_match_clusters(names),
        "fuzzy_token_sort_90": scoring.fuzzy_match_clusters(names, threshold=90.0),
    }


def score_clusters(method_results, node_to_item, gold):
    """Resolution scores for each method, clusters mapped from graph nodes to scored items."""
    scores = {}
    for name, res in method_results.items():
        clusters = [[node_to_item[n] for n in c if n in node_to_item] for c in res["clusters"]]
        s = scoring.resolution_scores(clusters, gold)
        s["seconds"] = res.get("seconds", 0.0)
        if res.get("error"):
            s["error"] = res["error"]
        if res.get("unknown_ids"):
            s["unknown_ids"] = len(res["unknown_ids"])
        llm = res.get("llm") or []
        s["llm_calls"] = len(llm)
        s["input_tokens"] = sum(r["input_tokens"] for r in llm)
        s["output_tokens"] = sum(r["output_tokens"] + r["thinking_tokens"] for r in llm)
        scores[name] = s
    return scores


def task_er(run):
    if "er_gold" in run.state:
        log.info("ER on gold mentions: already done")
        return
    log.info("ER on gold mentions with %s for the LLM method", run.er_spec)
    builder, node_to_mention = build_gold_mention_graph(run.truth)
    graph = builder.graph
    gold = {m["mention_id"]: m["entity_id"] for m in run.truth["mentions"]}
    results = run_dedup_methods(graph, run, run.er_spec, "er.llm_dedup")
    names = {n: graph.nodes[n].get("name", "") for n in graph.nodes}
    for bname, clusters in baseline_clusters(names).items():
        start = time.monotonic()
        results[bname] = {"clusters": [sorted(c) for c in clusters], "seconds": time.monotonic() - start}
    run.state["er_gold"] = {
        "llm_model": str(run.er_spec),
        "mentions": graph.number_of_nodes(),
        "entities": len(set(gold.values())),
        "scores": score_clusters(results, node_to_mention, gold),
        "clusters": {k: v["clusters"] for k, v in results.items()},
    }
    run.save_state()


# --------------------------------------------------------------------------
# GraphRAG++ indexing
# --------------------------------------------------------------------------
def build_graphrag_index(run):
    """Parse -> extract -> mention graph -> five dedup methods -> SQLite + Milvus for chosen graphs."""
    from src.entity_extractor import MultimodalEntityExtractor
    from src.kg_builder import KnowledgeGraphBuilder
    from src.token_manager import TokenManager
    import kg_to_sql
    from src.milvus_ingestion import MilvusIngestionEngine

    idx = run.state.get("graphrag_index", {})
    graphs_dir = run.work / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    raw_path = graphs_dir / "non_dedup.gpickle"

    if not idx.get("extraction_done"):
        parsed = run.parsed()
        shim = GenAIShim(run.backend, run.index_spec, "index.extraction")
        extractor = MultimodalEntityExtractor(shim, TokenManager(shim))
        extractor.model_name = run.index_spec.model
        from src.config import Config

        builder = KnowledgeGraphBuilder()
        per_doc = {}
        start = time.monotonic()
        rec_start = len(run.backend.records)
        for path in run.corpus_files():
            pages = parsed[path.stem]
            if run.args.gold_extraction:
                # Diagnostic: replace the LLM extraction by the gold mentions of this document.
                result = gold_extraction(run.truth, path.stem)
            else:
                result = extractor.extract_entities(pages, path.stem,
                                                    Config.ENTITIES_EXTRACTED_DIR / f"{path.stem}_entities.json")
            run.check_halt()
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])
            # Same page-number handling as BenchmarkHarness._process_files_iteratively
            page_numbers = []
            for entity in entities:
                attrs = entity.get("attributes", {}) or {}
                if "page_number" in attrs:
                    page_numbers.append(attrs["page_number"])
                pages_attr = attrs.get("page_numbers")
                if isinstance(pages_attr, list):
                    page_numbers.extend(p for p in pages_attr if isinstance(p, int))
            page_numbers = sorted(set(p for p in page_numbers if isinstance(p, int)))
            if entities:
                builder.add_entities(entities, relationships, source_doc=path.stem,
                                     page_numbers=page_numbers or None)
            per_doc[path.stem] = {"entities": len(entities), "relationships": len(relationships)}
        with open(raw_path, "wb") as f:
            pickle.dump(builder.graph, f)
        idx.update({
            "extraction_done": True,
            "parse_seconds": getattr(run, "parse_seconds", 0.0),
            "extraction_seconds": time.monotonic() - start,
            "extraction_llm": run.records_since(rec_start),
            "per_document": per_doc,
            "mention_nodes": builder.graph.number_of_nodes(),
            "mention_edges": builder.graph.number_of_edges(),
        })
        run.state["graphrag_index"] = idx
        run.save_state()

    with open(raw_path, "rb") as f:
        raw_graph = pickle.load(f)
    builder = KnowledgeGraphBuilder()
    builder.graph = raw_graph

    if not idx.get("dedup_done"):
        results = run_dedup_methods(raw_graph, run, run.index_spec, "index.llm_dedup")
        idx["dedup"] = {k: {kk: vv for kk, vv in v.items() if kk != "reasoning"} for k, v in results.items()}
        idx["dedup_done"] = True
        run.state["graphrag_index"] = idx
        run.save_state()

    # End-to-end entity resolution on the extracted mentions
    if "er_extracted" not in run.state:
        aligned = {}
        for node, attrs in raw_graph.nodes(data=True):
            eid = run.index.align(str(attrs.get("name", "")))
            if eid:
                aligned[node] = eid
        results = dict(idx["dedup"])
        names = {n: str(raw_graph.nodes[n].get("name", "")) for n in raw_graph.nodes}
        for bname, clusters in baseline_clusters(names).items():
            results[bname] = {"clusters": [sorted(c) for c in clusters], "seconds": 0.0}
        run.state["er_extracted"] = {
            "extracted_nodes": raw_graph.number_of_nodes(),
            "aligned_nodes": len(aligned),
            "entities_covered": len(set(aligned.values())),
            "scores": score_clusters(results, {n: n for n in raw_graph.nodes}, aligned),
        }
        run.save_state()

    idx.setdefault("graphs", {})
    for gname in run.args.graphrag_graphs:
        if gname in idx["graphs"] and (run.work / f"{gname}.db").exists():
            continue
        # IDs the LLM invented are dropped: create_deduplicated_graph raises KeyError on them.
        clusters = [{n for n in c if n in raw_graph.nodes} for c in idx["dedup"][gname]["clusters"]]
        clusters = [c for c in clusters if len(c) > 1]
        start = time.monotonic()
        dedup_graph = builder.create_deduplicated_graph(clusters)
        gpath = graphs_dir / f"dedup_{gname}.gpickle"
        with open(gpath, "wb") as f:
            pickle.dump(dedup_graph, f)
        db_path = run.work / f"{gname}.db"
        if db_path.exists():
            db_path.unlink()
        kg_to_sql.convert_graph_to_sql(gpath, db_path)
        sql_seconds = time.monotonic() - start
        milvus_path = run.work / f"{gname}_milvus.db"
        if milvus_path.exists():
            milvus_path.unlink()
        start = time.monotonic()
        engine = MilvusIngestionEngine(db_path=str(db_path), milvus_path=str(milvus_path))
        stats = engine.ingest_from_sql()
        engine.close()
        idx["graphs"][gname] = {
            "nodes": dedup_graph.number_of_nodes(), "edges": dedup_graph.number_of_edges(),
            "sql_seconds": sql_seconds, "milvus_seconds": time.monotonic() - start,
            "milvus_entities": stats.get("ingested", 0),
        }
        run.state["graphrag_index"] = idx
        run.save_state()
    _quiet_logs()
    return idx


# --------------------------------------------------------------------------
# Question answering
# --------------------------------------------------------------------------
def serialize_graph_context(result):
    """Render the orchestrator's output (nodes, edges, citations) as plain text for the answer call."""
    lines = ["Entities:"]
    for node in result.get("nodes", {}).values():
        attrs = {k: v for k, v in (node.get("attributes") or {}).items() if v not in (None, "", [], {})}
        attr_text = "; ".join(f"{k}={v}" for k, v in attrs.items())
        lines.append(f"- {node.get('name')} ({node.get('type')})" + (f": {attr_text}" if attr_text else ""))
    lines.append("Relationships:")
    for edge in result.get("mini_graph", []):
        props = {k: v for k, v in (edge.get("properties") or {}).items() if v not in (None, "", [], {})}
        prop_text = " (" + "; ".join(f"{k}={v}" for k, v in props.items()) + ")" if props else ""
        lines.append(f"- {edge.get('source_name')} -[{edge.get('relation')}]-> {edge.get('target_name')}{prop_text}")
    citations = result.get("citation_context") or {}
    if citations:
        lines.append("Sources: " + "; ".join(f"{doc} p. {', '.join(map(str, pages))}"
                                              for doc, pages in sorted(citations.items())))
    return "\n".join(lines)


class RagIndex:
    """Basic RAG baseline: fixed word chunks, dense retrieval with the pipeline's embedding model."""

    def __init__(self, texts, size, overlap, model_name):
        from sentence_transformers import SentenceTransformer

        start = time.monotonic()
        self.chunks = []
        for doc, text in texts.items():
            for chunk in scoring.chunk_words(text, size, overlap):
                self.chunks.append({"doc": doc, "text": chunk})
        self.chunk_seconds = time.monotonic() - start
        start = time.monotonic()
        self.model = SentenceTransformer(model_name)
        self.load_seconds = time.monotonic() - start
        start = time.monotonic()
        self.vectors = self.model.encode([c["text"] for c in self.chunks], normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False)
        self.embed_seconds = time.monotonic() - start
        self.bm25 = scoring.BM25([scoring.bm25_tokens(c["text"]) for c in self.chunks])

    def dense_top_k(self, query, k):
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.vectors @ q
        return sorted(range(len(self.chunks)), key=lambda i: (-float(sims[i]), i))[:k]

    def bm25_top_k(self, query, k):
        return self.bm25.top_k(scoring.bm25_tokens(query), k)

    def context(self, ids):
        return "\n\n".join(f"[{n}] (from {self.chunks[i]['doc']}) {self.chunks[i]['text']}"
                           for n, i in enumerate(ids, 1))


def answer_call(run, context, question):
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    text, _ = run.backend.generate(run.answer_spec, prompt, temperature=ANSWER_TEMPERATURE,
                                   max_output_tokens=ANSWER_MAX_TOKENS, purpose="query.answer")
    return text.strip()


def unit_llm(records):
    return {
        "calls": len(records),
        "live_calls": sum(not r["cached"] for r in records),
        "input_tokens": sum(r["input_tokens"] for r in records),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "thinking_tokens": sum(r["thinking_tokens"] for r in records),
        "llm_latency_s": sum(r["latency_s"] for r in records),
    }


def run_unit(run, system, q, orchestrators, rag, full_text):
    rec_start = len(run.backend.records)
    start = time.monotonic()
    record = {"id": q["id"], "type": q["type"], "system": system}
    context = ""
    if system.startswith("graphrag"):
        gname = system.split(":", 1)[1]
        result = orchestrators[gname].process_query(q["question"], USER_TAGS, max_depth=run.args.max_depth)
        run.check_halt()
        nodes = result.get("nodes") or {}
        edges = result.get("mini_graph") or []
        record["graph"] = {
            "status": result.get("status"),
            "mode": (result.get("intent") or {}).get("query_mode"),
            "nodes": len(nodes), "edges": len(edges),
            **scoring.subgraph_coverage(q, [n.get("name", "") for n in nodes.values()],
                                        [(e.get("source_name", ""), e.get("target_name", "")) for e in edges],
                                        run.index),
        }
        if nodes:
            context = serialize_graph_context(result)
    elif system == "rag":
        ids = rag.dense_top_k(q["question"], run.args.top_k)
        bm25_ids = rag.bm25_top_k(q["question"], run.args.top_k)
        context = rag.context(ids)
        record["retrieval"] = {
            "chunks": [f"{rag.chunks[i]['doc']}#{i}" for i in ids],
            "bm25_same_k": scoring.context_coverage(q, rag.context(bm25_ids), run.index),
        }
    elif system == "full_context":
        context = full_text
    else:
        raise ValueError(system)

    if context:
        answer = answer_call(run, context, q["question"])
    else:
        answer = "Not found"  # nothing retrieved: no answer call
        record["skipped_answer_call"] = True
    run.check_halt()
    measured = time.monotonic() - start
    records = run.records_since(rec_start)
    live_llm = sum(r["latency_s"] for r in records if not r["cached"])
    record.update({
        "answer": answer,
        "score": scoring.score_answer(q, answer, run.index),
        "context": {"chars": len(context), **scoring.context_coverage(q, context, run.index)},
        "llm": unit_llm(records),
        "llm_records": records,
        # wall time with cached LLM calls counted at their recorded latency
        "latency_s": max(measured - live_llm, 0.0) + sum(r["latency_s"] for r in records),
    })
    return record


def task_qa(run):
    args = run.args
    systems = []
    for s in args.systems:
        if s == "graphrag":
            systems.extend(f"graphrag:{g}" for g in args.graphrag_graphs)
        else:
            systems.append(s)
    questions = sorted(run.truth["questions"], key=lambda q: (q["priority"], q["id"]))
    if args.questions != "all":
        wanted = set(args.questions.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit_questions:
        questions = questions[:args.limit_questions]

    orchestrators = {}
    if any(s.startswith("graphrag") for s in systems):
        build_graphrag_index(run)
        from src.orchestrator import QueryOrchestrator

        shim = GenAIShim(run.backend, run.plan_spec, "query.plan")
        for gname in args.graphrag_graphs:
            with patched_genai_client(shim):
                orchestrators[gname] = QueryOrchestrator(
                    db_path=str(run.work / f"{gname}.db"),
                    milvus_path=str(run.work / f"{gname}_milvus.db"),
                    gemini_model=run.plan_spec.model)
    _quiet_logs()

    texts = run.baseline_texts()
    rag = None
    if "rag" in systems:
        from src.config import Config

        rag = RagIndex(texts, args.chunk_words, args.chunk_overlap, Config.SEMANTIC_EMBEDDING_MODEL)
        run.state["rag_index"] = {"chunks": len(rag.chunks), "chunk_words": args.chunk_words,
                                  "chunk_overlap": args.chunk_overlap, "top_k": args.top_k,
                                  "embedding_model": Config.SEMANTIC_EMBEDDING_MODEL,
                                  "embed_seconds": rag.embed_seconds, "model_load_seconds": rag.load_seconds}
    full_text = "\n\n".join(f"=== Document: {doc} ===\n{text}" for doc, text in texts.items())
    run.state["full_context"] = {"chars": len(full_text), "words": len(full_text.split())}

    run.state["qa_systems"] = systems
    run.state["qa_questions"] = [q["id"] for q in questions]
    try:
        for q in questions:
            for system in systems:
                key = f"{q['id']}|{system}"
                if key in run.state["units"]:
                    continue
                log.info("%s %-26s %s", q["id"], system, q["question"])
                record = run_unit(run, system, q, orchestrators, rag, full_text)
                log.info("    -> %s  [score %.2f]", record["answer"][:100].replace("\n", " "),
                         record["score"]["score"])
                run.state["units"][key] = record
                run.save_state()
    finally:
        for orch in orchestrators.values():
            orch.close()


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def aggregate_qa(state, truth, index):
    systems = state.get("qa_systems") or sorted({k.split("|", 1)[1] for k in state["units"]})
    qids = state.get("qa_questions") or sorted({k.split("|", 1)[0] for k in state["units"]})
    # Only questions every system finished are compared.
    complete = [qid for qid in qids if all(f"{qid}|{s}" in state["units"] for s in systems)]
    qtypes = {q["id"]: q["type"] for q in truth["questions"]}
    min_docs = {q["id"]: q.get("min_documents") for q in truth["questions"]}
    out = {"questions_planned": len(qids), "questions_scored": len(complete), "question_ids": complete,
           "systems": {}}
    # Re-score the stored answers with the current scoring code.
    by_id = {q["id"]: q for q in truth["questions"]}
    for key, rec in state["units"].items():
        rec["score"] = scoring.score_answer(by_id[rec["id"]], rec["answer"], index)
    for s in systems:
        recs = [state["units"][f"{qid}|{s}"] for qid in complete]
        if not recs:
            continue
        by_type = {}
        for t in sorted({r["type"] for r in recs}):
            tr = [r for r in recs if r["type"] == t]
            by_type[t] = {"n": len(tr), "score": _r(_mean(r["score"]["score"] for r in tr))}
        by_docs = {}
        for k in sorted({min_docs[r["id"]] for r in recs if min_docs[r["id"]]}):
            dr = [r for r in recs if min_docs[r["id"]] == k]
            by_docs[str(k)] = {"n": len(dr), "score": _r(_mean(r["score"]["score"] for r in dr))}
        entity_recs = [r for r in recs if r["score"].get("f1") is not None]
        summary = {
            "n": len(recs),
            "score": _r(_mean(r["score"]["score"] for r in recs)),
            "lenient_correct": _r(_mean(float(r["score"]["correct"]) for r in recs)),
            "exact_match": _r(_mean(float(r["score"]["em"]) for r in recs)),
            "entity_f1": _r(_mean(r["score"]["f1"] for r in entity_recs)),
            "abstained": sum(r["score"]["abstained"] for r in recs),
            "by_type": by_type,
            "by_min_documents": by_docs,
            "answer_in_context": _r(_mean(float(r["context"]["answer_in_context"]) for r in recs
                                          if r["context"]["answer_in_context"] is not None)),
            "evidence_recall": _r(_mean(r["context"]["evidence_recall"] for r in recs)),
            "mean_context_chars": round(_mean(r["context"]["chars"] for r in recs)),
            "llm_calls": sum(r["llm"]["calls"] for r in recs),
            "mean_input_tokens": round(_mean(r["llm"]["input_tokens"] for r in recs)),
            "mean_output_tokens": round(_mean(r["llm"]["output_tokens"] + r["llm"]["thinking_tokens"]
                                              for r in recs)),
            "mean_latency_s": _r(_mean(r["latency_s"] for r in recs), 1),
            "median_latency_s": _r(statistics.median(r["latency_s"] for r in recs), 1),
        }
        if s.startswith("graphrag"):
            g = [r["graph"] for r in recs]
            summary["graph"] = {
                "status": {k: sum(x["status"] == k for x in g) for k in sorted({x["status"] for x in g})},
                "laser_plans": sum(x["mode"] == "laser" for x in g),
                "answer_entities_in_subgraph": _r(_mean(float(x["answer_entities_in_subgraph"]) for x in g
                                                        if x["answer_entities_in_subgraph"] is not None)),
                "path_hit": _r(_mean(float(x["path_hit"]) for x in g if x["path_hit"] is not None)),
                "evidence_recall": _r(_mean(x["evidence_recall"] for x in g)),
                "mean_nodes": round(_mean(x["nodes"] for x in g)),
                "answer_calls_skipped": sum(bool(r.get("skipped_answer_call")) for r in recs),
            }
        if s == "rag":
            b = [r["retrieval"]["bm25_same_k"] for r in recs]
            summary["bm25_same_k"] = {
                "answer_in_context": _r(_mean(float(x["answer_in_context"]) for x in b
                                              if x["answer_in_context"] is not None)),
                "evidence_recall": _r(_mean(x["evidence_recall"] for x in b)),
            }
        out["systems"][s] = summary
    out["per_question"] = [
        {"id": qid, "type": qtypes[qid],
         **{s: {"answer": state["units"][f"{qid}|{s}"]["answer"],
                "score": _r(state["units"][f"{qid}|{s}"]["score"]["score"])} for s in systems}}
        for qid in complete]
    return out


def indexing_summary(state):
    out = {}
    idx = state.get("graphrag_index")
    if idx:
        ext = idx.get("extraction_llm", [])
        dedup = idx.get("dedup", {})
        llm_dedup = dedup.get("llm_full_context", {}).get("llm", [])
        out["graphrag"] = {
            "extraction": {"llm_calls": len(ext),
                           "input_tokens": sum(r["input_tokens"] for r in ext),
                           "output_tokens": sum(r["output_tokens"] + r["thinking_tokens"] for r in ext),
                           "llm_latency_s": _r(sum(r["latency_s"] for r in ext), 1),
                           "per_document": idx.get("per_document")},
            "mention_nodes": idx.get("mention_nodes"), "mention_edges": idx.get("mention_edges"),
            "dedup_seconds": {k: _r(v.get("seconds"), 1) for k, v in dedup.items()},
            "llm_dedup": {"llm_calls": len(llm_dedup),
                          "input_tokens": sum(r["input_tokens"] for r in llm_dedup),
                          "output_tokens": sum(r["output_tokens"] + r["thinking_tokens"] for r in llm_dedup),
                          "llm_latency_s": _r(sum(r["latency_s"] for r in llm_dedup), 1),
                          "clusters": len(dedup.get("llm_full_context", {}).get("clusters", [])),
                          "unknown_ids": len(dedup.get("llm_full_context", {}).get("unknown_ids", []))},
            "graphs": {k: {kk: (_r(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                       for k, v in idx.get("graphs", {}).items()},
        }
    if state.get("rag_index"):
        r = dict(state["rag_index"])
        r["embed_seconds"] = _r(r.get("embed_seconds"), 1)
        r["model_load_seconds"] = _r(r.get("model_load_seconds"), 1)
        out["rag"] = r
    if state.get("full_context"):
        out["full_context"] = {"llm_calls": 0, **state["full_context"]}
    return out


def _round_scores(scores):
    return {m: {k: (_r(v, 1) if k == "seconds" else _r(v) if isinstance(v, float) else v)
                for k, v in s.items()} for m, s in scores.items()}


def live_call_ledger(run):
    path = run.work / "live_calls.tsv"
    if not path.exists():
        return {}
    counts = {}
    for line in path.read_text().splitlines():
        _, model, status = line.split("\t", 2)
        counts.setdefault(model, {}).setdefault(status, 0)
        counts[model][status] += 1
    return counts


def resume_command():
    argv = [a for a in sys.argv[1:] if a != "--resume"]
    return "python benchmarks/run_benchmark.py " + " ".join(shlex.quote(a) for a in argv) + " --resume"


def write_results(run, status, message=None, resume_cmd=None):
    import importlib.metadata as md

    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    state = run.state
    args = run.args
    results = {
        "run_name": args.run_name,
        "status": status,
        "message": message,
        "updated": time.strftime("%Y-%m-%d"),
        "resume_command": (resume_cmd or resume_command()) if status != "complete" else None,
        "conditions": {
            "index_model": args.index_model, "plan_model": args.plan_model,
            "answer_model": args.answer_model, "er_llm_model": str(run.er_spec),
            "answer_prompt": ANSWER_PROMPT, "answer_temperature": ANSWER_TEMPERATURE,
            "answer_max_output_tokens": ANSWER_MAX_TOKENS,
            "pipeline_temperatures": "0.1 for extraction, LLM dedup and query planning (set in src/)",
            "ollama": ({"num_ctx": args.num_ctx, "num_predict_cap": args.max_predict, "seed": 0,
                        "images": "dropped (text-only model)"}
                       if "ollama" in (args.index_model + args.plan_model + args.answer_model
                                       + str(run.er_spec)) else None),
            "gemini_min_interval_s": args.min_interval,
            "max_live_calls": args.max_live_calls,
            "systems": args.systems, "graphrag_graphs": args.graphrag_graphs,
            "graphrag_extraction": "gold mentions (diagnostic)" if args.gold_extraction else "LLM",
            "graphrag_max_depth": args.max_depth, "user_tags": USER_TAGS,
            "rag": {"chunk_words": args.chunk_words, "chunk_overlap": args.chunk_overlap, "top_k": args.top_k,
                    "retriever": "dense, sentence-transformers/all-mpnet-base-v2, cosine"},
            "platform": {"python": platform.python_version(), "machine": platform.machine(),
                         "packages": {p: ver(p) for p in ["google-genai", "sentence-transformers", "pymilvus",
                                                          "milvus-lite", "splink", "rapidfuzz", "torch"]}},
            "hardware_note": "CPU only; the machine was shared with other jobs, so latencies are noisy.",
        },
        "corpus": {"documents": len(run.truth["documents"]), "entities": len(run.truth["entities"]),
                   "gold_mentions": len(run.truth["mentions"]), "questions": len(run.truth["questions"])},
    }
    if "er_gold" in state:
        results["er_gold_mentions"] = {k: v for k, v in state["er_gold"].items() if k != "clusters"}
        results["er_gold_mentions"]["scores"] = _round_scores(state["er_gold"]["scores"])
    if "er_extracted" in state:
        results["er_extracted_mentions"] = dict(state["er_extracted"])
        results["er_extracted_mentions"]["scores"] = _round_scores(state["er_extracted"]["scores"])
    if state.get("units"):
        results["qa"] = aggregate_qa(state, run.truth, run.index)
    results["indexing"] = indexing_summary(state)
    results["live_api_calls"] = live_call_ledger(run)
    run.results_path.parent.mkdir(parents=True, exist_ok=True)
    run.results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("Wrote %s (%s)", run.results_path.relative_to(REPO_ROOT), status)
    return results


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------
def _fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def report(paths):
    method_label = {"rswoosh": "R-Swoosh", "probabilistic": "Splink", "topological": "Topological",
                    "semantic": "Semantic (Milvus)", "llm_full_context": "LLM full-context",
                    "exact_normalised": "Baseline: exact after normalisation",
                    "fuzzy_token_sort_90": "Baseline: fuzzy token_sort_ratio ≥ 90"}
    for path in paths:
        res = json.loads(Path(path).read_text())
        c = res["conditions"]
        print(f"## {res['run_name']} ({res['status']})\n")
        print(f"index {c['index_model']} · plan {c['plan_model']} · answer {c['answer_model']} · "
              f"ER LLM {c['er_llm_model']}\n")
        for key, title in [("er_gold_mentions", "Entity resolution, gold mentions"),
                           ("er_extracted_mentions", "Entity resolution, extracted mentions")]:
            if key not in res:
                continue
            er = res[key]
            print(f"### {title}\n")
            print("| Method | Pairwise P | Pairwise R | Pairwise F1 | Purity | Inverse purity | Clusters | Time (s) |")
            print("|---|---|---|---|---|---|---|---|")
            for m, s in er["scores"].items():
                print(f"| {method_label.get(m, m)} | {_fmt(s['precision'])} | {_fmt(s['recall'])} | "
                      f"{_fmt(s['f1'])} | {_fmt(s['purity'])} | {_fmt(s['inverse_purity'])} | "
                      f"{s['predicted_clusters']} / {s['gold_clusters']} | {_fmt(s['seconds'], 1)} |")
            print()
        qa = res.get("qa")
        if qa:
            print(f"### QA ({qa['questions_scored']} of {qa['questions_planned']} questions scored)\n")
            systems = list(qa["systems"])
            print("| Metric | " + " | ".join(systems) + " |")
            print("|---|" + "---|" * len(systems))
            rows = [("Score (F1 for entity answers, else 0/1)", "score"),
                    ("Lenient (gold named at all)", "lenient_correct"), ("Exact match", "exact_match"),
                    ("Entity-set F1", "entity_f1"), ("Abstained", "abstained"),
                    ("Answer in context", "answer_in_context"), ("Evidence recall", "evidence_recall"),
                    ("LLM calls", "llm_calls"), ("Mean input tokens", "mean_input_tokens"),
                    ("Mean output tokens", "mean_output_tokens"), ("Mean latency (s)", "mean_latency_s")]
            for label, key in rows:
                nd = 1 if "latency" in key else 3
                print(f"| {label} | " + " | ".join(_fmt(qa["systems"][s].get(key), nd) for s in systems) + " |")
            types = sorted({t for s in systems for t in qa["systems"][s]["by_type"]})
            for t in types:
                cells = []
                for s in systems:
                    bt = qa["systems"][s]["by_type"].get(t)
                    cells.append(f"{_fmt(bt['score'])} (n={bt['n']})" if bt else "–")
                print(f"| {t} | " + " | ".join(cells) + " |")
            for k in sorted({k for s in systems for k in qa["systems"][s].get("by_min_documents", {})}):
                cells = []
                for s in systems:
                    bd = qa["systems"][s].get("by_min_documents", {}).get(k)
                    cells.append(f"{_fmt(bd['score'])} (n={bd['n']})" if bd else "–")
                print(f"| needs {k} document(s) | " + " | ".join(cells) + " |")
            print()
            for s in systems:
                if "graph" in qa["systems"][s]:
                    print(f"{s}: {json.dumps(qa['systems'][s]['graph'])}\n")
        print("Indexing: " + json.dumps(res.get("indexing", {}), indent=1) + "\n")
        print("Live API calls: " + json.dumps(res.get("live_api_calls", {})) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _quiet_logs():
    for name in ["src", "kg_to_sql", "httpx", "google", "sentence_transformers", "pymilvus", "splink",
                 "numexpr", "urllib3"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    log.setLevel(logging.INFO)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="local-llama3.2")
    p.add_argument("--tasks", default="er,qa", help="comma list of er, qa")
    p.add_argument("--index-model", default="ollama:llama3.2", help="extraction and LLM dedup in the QA index")
    p.add_argument("--plan-model", default="ollama:llama3.2", help="GraphRAG++ query planner")
    p.add_argument("--answer-model", default="ollama:llama3.2", help="answering model for every system")
    p.add_argument("--er-llm-model", default=None, help="LLM dedup on gold mentions (default: --index-model)")
    p.add_argument("--systems", default=",".join(SYSTEMS))
    p.add_argument("--graphrag-graphs", default="llm_full_context",
                   help="dedup graphs to query, comma list of " + ", ".join(DEDUP_METHODS))
    p.add_argument("--questions", default="all", help="'all' or comma list of question IDs")
    p.add_argument("--limit-questions", type=int, default=None,
                   help="only the first N questions in priority order (stratified by type)")
    p.add_argument("--gold-extraction", action="store_true",
                   help="diagnostic: feed the gold mentions to GraphRAG++ instead of LLM extraction")
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--chunk-words", type=int, default=100)
    p.add_argument("--chunk-overlap", type=int, default=20)
    p.add_argument("--max-live-calls", type=int, default=None, help="stop after this many uncached LLM calls")
    p.add_argument("--min-interval", type=float, default=0.0, help="seconds between live calls (pacing)")
    p.add_argument("--gemini-attempts", type=int, default=2, help="tries per Gemini call on HTTP 503")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--num-ctx", type=int, default=32768)
    p.add_argument("--max-predict", type=int, default=8192)
    p.add_argument("--resume", action="store_true", help="continue from benchmarks/.work/<run-name>/state.json")
    p.add_argument("--report", nargs="+", metavar="RESULTS_JSON", help="print markdown tables and exit")
    p.add_argument("--rescore", action="store_true",
                   help="rewrite results/<run-name>.json from the saved state with the current scoring code "
                        "(no LLM calls; the run's other options are read from the state)")
    args = p.parse_args(argv)
    args.systems = [s for s in args.systems.split(",") if s]
    args.graphrag_graphs = [g for g in args.graphrag_graphs.split(",") if g]
    args.tasks = [t for t in args.tasks.split(",") if t]
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.report:
        report(args.report)
        return 0
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _quiet_logs()
    os.chdir(REPO_ROOT)  # kg_to_sql and the orchestrator read sql/ relative to the repo root
    if args.rescore:
        state = json.loads((BENCH_DIR / ".work" / args.run_name / "state.json").read_text(encoding="utf-8"))
        for key, value in state["args"].items():
            if key not in ("rescore", "report", "resume"):
                setattr(args, key, value)
        args.resume = True
        run = Run(args)
        write_results(run, state.get("status", "partial"), state.get("message"), state.get("resume_command"))
        return 0
    run = Run(args)
    run.state["args"] = {k: v for k, v in vars(args).items() if k not in ("rescore", "report", "resume")}
    status, message = "complete", None
    try:
        for task in args.tasks:  # in the order given
            {"er": task_er, "qa": task_qa}[task](run)
    except (Halt, BudgetExhausted, QuotaExhausted) as exc:
        status, message = "partial", f"stopped: {exc}"
        log.warning("Run stopped early: %s", exc)
    finally:
        run.save_state()
    if status == "complete" and "qa" in args.tasks:
        qa = run.state.get("qa_questions", [])
        systems = run.state.get("qa_systems", [])
        if any(f"{q}|{s}" not in run.state["units"] for q in qa for s in systems):
            status = "partial"
    run.state.update({"status": status, "message": message, "resume_command": resume_command()})
    run.save_state()
    write_results(run, status, message)
    if status != "complete":
        print("\nResume with:\n  " + resume_command())
    return 0


if __name__ == "__main__":
    sys.exit(main())
