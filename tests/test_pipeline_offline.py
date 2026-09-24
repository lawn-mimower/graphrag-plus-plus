"""
Offline end-to-end run: parse the synthetic documents -> extract (fake Gemini)
-> build graph -> 5 dedup methods -> SQLite -> Milvus Lite -> Leiden -> query.
"""

import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

pytest.importorskip("rapidocr_onnxruntime")
pytest.importorskip("splink")
pytest.importorskip("leidenalg")


def test_full_pipeline_offline(dataset_dir, tmp_outputs, tmp_path, fake_gemini, fake_embedder):
    import kg_to_sql
    import scripts.run_leiden as run_leiden
    from src.benchmark_harness import BenchmarkHarness
    from src.milvus_ingestion import MilvusIngestionEngine
    from src.orchestrator import QueryOrchestrator

    # Phase 1: parse, extract, build graph, deduplicate with all methods
    BenchmarkHarness().run(dataset_dir)

    extracted = sorted(p.name for p in (tmp_outputs / "entities_extracted").iterdir())
    assert extracted == [
        "acme_annual_report_entities.json", "acme_board_minutes_entities.json",
        "acme_letterhead_entities.json", "acme_related_parties_entities.json",
    ]
    report = json.loads(next(tmp_outputs.glob("benchmark_report_*.json")).read_text())
    assert report["non_dedup_graph_stats"]["num_nodes"] == 14
    results = report["deduplication_results"]
    assert set(results) == {"rswoosh", "probabilistic", "topological", "semantic", "llm_full_context"}
    assert not [m for m, r in results.items() if "error" in r]
    assert results["llm_full_context"]["nodes_after"] == 5

    graph_path = tmp_outputs / "knowledge_graphs" / "dedup_llm_full_context_kg.gpickle"
    assert graph_path.exists()
    assert list((tmp_outputs / "reasoning").glob("llm_full_context_reasoning_*.json"))

    # Phase 2: SQL + Milvus Lite
    db_path = tmp_path / "knowledge_graph.db"
    kg_to_sql.convert_graph_to_sql(graph_path, db_path)
    milvus_path = str(tmp_path / "milvus_orchestrator.db")
    engine = MilvusIngestionEngine(db_path=str(db_path), milvus_path=milvus_path)
    assert engine.ingest_from_sql()["ingested"] == 5
    engine.close()

    # Phase 3A: communities with summaries
    assert run_leiden.run_leiden_pipeline(str(db_path), force=True, skip_summaries=False)

    # Query
    orch = QueryOrchestrator(db_path=str(db_path), milvus_path=milvus_path)
    try:
        fake_gemini.plan = {"query_mode": "flashlight", "entities": ["Director B"],
                            "entity_types": ["Person"], "relationship_types": []}
        result = orch.process_query("Who is Director B?", ["UNCLASSIFIED"], max_depth=1)
    finally:
        orch.close()

    assert result["status"] == "success"
    assert "Director B" in {n["name"] for n in result["nodes"].values()}
    assert result["citation_context"]["acme_annual_report"] == [1]


def _run(args, env_overrides=None, timeout=300):
    env = {k: v for k, v in os.environ.items() if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
    env.update(env_overrides or {})
    return subprocess.run(
        [sys.executable, *args], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=timeout,
    )


def test_cli_config_without_key():
    proc = _run(["main.py", "--config"])
    assert proc.returncode == 0, proc.stderr
    assert "Model (Heavy)" in proc.stdout


def test_cli_rejects_folder_without_documents(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing to parse")
    proc = _run(["main.py", "--dataset", str(tmp_path)])
    assert proc.returncode == 1
    assert "No supported documents" in proc.stdout + proc.stderr


def test_cli_kg_to_sql_and_leiden_without_key(kg_db, raw_builder, tmp_path):
    import pickle
    from conftest import dedup_by_name

    graph = tmp_path / "g.gpickle"
    with open(graph, "wb") as f:
        pickle.dump(dedup_by_name(raw_builder), f)
    db = tmp_path / "cli.db"

    proc = _run(["kg_to_sql.py", "--graph", str(graph), "--output", str(db)])
    assert proc.returncode == 0, proc.stderr
    proc = _run(["scripts/run_leiden.py", "--db", str(db), "--force", "--skip-summaries"])
    assert proc.returncode == 0, proc.stderr
    assert "LEIDEN PIPELINE COMPLETE" in proc.stderr + proc.stdout
