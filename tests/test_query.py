"""Milvus Lite ingestion/search and the query orchestrator (fake Gemini + fake embeddings)."""

import pytest

from conftest import entity_id


@pytest.fixture
def milvus_path(tmp_path):
    return str(tmp_path / "milvus_orchestrator.db")


@pytest.fixture
def ingested(kg_db, milvus_path, fake_embedder):
    from src.milvus_ingestion import MilvusIngestionEngine

    engine = MilvusIngestionEngine(db_path=str(kg_db), milvus_path=milvus_path)
    stats = engine.ingest_from_sql()
    engine.close()
    return stats


def test_ingestion_stats(ingested):
    assert ingested["total_entities"] == 5
    assert ingested["ingested"] == 5


def test_search_ranks_exact_match_first(kg_db, milvus_path, ingested, fake_embedder):
    from src.milvus_ingestion import MilvusIngestionEngine

    engine = MilvusIngestionEngine(db_path=str(kg_db), milvus_path=milvus_path)
    try:
        assert engine.milvus_client is not None  # auto-connected to the existing file
        results = engine.search_similar_entities("Beta Supplies LLP", top_k=5, similarity_threshold=0.0)
        assert results[0]["canonical_name"] == "Beta Supplies LLP"
        scores = [r["similarity_score"] for r in results]
        # scores are cosine similarities: best first, never above 1
        assert scores == sorted(scores, reverse=True)
        assert 0.0 < scores[0] <= 1.0
        assert scores[0] > scores[-1]
    finally:
        engine.close()


@pytest.fixture
def orchestrator(kg_db, milvus_path, ingested, fake_gemini):
    from src.orchestrator import QueryOrchestrator

    orch = QueryOrchestrator(db_path=str(kg_db), milvus_path=milvus_path)
    yield orch
    orch.close()


def test_flashlight_query(orchestrator, fake_gemini):
    fake_gemini.plan = {"query_mode": "flashlight", "entities": ["Beta Supplies LLP"],
                        "entity_types": [], "relationship_types": []}
    result = orchestrator.process_query("Who is Beta Supplies LLP related to?", ["UNCLASSIFIED"])

    assert result["status"] == "success"
    assert "Beta Supplies LLP" in {c["canonical_name"] for c in result["candidates"]}
    relations = {(e["source_name"], e["relation"], e["target_name"]) for e in result["mini_graph"]}
    assert ("Director A", "DESIGNATED_PARTNER_OF", "Beta Supplies LLP") in relations
    assert result["citation_context"]


def test_flashlight_by_entity_type(orchestrator, fake_gemini):
    fake_gemini.plan = {"query_mode": "flashlight", "entities": [],
                        "entity_types": ["Person"], "relationship_types": []}
    result = orchestrator.process_query("List all people", ["UNCLASSIFIED"], max_depth=0)

    assert result["status"] == "success"
    assert {c["canonical_name"] for c in result["candidates"]} == {"Director A", "Director B"}


def test_laser_query(orchestrator, fake_gemini, kg_db):
    a = entity_id(kg_db, "Director A")
    beta = entity_id(kg_db, "Beta Supplies LLP")
    fake_gemini.plan = {"query_mode": "laser",
                        "source_entity": {"id": a, "name": "Director A"},
                        "target_entity": {"id": beta, "name": "Beta Supplies LLP"}}
    result = orchestrator.process_query("How is Director A connected to Beta Supplies LLP?", ["UNCLASSIFIED"])

    assert result["status"] == "success"
    assert [e["relation"] for e in result["mini_graph"]] == ["DESIGNATED_PARTNER_OF"]


def test_unusable_plan_is_ambiguous(orchestrator, fake_gemini):
    fake_gemini.plan = {"query_mode": "flashlight", "entities": [], "entity_types": [],
                        "relationship_types": []}
    result = orchestrator.process_query("hello", ["UNCLASSIFIED"])
    assert result["status"] == "ambiguous"


def test_no_access(orchestrator, fake_gemini, kg_db):
    import sqlite3

    conn = sqlite3.connect(kg_db)
    conn.execute("UPDATE documents SET access_tags = '[\"BOARD\"]'")
    conn.commit()
    conn.close()

    fake_gemini.plan = {"query_mode": "flashlight", "entities": [],
                        "entity_types": ["Company"], "relationship_types": []}
    result = orchestrator.process_query("List all companies", ["FINANCE"])
    assert result["status"] == "no_access"
