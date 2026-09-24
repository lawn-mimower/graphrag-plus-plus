"""Graph metrics, Leiden communities and community summaries."""

import json
import sqlite3

import pytest

pytest.importorskip("leidenalg")
pytest.importorskip("igraph")


@pytest.fixture
def leiden_db(kg_db):
    import scripts.run_leiden as run_leiden

    assert run_leiden.apply_schema(str(kg_db))
    return kg_db


def test_metrics_and_resolutions(kg_db):
    from src.graph_metrics import GraphMetricsCalculator

    calc = GraphMetricsCalculator(str(kg_db))
    metrics = calc.calculate_all_metrics()
    assert metrics["node_count"] == 5
    assert metrics["connected_components"] == 1
    resolutions = calc.calculate_optimal_resolutions(metrics)
    assert set(resolutions) == {"micro", "meso", "macro"}
    assert resolutions["micro"] >= resolutions["meso"] >= resolutions["macro"]


def test_build_communities(leiden_db):
    from src.graph_metrics import GraphMetricsCalculator
    from src.leiden_builder import LeidenCommunityBuilder

    assert GraphMetricsCalculator(str(leiden_db)).should_recompute_leiden()[0] is True

    stats = LeidenCommunityBuilder(str(leiden_db)).build_communities(force=True)
    assert stats["total_entities"] == 5
    assert set(stats["levels"]) == {"micro", "meso", "macro"}
    assert all(n >= 1 for n in stats["levels"].values())

    conn = sqlite3.connect(leiden_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM leiden_communities").fetchone()[0] == 15
        seconds, resolutions = conn.execute(
            "SELECT computation_time_seconds, resolutions FROM leiden_metadata"
        ).fetchone()
        assert 0 <= seconds < 600  # elapsed time, not a Unix timestamp
        assert set(json.loads(resolutions)) == {"micro", "meso", "macro"}
    finally:
        conn.close()

    # unchanged graph -> no recomputation needed
    assert GraphMetricsCalculator(str(leiden_db)).should_recompute_leiden()[0] is False


def test_community_summaries_with_fake_client(leiden_db, fake_gemini):
    from src.community_summarizer import CommunitySummarizer
    from src.leiden_builder import LeidenCommunityBuilder

    LeidenCommunityBuilder(str(leiden_db)).build_communities(force=True)
    summarizer = CommunitySummarizer(str(leiden_db))
    counts = summarizer.generate_all_summaries()

    conn = sqlite3.connect(leiden_db)
    try:
        per_level = dict(conn.execute(
            "SELECT level, COUNT(DISTINCT community_id) FROM leiden_communities GROUP BY level"
        ))
        rows = conn.execute("SELECT level, community_id, summary FROM community_summaries").fetchall()
    finally:
        conn.close()

    assert counts == per_level
    assert all(summary.startswith("Community of") for _, _, summary in rows)
    assert summarizer.get_summary(rows[0][1], rows[0][0]) == rows[0][2]


def test_run_leiden_pipeline_without_api_key(kg_db, monkeypatch):
    """--skip-summaries needs no Gemini key."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import scripts.run_leiden as run_leiden

    assert run_leiden.run_leiden_pipeline(str(kg_db), force=True, skip_summaries=True) is True
