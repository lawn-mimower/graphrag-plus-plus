"""The five deduplication methodologies on a small synthetic graph."""

import networkx as nx
import pytest

from conftest import FakeGeminiClient


def _names(graph, cluster):
    return sorted(graph.nodes[n].get("name") for n in cluster)


def _assert_valid_clusters(graph, clusters):
    seen = set()
    for cluster in clusters:
        assert isinstance(cluster, set) and len(cluster) > 1
        assert cluster <= set(graph.nodes)
        assert not (cluster & seen)
        seen |= cluster


# ---------------------------------------------------------------- R-Swoosh
def test_rswoosh_merges_same_company_spellings(raw_builder):
    from src.methodologies.rswoosh import RSwooshDeduplicator

    g = raw_builder.graph
    clusters = RSwooshDeduplicator().deduplicate(g)
    _assert_valid_clusters(g, clusters)

    acme = next(c for c in clusters if "Acme Widgets Pvt Ltd" in _names(g, c))
    assert "ACME WIDGETS PVT LTD" in _names(g, acme)
    beta = next(c for c in clusters if "Beta Supplies LLP" in _names(g, c))
    assert len(beta) == 3


def test_rswoosh_similarity_is_bounded():
    from src.methodologies.rswoosh import RSwooshDeduplicator

    dedup = RSwooshDeduplicator()
    same = dedup._compute_similarity({"type": "Company", "name": "Acme"}, {"type": "Company", "name": "Acme"})
    assert same == pytest.approx(1.0)

    # Name-only comparison at ~50% similarity must stay below the 0.85 threshold
    low = dedup._compute_similarity(
        {"type": "Company", "name": "Acme Widgets"}, {"type": "Company", "name": "Acme Gadgets Ltd"}
    )
    assert 0.0 < low < dedup.threshold

    # Different types never match
    assert dedup._compute_similarity({"type": "Person", "name": "X"}, {"type": "Company", "name": "X"}) == 0.0


def test_rswoosh_conflicting_ids_block_merge():
    from src.methodologies.rswoosh import RSwooshDeduplicator

    a = {"type": "Person", "name": "Director A", "din": "00000001"}
    b = {"type": "Person", "name": "Director B", "din": "00000002"}
    assert RSwooshDeduplicator()._compute_similarity(a, b) < 0.85


# ------------------------------------------------------------- Topological
def test_topological_groups_shared_neighbourhoods():
    from src.methodologies.topological import TopologicalDeduplicator

    g = nx.DiGraph()
    g.add_node("acme", type="Company", name="Acme Widgets Pvt Ltd")
    g.add_node("beta", type="Company", name="Beta Supplies LLP")
    g.add_node("a1", type="Person", name="Director A")
    g.add_node("a2", type="Person", name="Director A")
    g.add_node("b", type="Person", name="Director B")
    for n in ("a1", "a2"):
        g.add_edge(n, "acme")
        g.add_edge(n, "beta")
    g.add_edge("b", "acme")

    clusters = TopologicalDeduplicator().deduplicate(g)
    assert clusters == [{"a1", "a2"}]


# ----------------------------------------------------------- Probabilistic
def test_probabilistic_runs_on_small_graph(raw_builder):
    pytest.importorskip("splink")
    from src.methodologies.probabilistic import ProbabilisticDeduplicator

    clusters = ProbabilisticDeduplicator().deduplicate(raw_builder.graph)
    _assert_valid_clusters(raw_builder.graph, clusters)


# ---------------------------------------------------------------- Semantic
def test_semantic_merges_identical_and_keeps_unrelated_apart(fake_embedder, tmp_outputs, raw_builder):
    from src.methodologies.semantic import SemanticDeduplicator

    g = raw_builder.graph
    clusters = SemanticDeduplicator(similarity_threshold=0.95).deduplicate(g)
    _assert_valid_clusters(g, clusters)

    beta = next(c for c in clusters if "Beta Supplies LLP" in _names(g, c))
    assert _names(g, beta) == ["Beta Supplies LLP"] * 3
    # Unrelated entities must never share a cluster (inverted COSINE scores did this)
    for cluster in clusters:
        assert len({g.nodes[n]["type"] for n in cluster}) == 1
        assert "Example & Co" not in _names(g, cluster)
    assert (tmp_outputs / "milvus_lite.db").exists()


# -------------------------------------------------------- LLM full context
def test_llm_full_context_with_fake_client(raw_builder):
    from src.methodologies.llm_full_context import LLMFullContextDeduplicator

    g = raw_builder.graph
    client = FakeGeminiClient()
    dedup = LLMFullContextDeduplicator(client)

    clusters = dedup.deduplicate(g)
    _assert_valid_clusters(g, clusters)
    by_name = {tuple(sorted(set(_names(g, c)))): len(c) for c in clusters}
    assert by_name[("Director A",)] == 4
    assert by_name[("ACME WIDGETS PVT LTD", "Acme Widgets Private Limited", "Acme Widgets Pvt Ltd")] == 3

    details = dedup.deduplicate_with_details(g)
    assert all(d["deduplication_decision"].startswith("MERGE") for d in details["duplicates"])
    assert client.calls[0]["model"] == dedup.model_name


def test_llm_full_context_bad_json_returns_no_clusters(raw_builder):
    from src.methodologies.llm_full_context import LLMFullContextDeduplicator

    client = FakeGeminiClient()
    client.respond = lambda prompt: "I could not decide."
    assert LLMFullContextDeduplicator(client).deduplicate(raw_builder.graph) == []
