"""Knowledge graph construction and merge of duplicate clusters."""

import pickle

from conftest import dedup_by_name


def test_every_mention_becomes_a_node(raw_builder):
    stats = raw_builder.get_statistics()
    assert stats["num_nodes"] == 14
    assert stats["source_documents"]["acme_annual_report"] == 5
    assert stats["entity_types"]["Person"] == 7


def test_attributes_do_not_overwrite_node_metadata(raw_builder):
    auditor = next(
        attrs for _, attrs in raw_builder.graph.nodes(data=True)
        if attrs.get("name") == "Example & Co"
    )
    # extracted attribute "type": "Chartered Accountants" must not replace the entity type
    assert auditor["type"] == "Organization"
    assert auditor["attr_type"] == "Chartered Accountants"
    assert auditor["source_doc"] == "acme_annual_report"
    assert auditor["page_numbers"] == [1]


def test_relationships_map_to_internal_ids(raw_builder):
    g = raw_builder.graph
    edge_types = {attrs["relationship_type"] for _, _, attrs in g.edges(data=True)}
    assert "DESIGNATED_PARTNER_OF" in edge_types
    for u, v in g.edges():
        assert u.startswith("node_") and v.startswith("node_")


def test_create_deduplicated_graph(raw_builder):
    dedup = dedup_by_name(raw_builder)
    names = sorted(attrs["name"] for _, attrs in dedup.nodes(data=True))
    assert names == sorted([
        "Acme Widgets Pvt Ltd", "Director A", "Director B", "Example & Co", "Beta Supplies LLP",
    ])

    sizes = {attrs["name"]: attrs["cluster_size"] for _, attrs in dedup.nodes(data=True)}
    assert sizes["Director A"] == 4
    # entities that were never duplicated count as a cluster of one, not zero
    assert sizes["Example & Co"] == 1

    director_a = next(a for _, a in dedup.nodes(data=True) if a["name"] == "Director A")
    assert set(director_a["source_docs"]) == {
        "acme_annual_report", "acme_board_minutes", "acme_related_parties", "acme_letterhead",
    }
    # merged edges never become self loops
    assert all(u != v for u, v in dedup.edges())


def test_save_and_load_roundtrip(raw_builder, tmp_path):
    from src.kg_builder import KnowledgeGraphBuilder

    raw_builder.save_graph(tmp_path / "g.gpickle", format="gpickle")
    raw_builder.save_graph(tmp_path / "g.graphml", format="graphml")

    loaded = KnowledgeGraphBuilder()
    loaded.load_graph(tmp_path / "g.gpickle")
    assert loaded.graph.number_of_nodes() == raw_builder.graph.number_of_nodes()
    with open(tmp_path / "g.gpickle", "rb") as f:
        assert pickle.load(f).number_of_edges() == raw_builder.graph.number_of_edges()
    assert (tmp_path / "g.graphml").stat().st_size > 0
