"""Graph -> SQLite conversion and ACL-enforced graph traversal."""

import json
import sqlite3

import pytest

from conftest import entity_id


def _count(db, table):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def test_kg_to_sql_tables(kg_db):
    assert _count(kg_db, "documents") == 4
    assert _count(kg_db, "entities") == 5
    assert _count(kg_db, "relationships") > 0
    assert _count(kg_db, "entity_occurrences") >= 5

    conn = sqlite3.connect(kg_db)
    try:
        tags = {row[0] for row in conn.execute("SELECT access_tags FROM documents")}
        assert tags == {'["UNCLASSIFIED"]'}
        sizes = dict(conn.execute("SELECT canonical_name, cluster_size FROM entities"))
        assert sizes["Director A"] == 4
        assert sizes["Example & Co"] == 1
        attrs = json.loads(conn.execute(
            "SELECT attributes FROM entities WHERE canonical_name = 'Example & Co'"
        ).fetchone()[0])
        assert attrs["attr_type"] == "Chartered Accountants"
    finally:
        conn.close()


def test_get_context_returns_subgraph_and_citations(kg_db):
    from src.inference_engine import SecureGraphTraverser

    acme = entity_id(kg_db, "Acme Widgets Pvt Ltd")
    ctx = SecureGraphTraverser(str(kg_db)).get_context([acme], ["UNCLASSIFIED"], max_depth=1)

    names = {n["name"] for n in ctx["nodes"].values()}
    assert {"Acme Widgets Pvt Ltd", "Director A", "Director B", "Example & Co"} <= names
    relations = {(e["source_name"], e["relation"], e["target_name"]) for e in ctx["mini_graph"]}
    assert ("Director A", "DIRECTOR_OF", "Acme Widgets Pvt Ltd") in relations
    assert ctx["citation_context"]["acme_annual_report"] == [1]


def test_find_shortest_path(kg_db):
    from src.inference_engine import SecureGraphTraverser

    traverser = SecureGraphTraverser(str(kg_db))
    path = traverser.find_shortest_path(
        entity_id(kg_db, "Example & Co"), entity_id(kg_db, "Beta Supplies LLP"), ["UNCLASSIFIED"]
    )
    hops = [e["relation"] for e in path["mini_graph"]]
    assert len(hops) == 2
    assert hops[0] == "AUDITOR_OF"


def test_acl_hides_restricted_documents(kg_db):
    from src.inference_engine import SecureGraphTraverser

    conn = sqlite3.connect(kg_db)
    try:
        # Only the annual report mentions the auditor; restrict it to the AUDIT tag
        conn.execute(
            "UPDATE documents SET access_tags = '[\"AUDIT\"]' WHERE document_name = 'acme_annual_report'"
        )
        conn.commit()
    finally:
        conn.close()

    traverser = SecureGraphTraverser(str(kg_db))
    acme = entity_id(kg_db, "Acme Widgets Pvt Ltd")

    public = traverser.get_context([acme], ["FINANCE"], max_depth=1)
    public_names = {n["name"] for n in public["nodes"].values()}
    assert "Example & Co" not in public_names
    assert "acme_annual_report" not in public["citation_context"]

    audit = traverser.get_context([acme], ["AUDIT"], max_depth=1)
    assert "Example & Co" in {n["name"] for n in audit["nodes"].values()}
    assert audit["citation_context"]["acme_annual_report"] == [1]


def test_traverser_rejects_missing_tables(tmp_path):
    from src.inference_engine import SecureGraphTraverser

    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    with pytest.raises(ValueError, match="missing required tables"):
        SecureGraphTraverser(str(empty))
