import networkx as nx

from src.methodologies.fuzzy import FuzzyDeduplicator, normalize_name


def test_normalize_name_folds_suffixes_honorifics_and_punctuation():
    assert normalize_name("M/s Kestrel Alloys LLP") == "kestrel alloys llp"
    assert normalize_name("Larkspur Castings Pvt. Ltd.") == "larkspur castings pvt ltd"
    assert normalize_name("Larkspur Castings Private Limited") == "larkspur castings pvt ltd"
    assert normalize_name("Mr. Arun Mehta") == "arun mehta"
    assert normalize_name("Seabright Co-operative Bank Ltd") == "seabright cooperative bank ltd"


def graph(nodes):
    g = nx.DiGraph()
    for node_id, name, etype in nodes:
        g.add_node(node_id, name=name, type=etype)
    return g


def test_merges_variants_of_the_same_entity_only_within_a_type():
    g = graph([
        ("c1", "Larkspur Castings Private Limited", "Company"),
        ("c2", "Larkspur Castings Pvt. Ltd.", "Company"),
        ("c3", "LARKSPUR CASTINGS LTD", "Company"),
        ("p1", "Larkspur Castings", "Person"),  # same words, different type: never merged
        ("c4", "Harbourline Logistics Pvt Ltd", "Company"),
    ])
    clusters = FuzzyDeduplicator(threshold=90).deduplicate(g)
    assert clusters == [{"c1", "c2", "c3"}]


def test_honorific_and_initial_variants_of_a_person_merge():
    g = graph([("p1", "Mr Arun Mehta", "Person"), ("p2", "Arun Mehta", "Person"), ("p3", "A. Mehta", "Person")])
    clusters = FuzzyDeduplicator(threshold=90).deduplicate(g)
    assert {"p1", "p2"} <= next(c for c in clusters if "p1" in c)
    assert not any("p3" in c for c in clusters), "an initial alone is below the similarity threshold"


def test_nodes_without_names_are_ignored():
    g = nx.DiGraph()
    g.add_node("x", type="Company")
    g.add_node("y", name="", type="Company")
    assert FuzzyDeduplicator().deduplicate(g) == []
