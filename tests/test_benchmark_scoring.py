"""Scoring code of the benchmark in benchmarks/ (offline, no LLM)."""

import json

import pytest

from benchmarks import scoring
from benchmarks.run_benchmark import serialize_graph_context

ENTITIES = [
    {"id": "P1", "type": "Person", "canonical_name": "Arjun Mehta",
     "aliases": ["Arjun Mehta", "Mr. Arjun Mehta"], "ambiguous_surfaces": ["A. Mehta"]},
    {"id": "P6", "type": "Person", "canonical_name": "Arun Mehta",
     "aliases": ["Arun Mehta"], "ambiguous_surfaces": ["A. Mehta"]},
    {"id": "C1", "type": "Company", "canonical_name": "Larkspur Castings Private Limited",
     "aliases": ["Larkspur Castings Pvt. Ltd.", "Larkspur Castings", "LCPL"]},
    {"id": "C4", "type": "Company", "canonical_name": "Larkspur Foundry Services Private Limited",
     "aliases": ["Larkspur Foundry Services", "LFSPL"]},
    {"id": "C5", "type": "Company", "canonical_name": "Kestrel Alloys LLP",
     "aliases": ["M/s Kestrel Alloys LLP", "Kestrel Alloys"]},
    {"id": "C9", "type": "Company", "canonical_name": "Kestrel Alloy Castings Limited",
     "aliases": ["Kestrel Alloy Castings Ltd"]},
    {"id": "C11", "type": "Organization", "canonical_name": "Seabright Cooperative Bank Limited",
     "aliases": ["Seabright Co-operative Bank Ltd"]},
]


@pytest.fixture
def index():
    return scoring.AliasIndex(ENTITIES)


# ------------------------------------------------------------ normalisation
@pytest.mark.parametrize("raw, expected", [
    ("M/s Kestrel Alloys LLP", "kestrel alloys llp"),
    ("Larkspur Castings Pvt. Ltd.", "larkspur castings pvt ltd"),
    ("Larkspur Castings Private Limited", "larkspur castings pvt ltd"),
    ("Seabright Co-operative Bank Ltd", "seabright cooperative bank ltd"),
    ("Mr. Arjun K. Mehta", "arjun k mehta"),
    ("Sen & Varma LLP", "sen and varma llp"),
    ("INR 12,600,000", "inr 12600000"),
])
def test_normalize_name(raw, expected):
    assert scoring.normalize_name(raw) == expected


def test_alias_index_recognises_variants_and_keeps_similar_names_apart(index):
    assert index.recognize("Mr. Arjun Mehta, Managing Director") == ["P1"]
    assert index.recognize("signed by Arun Mehta") == ["P6"]
    # Declared ambiguous: never counted for either person
    assert index.recognize("A. Mehta, Partner") == []
    assert index.recognize("Kestrel Alloy Castings Ltd and M/s Kestrel Alloys LLP") == ["C9", "C5"]
    # Longest alias wins: "Larkspur Foundry Services" is not read as Larkspur Castings
    assert index.recognize("Larkspur Foundry Services Pvt Ltd") == ["C4"]
    assert index.recognize("LARKSPUR CASTINGS PRIVATE LIMITED") == ["C1"]


def test_alias_index_align(index):
    assert index.align("Larkspur Castings Private Ltd.") == "C1"
    assert index.align("Patel Rao & Associates") is None
    assert index.align("Seabright Co-operative Bank Ltd, Riverton") == "C11"
    assert index.align("Arjun Mehta and Arun Mehta") is None  # two entities: not aligned


def test_colliding_aliases_are_dropped():
    idx = scoring.AliasIndex([
        {"id": "A", "canonical_name": "Alpha One", "aliases": ["Alpha"]},
        {"id": "B", "canonical_name": "Alpha Two", "aliases": ["Alpha"]},
    ])
    assert idx.recognize("Alpha") == []
    assert idx.recognize("Alpha Two") == ["B"]


# ---------------------------------------------------- resolution metrics
GOLD = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B", "c1": "C"}


def test_perfect_clustering():
    s = scoring.resolution_scores([{"a1", "a2", "a3"}, {"b1", "b2"}], GOLD)
    assert s["precision"] == s["recall"] == s["f1"] == 1.0
    assert s["purity"] == s["inverse_purity"] == 1.0
    assert s["gold_pairs"] == 4 and s["predicted_clusters"] == 3 and s["gold_clusters"] == 3


def test_no_merges_has_undefined_precision_and_zero_f1():
    s = scoring.resolution_scores([], GOLD)
    assert s["precision"] is None
    assert s["recall"] == 0.0 and s["f1"] == 0.0
    assert s["purity"] == 1.0
    assert s["inverse_purity"] == pytest.approx(3 / 6)


def test_over_merge_lowers_precision_and_purity():
    s = scoring.resolution_scores([{"a1", "a2", "a3", "b1", "b2", "c1"}], GOLD)
    assert s["recall"] == 1.0
    assert s["precision"] == pytest.approx(4 / 15)
    assert s["purity"] == pytest.approx(3 / 6)
    assert s["f1"] == pytest.approx(2 * (4 / 15) / (4 / 15 + 1))


def test_partial_clustering_and_foreign_ids():
    # "zz" is not a scored mention (e.g. an ID the LLM invented) and is ignored
    s = scoring.resolution_scores([{"a1", "a2", "zz"}, {"b1", "c1"}], GOLD)
    assert s["true_pairs"] == 1 and s["predicted_pairs"] == 2
    assert s["precision"] == 0.5 and s["recall"] == 0.25


def test_item_in_two_clusters_keeps_first():
    labels = scoring.clusters_to_labels([{"a1", "a2"}, {"a2", "b1"}], ["a1", "a2", "b1"])
    assert labels["a1"] == labels["a2"] != labels["b1"]


def test_exact_and_fuzzy_baselines():
    names = {
        "m1": "Mr. Arjun Mehta", "m2": "Arjun Mehta", "m3": "Arun Mehta",
        "m4": "Larkspur Castings Pvt. Ltd.", "m5": "Larkspur Castings Private Limited",
        "m6": "LCPL", "m7": "M. Iyer", "m8": "Meera Iyer",
    }
    exact = scoring.exact_match_clusters(names)
    assert sorted(map(sorted, exact)) == [["m1", "m2"], ["m4", "m5"]]

    fuzzy = scoring.fuzzy_match_clusters(names, threshold=90)
    merged = next(c for c in fuzzy if "m1" in c)
    assert "m3" in merged  # the classic false merge: Arjun vs Arun Mehta
    assert not any("m6" in c for c in fuzzy)  # abbreviations are out of reach
    assert not any({"m7", "m8"} <= c for c in fuzzy)


# ---------------------------------------------------------- answer scoring
def q(atype, answer, text="Question?"):
    return {"question": text, "answer_type": atype, "answer": answer,
            "evidence_entities": [], "evidence_paths": []}


def test_entity_answers(index):
    question = q("entity", ["P1"], "Who is the Managing Director of Larkspur Castings?")
    exact = scoring.score_answer(question, "Arjun Mehta", index)
    assert exact["correct"] and exact["em"] and exact["f1"] == 1.0 and exact["score"] == 1.0

    verbose = scoring.score_answer(question, "Mr. Arjun Mehta is the MD of Larkspur Castings.", index)
    assert verbose["correct"] and not verbose["em"]
    assert verbose["f1"] == 1.0  # entities named in the question do not count against the answer

    wrong = scoring.score_answer(question, "Arun Mehta", index)
    assert not wrong["correct"] and wrong["score"] == 0.0

    hedge = scoring.score_answer(question, "Arjun Mehta or Arun Mehta", index)
    assert hedge["correct"] and hedge["f1"] == pytest.approx(2 / 3)
    assert hedge["score"] == pytest.approx(2 / 3)  # the headline score penalises hedging


def test_abstention(index):
    s = scoring.score_answer(q("entity", ["P1"]), "Not found.", index)
    assert s["abstained"] and s["score"] == 0.0 and s["f1"] == 0.0
    s = scoring.score_answer(q("yes_no", "no"), "Not found", index)
    assert s["abstained"] and not s["correct"]


def test_entity_any(index):
    question = q("entity_any", ["C4", "C5"], "How is Tomas Fernandes connected to Larkspur Castings?")
    assert scoring.score_answer(question, "Through Kestrel Alloys LLP", index)["correct"]
    assert scoring.score_answer(question, "Larkspur Foundry Services", index)["score"] == 1.0
    assert not scoring.score_answer(question, "Larkspur Castings", index)["correct"]


def test_list_answers(index):
    question = q("entity_list", ["C1", "C4"], "Which companies does Sen & Varma LLP audit?")
    full = scoring.score_answer(question, "Larkspur Castings Pvt Ltd; LFSPL", index)
    assert full["em"] and full["score"] == 1.0
    half = scoring.score_answer(question, "Larkspur Castings", index)
    assert not half["correct"] and half["score"] == pytest.approx(2 / 3)
    extra = scoring.score_answer(question, "LCPL; LFSPL; Kestrel Alloys LLP", index)
    assert extra["score"] == pytest.approx(0.8)


@pytest.mark.parametrize("answer, ok, em", [
    ("12600000", True, True),
    ("INR 12,600,000", True, True),
    ("INR 1,26,00,000", True, True),
    ("12.6 million", True, False),
    ("Rs 1.26 crore", True, False),
    ("INR 15,000,000", False, False),
])
def test_number_answers(index, answer, ok, em):
    s = scoring.score_answer(q("number", 12600000), answer, index)
    assert s["correct"] is ok and s["em"] is em


def test_percentage_and_id_answers(index):
    assert scoring.score_answer(q("number", 62), "62%", index)["em"]
    s = scoring.score_answer(q("string", "08220202"), "DIN 08220202", index)
    assert s["correct"] and not s["em"] and s["score"] == 1.0
    hedge = scoring.score_answer(q("string", "08220202"), "07410101; 08220202", index)
    assert hedge["correct"] and hedge["score"] == 0.0
    assert not scoring.score_answer(q("string", "08220202"), "8220202", index)["correct"]


@pytest.mark.parametrize("answer, reading", [
    ("No", "no"), ("No.", "no"), ("Yes, she is a director.", "yes"),
    ("Kestrel Alloy Castings Limited is not a related party.", "no"), ("Maybe", None),
])
def test_yes_no(answer, reading):
    assert scoring.parse_yes_no(answer) == reading


def test_parse_numbers():
    assert scoring.parse_numbers("INR 40,000,000 over 60 months") == [40000000.0, 60.0]
    assert scoring.parse_numbers("FY2025") == []


# ------------------------------------------------------------- retrieval
def test_context_and_subgraph_coverage(index):
    question = {"question": "How is Arun Mehta connected to Larkspur Castings?", "answer_type": "entity",
                "answer": ["C5"], "evidence_entities": ["P6", "C5", "C1"],
                "evidence_paths": [[["P6", "C5"], ["C5", "C1"]]]}
    ctx = scoring.context_coverage(question, "Arun Mehta is a partner. Kestrel Alloys supplies LCPL.", index)
    assert ctx == {"evidence_recall": 1.0, "answer_in_context": True}

    sub = scoring.subgraph_coverage(
        question, ["Arun Mehta", "Kestrel Alloys LLP", "Larkspur Castings Pvt Ltd"],
        [("Arun Mehta", "Kestrel Alloys LLP"), ("Larkspur Castings Pvt Ltd", "Kestrel Alloys LLP")], index)
    assert sub["answer_entities_in_subgraph"] and sub["path_hit"] and sub["evidence_recall"] == 1.0

    broken = scoring.subgraph_coverage(question, ["Arun Mehta", "Kestrel Alloys LLP"],
                                       [("Arun Mehta", "Kestrel Alloys LLP")], index)
    assert not broken["path_hit"] and broken["evidence_recall"] == pytest.approx(2 / 3)


def test_chunk_words():
    words = " ".join(f"w{i}" for i in range(250))
    chunks = scoring.chunk_words(words, size=100, overlap=20)
    assert [len(c.split()) for c in chunks] == [100, 100, 90]
    assert chunks[1].split()[0] == "w80"
    assert scoring.chunk_words("", 100, 20) == []
    with pytest.raises(ValueError):
        scoring.chunk_words(words, size=10, overlap=10)


def test_bm25_ranks_matching_document_first():
    docs = [scoring.bm25_tokens(t) for t in [
        "Kestrel Alloys LLP supplies alloy ingots",
        "Harbourline Logistics borrowed from Seabright Bank",
        "The board approved the minutes",
    ]]
    bm25 = scoring.BM25(docs)
    assert bm25.top_k(scoring.bm25_tokens("Which bank lent to Harbourline?"), 2)[0] == 1
    assert bm25.scores(["unseen"]) == [0.0, 0.0, 0.0]


def test_serialize_graph_context():
    text = serialize_graph_context({
        "nodes": {"x": {"name": "Arun Mehta", "type": "Person", "attributes": {"role": "Partner", "din": ""}}},
        "mini_graph": [{"source_name": "Arun Mehta", "relation": "PARTNER_OF", "target_name": "Sen & Varma LLP",
                        "properties": {}}],
        "citation_context": {"larkspur_annual_report_fy2025": [2]},
    })
    assert "- Arun Mehta (Person): role=Partner" in text
    assert "Arun Mehta -[PARTNER_OF]-> Sen & Varma LLP" in text
    assert "larkspur_annual_report_fy2025 p. 2" in text


# --------------------------------------------------------- ground truth
def test_committed_ground_truth_matches_generator(tmp_path):
    pytest.importorskip("fitz")
    pytest.importorskip("docx")
    pytest.importorskip("openpyxl")
    from conftest import REPO_ROOT
    from benchmarks import generate_corpus

    generate_corpus.build(tmp_path)
    committed = REPO_ROOT / "benchmarks" / "ground_truth"
    for path in sorted((tmp_path / "ground_truth").iterdir()):
        assert json.loads(path.read_text()) == json.loads((committed / path.name).read_text()), path.name
    for path in sorted((tmp_path / "corpus").iterdir()):
        assert path.read_bytes() == (REPO_ROOT / "benchmarks" / "corpus" / path.name).read_bytes(), path.name


def test_ground_truth_is_consistent():
    from conftest import REPO_ROOT

    gt = REPO_ROOT / "benchmarks" / "ground_truth"
    entities = json.loads((gt / "entities.json").read_text())["entities"]
    mentions = json.loads((gt / "mentions.json").read_text())["mentions"]
    questions = json.loads((gt / "questions.json").read_text())["questions"]
    idx = scoring.AliasIndex(entities)
    ids = {e["id"] for e in entities}
    assert len(entities) == 20 and len(mentions) == 66 and len(questions) == 39
    assert {m["entity_id"] for m in mentions} == ids
    for question in questions:
        gold = question["answer"] if question["answer_type"].startswith("entity") else []
        # a gold answer entity must not be named in the question itself
        assert not set(gold) & set(idx.recognize(question["question"])), question["id"]
        # every gold entity can be recognised from its canonical name
        for eid in gold:
            assert idx.recognize(idx.canonical[eid]) == [eid]


def test_cross_document_questions_need_two_documents():
    from conftest import REPO_ROOT

    questions = json.loads((REPO_ROOT / "benchmarks" / "ground_truth" / "questions.json").read_text())["questions"]
    cross = [q for q in questions if q["type"] == "cross_doc"]
    assert cross and all(q["min_documents"] >= 2 for q in cross)
