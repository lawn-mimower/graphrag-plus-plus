"""Full-context LLM deduplication must survive large graphs and imperfect replies."""
import json
from types import SimpleNamespace

import networkx as nx

from src.methodologies.llm_full_context import LLMFullContextDeduplicator


class FakeModels:
    """Answers each call by grouping entities whose names share a first word."""

    def __init__(self, truncate_first=False):
        self.calls = []
        self.truncate_first = truncate_first

    def generate_content(self, model=None, contents=None, config=None):
        prompt = contents[0]
        self.calls.append(prompt)
        start = prompt.rindex("{", 0, prompt.index('"entities"'))
        entities = json.JSONDecoder().raw_decode(prompt, start)[0]["entities"]
        groups = {}
        for e in entities:
            groups.setdefault(e["attributes"]["name"].split()[0], []).append(e["id"])
        dups = [{"cluster_id": n, "entities": ids, "reasoning": "same first word"}
                for n, ids in enumerate(v for v in groups.values() if len(v) > 1)]
        text = json.dumps({"duplicates": dups, "summary": {}})
        if self.truncate_first and len(self.calls) == 1:
            text = text[: len(text) - 8]  # cut the closing of the JSON, as an output cap would
        return SimpleNamespace(text="```json\n" + text + "\n```")


def make_graph(n_people, n_companies):
    g = nx.DiGraph()
    for k in range(n_people):
        g.add_node(f"P{k}", type="Person", name=f"Person{k % 5} variant {k}")
    for k in range(n_companies):
        g.add_node(f"C{k}", type="Company", name=f"Company{k % 4} Ltd {k}")
    return g


def test_small_graph_uses_one_call():
    dedup = LLMFullContextDeduplicator(SimpleNamespace(models=FakeModels()), batch_size=30)
    g = make_graph(6, 4)
    clusters = dedup.deduplicate(g)
    assert len(dedup.client.models.calls) == 1
    assert frozenset({"P0", "P5"}) in {frozenset(c) for c in clusters}


def test_large_graph_is_batched_by_type_and_reconciled():
    models = FakeModels()
    dedup = LLMFullContextDeduplicator(SimpleNamespace(models=models), batch_size=10)
    g = make_graph(25, 12)  # Person: 3 batches, Company: 2 batches
    details = dedup.deduplicate_with_details(g)
    assert details["batches"] > 5  # 5 type batches plus reconciliation passes
    clusters = {frozenset(c["entities"]) for c in details["duplicates"]}
    # every Person{k % 5} family (5 members each) must end up in ONE cluster despite spanning batches
    for fam in range(5):
        members = frozenset(f"P{k}" for k in range(25) if k % 5 == fam)
        assert members in clusters
    assert all(c["reasoning"] == "same first word" for c in details["duplicates"])
    assert details["failed_batches"] == 0


def test_truncated_reply_is_repaired_not_discarded():
    models = FakeModels(truncate_first=True)
    dedup = LLMFullContextDeduplicator(SimpleNamespace(models=models), batch_size=30)
    clusters = dedup.deduplicate(make_graph(6, 0))
    assert clusters, "a truncated JSON reply should still yield the recoverable clusters"
    assert dedup.failed_batches == 0


def test_empty_reply_counts_as_a_failed_batch():
    class Empty:
        calls = []

        def generate_content(self, **kwargs):
            return SimpleNamespace(text="")

    dedup = LLMFullContextDeduplicator(SimpleNamespace(models=Empty()), batch_size=30)
    assert dedup.deduplicate(make_graph(4, 0)) == []
    assert dedup.failed_batches == 1
