"""
Shared fixtures for the test suite.

Offline tests replace the Gemini client with FakeGeminiClient and the
sentence-transformers model with FakeSentenceTransformer, so they need no
API key, no network and no model download. Milvus runs as Milvus Lite
(a local file) and SQLite databases live in pytest's tmp_path.
"""

import hashlib
import json
import re
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_FILES = [
    "acme_annual_report.pdf",
    "acme_board_minutes.docx",
    "acme_related_parties.xlsx",
    "acme_letterhead.png",
]


# ---------------------------------------------------------------------------
# Fake embedding model
# ---------------------------------------------------------------------------
class FakeSentenceTransformer:
    """Deterministic character-trigram embedder with the SentenceTransformer API."""

    DIM = 64

    def __init__(self, model_name=None, *args, **kwargs):
        self.model_name = model_name

    def get_sentence_embedding_dimension(self):
        return self.DIM

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            text = f"  {text.lower()}  "
            for j in range(len(text) - 2):
                bucket = int(hashlib.md5(text[j:j + 3].encode()).hexdigest(), 16) % self.DIM
                out[i, bucket] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


# ---------------------------------------------------------------------------
# Fake Gemini client
# ---------------------------------------------------------------------------
EXTRACTIONS = {
    "annual_report": {
        "entities": [
            {"id": "COMPANY_001", "type": "Company",
             "attributes": {"name": "Acme Widgets Pvt Ltd", "cin": "U00000XX0000PTC000000",
                            "page_numbers": [1]}},
            {"id": "PERSON_001", "type": "Person",
             "attributes": {"name": "Director A", "din": "00000001",
                            "role": "Managing Director", "page_numbers": [1]}},
            {"id": "PERSON_002", "type": "Person",
             "attributes": {"name": "Director B", "din": "00000002",
                            "role": "Independent Director", "page_numbers": [1]}},
            {"id": "ORG_001", "type": "Organization",
             "attributes": {"name": "Example & Co", "type": "Chartered Accountants",
                            "page_numbers": [1]}},
            {"id": "COMPANY_002", "type": "Company",
             "attributes": {"name": "Beta Supplies LLP", "page_numbers": [1]}},
        ],
        "relationships": [
            {"from_id": "PERSON_001", "to_id": "COMPANY_001", "type": "DIRECTOR_OF", "attributes": {}},
            {"from_id": "PERSON_002", "to_id": "COMPANY_001", "type": "DIRECTOR_OF", "attributes": {}},
            {"from_id": "ORG_001", "to_id": "COMPANY_001", "type": "AUDITOR_OF", "attributes": {}},
            {"from_id": "PERSON_001", "to_id": "COMPANY_002", "type": "DESIGNATED_PARTNER_OF",
             "attributes": {}},
            {"from_id": "COMPANY_001", "to_id": "COMPANY_002", "type": "PURCHASED_FROM",
             "attributes": {"amount_inr": 1200000}},
        ],
    },
    "board_minutes": {
        "entities": [
            {"id": "COMPANY_001", "type": "Company",
             "attributes": {"name": "Acme Widgets Private Limited", "page_numbers": [1]}},
            {"id": "PERSON_001", "type": "Person",
             "attributes": {"name": "Director A", "role": "Chairperson", "page_numbers": [1]}},
            {"id": "PERSON_002", "type": "Person",
             "attributes": {"name": "Director B", "page_numbers": [1]}},
            {"id": "COMPANY_002", "type": "Company",
             "attributes": {"name": "Beta Supplies LLP", "page_numbers": [1]}},
        ],
        "relationships": [
            {"from_id": "PERSON_001", "to_id": "COMPANY_001", "type": "DIRECTOR_OF", "attributes": {}},
            {"from_id": "PERSON_002", "to_id": "COMPANY_001", "type": "LENDER_TO",
             "attributes": {"amount_inr": 500000}},
            {"from_id": "COMPANY_001", "to_id": "COMPANY_002", "type": "CONTRACT_WITH",
             "attributes": {}},
        ],
    },
    "related_parties": {
        "entities": [
            {"id": "COMPANY_002", "type": "Company",
             "attributes": {"name": "Beta Supplies LLP", "page_numbers": [1]}},
            {"id": "PERSON_001", "type": "Person",
             "attributes": {"name": "Director A", "page_numbers": [1]}},
            {"id": "PERSON_002", "type": "Person",
             "attributes": {"name": "Director B", "page_numbers": [1]}},
        ],
        "relationships": [
            {"from_id": "PERSON_001", "to_id": "COMPANY_002", "type": "DESIGNATED_PARTNER_OF",
             "attributes": {}},
        ],
    },
    "letterhead": {
        "entities": [
            {"id": "COMPANY_001", "type": "Company",
             "attributes": {"name": "ACME WIDGETS PVT LTD", "page_numbers": [1]}},
            {"id": "PERSON_001", "type": "Person",
             "attributes": {"name": "Director A", "role": "Managing Director",
                            "page_numbers": [1]}},
        ],
        "relationships": [
            {"from_id": "PERSON_001", "to_id": "COMPANY_001", "type": "MANAGING_DIRECTOR_OF",
             "attributes": {}},
        ],
    },
}

# Canonical spellings used when the fake "LLM" deduplicates by name
CANONICAL = {
    "acme widgets pvt ltd": "acme",
    "acme widgets private limited": "acme",
}


def _contents_text(contents) -> str:
    if isinstance(contents, str):
        return contents
    parts = []
    for item in contents:
        if isinstance(item, str):
            parts.append(item)
        elif getattr(item, "text", None):
            parts.append(item.text)
    return "\n".join(parts)


def _extract_json_block(prompt: str, start_marker: str, end_marker: str) -> str:
    start = prompt.index(start_marker) + len(start_marker)
    end = prompt.index(end_marker, start)
    return prompt[start:end]


class FakeModels:
    def __init__(self, client):
        self.client = client

    def generate_content(self, model, contents, config=None):
        self.client.calls.append({"model": model, "config": config})
        prompt = _contents_text(contents)
        return SimpleNamespace(text=self.client.respond(prompt))

    def count_tokens(self, model, contents):
        return SimpleNamespace(total_tokens=len(_contents_text(contents)) // 4)


class FakeGeminiClient:
    """Stands in for google.genai.Client and answers every prompt the code sends."""

    def __init__(self, api_key=None, **kwargs):
        self.api_key = api_key
        self.calls = []
        self.models = FakeModels(self)
        self.plan = None  # set by tests to force a query plan

    # -- dispatch -----------------------------------------------------------
    def respond(self, prompt: str) -> str:
        if "Query Planner" in prompt:
            return json.dumps(self.plan or self._default_plan(prompt))
        if "entity resolution and deduplication" in prompt:
            return self._dedup(prompt)
        if "organizational analyst" in prompt:
            return self._summaries(prompt)
        return self._extraction(prompt)

    def _extraction(self, prompt: str) -> str:
        if "SHEET: Related Parties" in prompt:
            key = "related_parties"
        elif "Minutes of the" in prompt:
            key = "board_minutes"
        elif "Annual Report" in prompt:
            key = "annual_report"
        else:
            key = "letterhead"
        # Wrap in a code fence like the real model often does
        return "```json\n" + json.dumps(EXTRACTIONS[key]) + "\n```"

    def _dedup(self, prompt: str) -> str:
        data = json.loads(_extract_json_block(prompt, "```json\n", "\n```"))
        groups = {}
        for ent in data["entities"]:
            name = str(ent["attributes"].get("name", "")).lower()
            key = (ent["type"], CANONICAL.get(name, name))
            groups.setdefault(key, []).append(ent["id"])
        duplicates = [
            {
                "cluster_id": i,
                "entities": ids,
                "analysis": f"Same {key[0]} named {key[1]}",
                "deduplication_decision": "MERGE - same entity",
            }
            for i, (key, ids) in enumerate(sorted(groups.items()), 1)
            if len(ids) > 1
        ]
        return json.dumps({"duplicates": duplicates, "summary": {"total_clusters": len(duplicates)}})

    def _summaries(self, prompt: str) -> str:
        data = json.loads(_extract_json_block(prompt, "COMMUNITIES DATA:\n", "\n\nOUTPUT REQUIREMENTS"))
        return json.dumps({
            c["community_id"]: f"Community of {c['size']} entities including {', '.join(c['top_members'])}."
            for c in data
        })

    @staticmethod
    def _default_plan(prompt: str) -> dict:
        query = re.search(r'USER QUERY:\n"(.*)"', prompt).group(1)
        return {"query_mode": "flashlight", "entities": [query],
                "entity_types": [], "relationship_types": []}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_embedder(monkeypatch):
    """Replace sentence-transformers everywhere it is used."""
    import src.milvus_ingestion as milvus_ingestion
    monkeypatch.setattr(milvus_ingestion, "SentenceTransformer", FakeSentenceTransformer)
    try:
        import src.methodologies.semantic as semantic
        monkeypatch.setattr(semantic, "SentenceTransformer", FakeSentenceTransformer)
    except ImportError:
        pass
    return FakeSentenceTransformer


@pytest.fixture
def fake_gemini(monkeypatch):
    """Patch google.genai.Client so every module gets the same fake instance."""
    from google import genai

    client = FakeGeminiClient()
    monkeypatch.setattr(genai, "Client", lambda *a, **k: client)
    return client


@pytest.fixture
def tmp_outputs(monkeypatch, tmp_path):
    """Point every Config output directory at tmp_path."""
    from src.config import Config

    outputs = tmp_path / "outputs"
    dirs = {
        "OUTPUTS_DIR": outputs,
        "PARSED_DOCS_DIR": outputs / "parsed_documents",
        "ENTITIES_EXTRACTED_DIR": outputs / "entities_extracted",
        "KNOWLEDGE_GRAPHS_DIR": outputs / "knowledge_graphs",
        "REASONING_DIR": outputs / "reasoning",
    }
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Config, name, path)
    monkeypatch.setattr(Config, "DATASET_DIR", tmp_path / "dataset")
    return outputs


@pytest.fixture
def dataset_dir(tmp_path):
    """A dataset folder containing copies of the synthetic fixture documents."""
    target = tmp_path / "dataset"
    target.mkdir(exist_ok=True)
    for name in FIXTURE_FILES:
        shutil.copy(FIXTURES_DIR / name, target / name)
    return target


def build_raw_graph():
    """Non-deduplicated graph built from the canned extractions (one node per mention)."""
    from src.kg_builder import KnowledgeGraphBuilder

    builder = KnowledgeGraphBuilder()
    for doc, result in EXTRACTIONS.items():
        builder.add_entities(
            json.loads(json.dumps(result["entities"])),
            json.loads(json.dumps(result["relationships"])),
            source_doc=f"acme_{doc}",
        )
    return builder


@pytest.fixture
def raw_builder():
    return build_raw_graph()


def dedup_by_name(builder) -> nx.DiGraph:
    """Deduplicate the raw graph by (type, canonical name) - the expected answer."""
    groups = {}
    for node, attrs in builder.graph.nodes(data=True):
        name = str(attrs.get("name", "")).lower()
        groups.setdefault((attrs["type"], CANONICAL.get(name, name)), set()).add(node)
    clusters = [ids for ids in groups.values() if len(ids) > 1]
    return builder.create_deduplicated_graph(clusters)


@pytest.fixture
def kg_db(tmp_path, raw_builder):
    """SQLite knowledge graph (schema.sql) loaded through kg_to_sql."""
    import pickle
    import kg_to_sql

    graph_path = tmp_path / "dedup_graph.gpickle"
    with open(graph_path, "wb") as f:
        pickle.dump(dedup_by_name(raw_builder), f)

    db_path = tmp_path / "knowledge_graph.db"
    kg_to_sql.convert_graph_to_sql(graph_path, db_path)
    return db_path


def entity_id(db_path, name) -> str:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT unique_entity_id FROM entities WHERE canonical_name = ?", (name,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()
