"""
Live end-to-end test against the Gemini API (two generate_content calls).

Skipped unless GOOGLE_API_KEY is set; excluded from the default run.
Run with:  pytest -m e2e
Model names can be overridden with GEMINI_MODEL_HEAVY.
"""

import os
import pickle

import pytest

from conftest import FIXTURES_DIR

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"),
]


def test_live_pdf_to_answer(tmp_path, tmp_outputs):
    pytest.importorskip("rapidocr_onnxruntime")
    pytest.importorskip("leidenalg")
    from google import genai

    import kg_to_sql
    import scripts.run_leiden as run_leiden
    from src.config import Config
    from src.entity_extractor import MultimodalEntityExtractor
    from src.kg_builder import KnowledgeGraphBuilder
    from src.methodologies.rswoosh import RSwooshDeduplicator
    from src.milvus_ingestion import MilvusIngestionEngine
    from src.orchestrator import QueryOrchestrator
    from src.rapidocr_parser import RapidOCRParser
    from src.token_manager import TokenManager

    # Parse + extract (Gemini call 1)
    pages = RapidOCRParser().parse(FIXTURES_DIR / "acme_annual_report.pdf")
    client = genai.Client(api_key=Config.GOOGLE_API_KEY)
    result = MultimodalEntityExtractor(client, TokenManager(client)).extract_entities(
        pages, "acme_annual_report"
    )
    names = {str(e.get("attributes", {}).get("name", "")).lower() for e in result["entities"]}
    assert any("acme widgets" in n for n in names)
    assert any("director a" in n for n in names)
    assert result["relationships"]

    # Graph, dedup, SQL, vectors, communities
    builder = KnowledgeGraphBuilder()
    builder.add_entities(result["entities"], result["relationships"], source_doc="acme_annual_report")
    dedup = builder.create_deduplicated_graph(RSwooshDeduplicator().deduplicate(builder.graph))
    graph_path = tmp_path / "graph.gpickle"
    with open(graph_path, "wb") as f:
        pickle.dump(dedup, f)
    db_path = tmp_path / "knowledge_graph.db"
    kg_to_sql.convert_graph_to_sql(graph_path, db_path)

    milvus_path = str(tmp_path / "milvus_orchestrator.db")
    engine = MilvusIngestionEngine(db_path=str(db_path), milvus_path=milvus_path)
    assert engine.ingest_from_sql()["ingested"] > 0
    engine.close()
    assert run_leiden.run_leiden_pipeline(str(db_path), force=True, skip_summaries=True)

    # Natural-language query (Gemini call 2)
    orch = QueryOrchestrator(db_path=str(db_path), milvus_path=milvus_path)
    try:
        answer = orch.process_query(
            "How is Director A connected to Beta Supplies LLP?", ["UNCLASSIFIED"], max_depth=2
        )
    finally:
        orch.close()

    assert answer["status"] == "success", answer
    node_names = " ".join(n["name"].lower() for n in answer["nodes"].values())
    assert "director a" in node_names
    assert answer["mini_graph"]
    assert answer["citation_context"]
