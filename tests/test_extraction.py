"""Token chunking and Gemini entity extraction (with a fake client)."""

import json

from PIL import Image

from conftest import FakeGeminiClient


def _pages(n, text="Acme Widgets Pvt Ltd - Annual Report page"):
    return [
        (Image.new("RGB", (20, 20), "white"), f"{text} {i}", {"page_number": i})
        for i in range(1, n + 1)
    ]


def test_single_chunk_when_under_limit():
    from src.token_manager import TokenManager

    tm = TokenManager(FakeGeminiClient())
    chunks = tm.chunk_multimodal_content(_pages(3), "Extract entities")
    assert len(chunks) == 1
    # instruction + (text, image) per page
    assert len(chunks[0]) == 1 + 3 * 2


def test_chunking_keeps_page_pairs_together():
    from src.token_manager import TokenManager

    tm = TokenManager(FakeGeminiClient())
    tm.max_tokens = 8000  # about two page images per chunk
    chunks = tm.chunk_multimodal_content(_pages(5), "Extract entities")
    assert len(chunks) == 3
    for chunk in chunks:
        body = chunk[1:]
        assert len(body) % 2 == 0
        for text_part, image_part in zip(body[::2], body[1::2]):
            assert "OCR Text (PRIMARY)" in text_part.text
            assert image_part.inline_data.mime_type == "image/png"


def test_extract_entities_parses_fenced_json(tmp_path):
    from src.entity_extractor import MultimodalEntityExtractor
    from src.token_manager import TokenManager

    client = FakeGeminiClient()
    extractor = MultimodalEntityExtractor(client, TokenManager(client))
    save_path = tmp_path / "report_entities.json"

    result = extractor.extract_entities(_pages(1), "acme_annual_report", save_path)

    names = {e["attributes"]["name"] for e in result["entities"]}
    assert {"Acme Widgets Pvt Ltd", "Director A", "Director B", "Beta Supplies LLP"} <= names
    assert len(result["relationships"]) == 5
    assert json.loads(save_path.read_text()) == result
    assert client.calls[0]["model"] == extractor.model_name


def test_extract_entities_handles_empty_and_invalid_responses():
    from src.entity_extractor import MultimodalEntityExtractor
    from src.token_manager import TokenManager

    for reply in ["", "not json at all"]:
        client = FakeGeminiClient()
        client.respond = lambda prompt, reply=reply: reply
        extractor = MultimodalEntityExtractor(client, TokenManager(client))
        assert extractor.extract_entities(_pages(1), "doc") == {"entities": [], "relationships": []}


def test_multi_chunk_ids_are_remapped():
    from src.entity_extractor import MultimodalEntityExtractor
    from src.token_manager import TokenManager

    client = FakeGeminiClient()
    tm = TokenManager(client)
    tm.max_tokens = 5000  # one page per chunk
    extractor = MultimodalEntityExtractor(client, tm)

    result = extractor.extract_entities(_pages(2), "acme_annual_report")
    ids = [e["id"] for e in result["entities"]]
    assert len(ids) == len(set(ids))
    assert any(i.endswith("_chunk0") for i in ids) and any(i.endswith("_chunk1") for i in ids)
    for rel in result["relationships"]:
        assert rel["from_id"] in ids and rel["to_id"] in ids
