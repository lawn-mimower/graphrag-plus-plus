"""Extraction must survive imperfect model replies instead of silently returning nothing."""
from types import SimpleNamespace

from src.entity_extractor import MultimodalEntityExtractor


class FakeModels:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_content(self, model=None, contents=None, config=None):
        self.calls.append(contents)
        return SimpleNamespace(text=self.replies.pop(0), candidates=[], prompt_feedback=None)


def make_extractor(replies):
    client = SimpleNamespace(models=FakeModels(replies))
    extractor = MultimodalEntityExtractor(client, token_manager=None)
    return extractor, client.models


PART = SimpleNamespace(text="Larkspur Castings Private Limited appointed Director A as managing director. " * 5)
GOOD = '```json\n{"entities": [{"id": "C1", "type": "Company", "attributes": {"name": "Larkspur Castings"}}], "relationships": []}\n```'


def test_trailing_comma_is_repaired():
    extractor, models = make_extractor([
        '{"entities": [{"id": "C1", "type": "Company", "attributes": {"name": "Larkspur Castings",}},], "relationships": [],}'
    ])
    result = extractor._extract_from_chunk([SimpleNamespace(text="instruction"), PART], "doc")
    assert [e["id"] for e in result["entities"]] == ["C1"]
    assert len(models.calls) == 1
    assert extractor.failed_documents == []


def test_empty_reply_is_retried_with_a_json_only_instruction():
    extractor, models = make_extractor(["", GOOD])
    result = extractor._extract_from_chunk([SimpleNamespace(text="instruction"), PART], "doc")
    assert len(result["entities"]) == 1
    assert len(models.calls) == 2
    assert "only the JSON object" in models.calls[1][-1].text
    assert extractor.failed_documents == []


def test_two_failures_are_recorded_not_hidden():
    extractor, models = make_extractor(["not json at all", ""])
    result = extractor._extract_from_chunk([SimpleNamespace(text="instruction"), PART], "board_minutes")
    assert result == {"entities": [], "relationships": []}
    assert extractor.failed_documents == ["board_minutes"]
    assert len(models.calls) == 2


def test_empty_result_from_a_short_chunk_is_accepted_without_retry():
    extractor, models = make_extractor(['{"entities": [], "relationships": []}'])
    result = extractor._extract_from_chunk([SimpleNamespace(text="instruction"), SimpleNamespace(text="Page 3")], "cover")
    assert result["entities"] == []
    assert len(models.calls) == 1
    assert extractor.failed_documents == []
