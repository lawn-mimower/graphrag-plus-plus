from pathlib import Path

from src.text_fallback import BM25, TextFallbackIndex, chunk_words, tokenize


def test_chunk_words_overlaps():
    chunks = chunk_words(" ".join(str(i) for i in range(10)), size=4, overlap=1)
    assert chunks[0] == "0 1 2 3"
    assert chunks[1] == "3 4 5 6"


def test_bm25_ranks_matching_document_first():
    docs = [tokenize(t) for t in ["the auditor signed the report", "the loan was sanctioned by the bank", "board minutes of july"]]
    index = BM25(docs)
    assert index.top_k(tokenize("who sanctioned the loan"), 2)[0] == 1
    assert index.top_k(tokenize("nothing matches here"), 2) == []


def test_index_search_returns_cited_passages():
    index = TextFallbackIndex({
        "minutes": "The board approved the related-party schedule. Director A chaired the meeting.",
        "letter": "Seabright Finance sanctioned a term loan of INR 4 crore to Larkspur Castings.",
    }, chunk_size=6, overlap=2)
    hits = index.search("term loan sanctioned to Larkspur", k=2)
    assert hits and hits[0]["document"] == "letter"
    assert "loan" in hits[0]["text"]
    assert hits[0]["score"] > 0


def test_from_directory_reads_text_and_json(tmp_path: Path):
    (tmp_path / "a.md").write_text("Director B resigned in March.", encoding="utf-8")
    (tmp_path / "b.json").write_text('{"pages": [{"text": "Authorised capital is INR 10 crore."}]}', encoding="utf-8")
    index = TextFallbackIndex.from_directory(tmp_path, chunk_size=5, overlap=1)
    assert {c["document"] for c in index.chunks} == {"a", "b"}
    assert index.search("authorised capital", k=1)[0]["document"] == "b"
