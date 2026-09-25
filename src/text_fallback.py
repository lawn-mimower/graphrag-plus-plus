"""Passage retrieval over the source documents, used when the graph cannot answer.

The orchestrator returns cited subgraphs. When intent extraction finds no usable
entities, or the traversal returns an empty subgraph, it previously gave up with
``status: ambiguous``. This module gives it a second route: a small BM25 index over
the parsed document text, so the caller still receives cited passages to answer from.

No third-party dependencies: the BM25 here is a plain Okapi implementation.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.&'-][a-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def chunk_words(text: str, size: int, overlap: int) -> List[str]:
    """Split text into windows of ``size`` words that overlap by ``overlap`` words."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


class BM25:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [list(d) for d in docs]
        self.n = len(self.docs)
        self.avg_len = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query: Sequence[str], idx: int) -> float:
        tf, length = self.tf[idx], len(self.docs[idx])
        total = 0.0
        for term in query:
            if term not in tf:
                continue
            f = tf[term]
            denom = f + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            total += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
        return total

    def top_k(self, query: Sequence[str], k: int) -> List[int]:
        scored = [(self.score(query, i), i) for i in range(self.n)]
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [i for s, i in scored[:k] if s > 0]


class TextFallbackIndex:
    """BM25 over word chunks of every document, keyed by document name."""

    def __init__(self, texts: Dict[str, str], chunk_size: int = 180, overlap: int = 40):
        self.chunks: List[Dict[str, object]] = []
        for doc, text in texts.items():
            for n, chunk in enumerate(chunk_words(text, chunk_size, overlap)):
                self.chunks.append({"document": doc, "chunk": n, "text": chunk})
        self.bm25 = BM25([tokenize(c["text"]) for c in self.chunks])
        logger.info(f"Text fallback index: {len(texts)} documents, {len(self.chunks)} chunks")

    def search(self, query: str, k: int = 5) -> List[Dict[str, object]]:
        ids = self.bm25.top_k(tokenize(query), k)
        return [{**self.chunks[i], "score": round(self.bm25.score(tokenize(query), i), 3)} for i in ids]

    @classmethod
    def from_directory(cls, directory: Path | str, **kwargs) -> "TextFallbackIndex":
        """Load parsed documents from a folder of .txt/.md files, or .json page lists."""
        texts: Dict[str, str] = {}
        for path in sorted(Path(directory).iterdir()):
            if path.suffix.lower() in (".txt", ".md"):
                texts[path.stem] = path.read_text(encoding="utf-8", errors="ignore")
            elif path.suffix.lower() == ".json":
                texts[path.stem] = _json_text(path)
        return cls(texts, **kwargs)


def _json_text(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    pages: Iterable = data.get("pages", data) if isinstance(data, dict) else data
    parts = []
    for page in pages:
        if isinstance(page, str):
            parts.append(page)
        elif isinstance(page, dict):
            parts.append(str(page.get("text") or page.get("content") or ""))
    return "\n".join(parts)
