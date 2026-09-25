"""Fuzzy string deduplication: normalised names compared with token-sort similarity.

Two mentions of the same entity type are merged when their normalised names are
identical, or when rapidfuzz's ``token_sort_ratio`` between them reaches ``threshold``.
Normalisation lowercases, strips honorifics and "M/s", folds "&" to "and", removes
punctuation, and maps legal suffixes ("Private Limited" -> "pvt ltd", "Limited" -> "ltd").

This is the cheapest method in the suite and needs no model. On the benchmark corpus it
resolves extracted mentions with higher precision than the LLM method when the LLM is a
small local model, so it is a sensible default for such deployments.
"""

from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Any, Dict, List, Set

import networkx as nx

logger = logging.getLogger(__name__)

HONORIFICS = {"mr", "mrs", "ms", "dr", "shri", "smt", "messrs", "miss"}
_LEGAL_SUFFIXES = [
    (r"\bprivate limited\b", "pvt ltd"),
    (r"\bprivate ltd\b", "pvt ltd"),
    (r"\bpvt limited\b", "pvt ltd"),
    (r"\blimited\b", "ltd"),
]


def normalize_name(text: Any) -> str:
    """"M/s Kestrel Alloys LLP" -> "kestrel alloys llp"; "Larkspur Castings Pvt. Ltd." -> "larkspur castings pvt ltd"."""
    if text is None:
        return ""
    s = str(text).lower()
    s = re.sub(r"\bm/s\.?", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " ".join(t for t in s.split() if t not in HONORIFICS)
    for pattern, repl in _LEGAL_SUFFIXES:
        s = re.sub(pattern, repl, s)
    return s


class FuzzyDeduplicator:
    """Merge mentions of one entity type whose normalised names match exactly or fuzzily."""

    def __init__(self, threshold: float = 90.0):
        self.threshold = threshold

    def deduplicate(self, graph: nx.DiGraph) -> List[Set[str]]:
        from rapidfuzz import fuzz

        by_type: Dict[str, List[str]] = {}
        norm: Dict[str, str] = {}
        for node_id, attrs in graph.nodes(data=True):
            name = normalize_name(attrs.get("name") or attrs.get("canonical_name") or "")
            if not name:
                continue
            norm[node_id] = name
            by_type.setdefault(str(attrs.get("type", "Unknown")), []).append(node_id)

        parent = {n: n for n in norm}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for members in by_type.values():
            for a, b in combinations(sorted(members), 2):
                if norm[a] == norm[b] or fuzz.token_sort_ratio(norm[a], norm[b]) >= self.threshold:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra

        groups: Dict[str, Set[str]] = {}
        for n in norm:
            groups.setdefault(find(n), set()).add(n)
        clusters = [g for g in groups.values() if len(g) > 1]
        logger.info(f"Fuzzy deduplication: {len(clusters)} clusters from {len(norm)} named nodes")
        return clusters

    def get_methodology_name(self) -> str:
        return "fuzzy"

    def get_parameters(self) -> Dict[str, Any]:
        return {"threshold": self.threshold, "scorer": "rapidfuzz.token_sort_ratio", "scope": "same entity type"}
