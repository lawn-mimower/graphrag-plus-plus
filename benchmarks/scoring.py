"""
Scoring and baseline helpers for the benchmark. No LLM calls, no I/O.

- Name normalisation and an alias index that finds gold entities in free text.
- Entity-resolution metrics: pairwise precision/recall/F1 and cluster purity.
- The two conventional entity-resolution baselines: exact match after
  normalisation, and rapidfuzz token_sort_ratio at a fixed threshold.
- Answer scoring: exact match, lenient match, entity-set F1.
- Retrieval helpers for the basic-RAG baseline: word chunking and BM25.
"""

import math
import re
from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Set

HONORIFICS = {"mr", "mrs", "ms", "dr", "shri", "smt", "messrs", "miss"}
_LEGAL_SUFFIXES = [
    (r"\bprivate limited\b", "pvt ltd"),
    (r"\bprivate ltd\b", "pvt ltd"),
    (r"\bpvt limited\b", "pvt ltd"),
    (r"\blimited\b", "ltd"),
]
ABSTAIN_PREFIXES = ("not found", "unknown", "i dont know", "i do not know", "no answer",
                    "cannot be determined", "not mentioned", "not stated", "not available")


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def normalize_name(text: str) -> str:
    """Lowercase, drop honorifics and punctuation, canonicalise legal suffixes.

    "M/s Kestrel Alloys LLP" -> "kestrel alloys llp"
    "Larkspur Castings Pvt. Ltd." -> "larkspur castings pvt ltd"
    "Seabright Co-operative Bank Ltd" -> "seabright cooperative bank ltd"
    """
    if text is None:
        return ""
    s = str(text).lower()
    s = re.sub(r"\bm/s\.?", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)          # 12,600,000 -> 12600000
    s = re.sub(r"[.'’-]", "", s)               # Pvt. -> pvt, Co-operative -> cooperative
    s = re.sub(r"[^a-z0-9]+", " ", s)
    tokens = [t for t in s.split() if t not in HONORIFICS]
    s = " ".join(tokens)
    for pattern, repl in _LEGAL_SUFFIXES:
        s = re.sub(pattern, repl, s)
    return s


def normalize_text(text: str) -> str:
    """Same normalisation, applied to answers and contexts."""
    return normalize_name(text)


# --------------------------------------------------------------------------
# Alias index
# --------------------------------------------------------------------------
class AliasIndex:
    """Maps normalised aliases to gold entity IDs and finds entities in text.

    Aliases that normalise to the same string for two entities, and surfaces
    declared ambiguous ("A. Mehta"), are never used for recognition.
    """

    def __init__(self, entities: Sequence[dict]):
        owners: Dict[str, Set[str]] = {}
        for ent in entities:
            for alias in [ent.get("canonical_name", "")] + list(ent.get("aliases", [])):
                norm = normalize_name(alias)
                if norm:
                    owners.setdefault(norm, set()).add(ent["id"])
        ambiguous = {normalize_name(a) for ent in entities for a in ent.get("ambiguous_surfaces", [])}
        self.alias_to_entity = {a: next(iter(ids)) for a, ids in owners.items()
                                if len(ids) == 1 and a not in ambiguous}
        self._by_len: Dict[int, Dict[tuple, str]] = {}
        for alias, eid in self.alias_to_entity.items():
            toks = tuple(alias.split())
            self._by_len.setdefault(len(toks), {})[toks] = eid
        self._lengths = sorted(self._by_len, reverse=True)
        self.entity_type = {ent["id"]: ent.get("type") for ent in entities}
        self.canonical = {ent["id"]: ent.get("canonical_name") for ent in entities}

    def aliases_of(self, entity_id: str) -> Set[str]:
        return {a for a, e in self.alias_to_entity.items() if e == entity_id}

    def recognize(self, text: str) -> List[str]:
        """Entity IDs whose aliases occur in text (whole tokens, longest match first)."""
        tokens = normalize_text(text).split()
        found: List[str] = []
        i = 0
        while i < len(tokens):
            for length in self._lengths:
                if i + length <= len(tokens):
                    eid = self._by_len[length].get(tuple(tokens[i:i + length]))
                    if eid:
                        if eid not in found:
                            found.append(eid)
                        i += length
                        break
            else:
                i += 1
        return found

    def align(self, name: str) -> Optional[str]:
        """Gold entity for an extracted entity name: exact alias, else a single recognised entity."""
        norm = normalize_name(name)
        if norm in self.alias_to_entity:
            return self.alias_to_entity[norm]
        found = self.recognize(name)
        return found[0] if len(found) == 1 else None


# --------------------------------------------------------------------------
# Entity-resolution metrics and baselines
# --------------------------------------------------------------------------
def clusters_to_labels(clusters: Iterable[Iterable[str]], items: Iterable[str]) -> Dict[str, int]:
    """Cluster label per item. Items in no cluster are singletons; an item in
    several clusters keeps the first."""
    labels: Dict[str, int] = {}
    next_label = 0
    for cluster in clusters:
        members = [m for m in cluster if m not in labels]
        if len(members) < 1:
            continue
        for m in members:
            labels[m] = next_label
        next_label += 1
    for item in items:
        if item not in labels:
            labels[item] = next_label
            next_label += 1
    return labels


def _pairs(labels: Dict[str, int], items: Sequence[str]) -> Set[tuple]:
    by_label: Dict[int, List[str]] = {}
    for item in items:
        by_label.setdefault(labels[item], []).append(item)
    pairs = set()
    for members in by_label.values():
        for a, b in combinations(sorted(members), 2):
            pairs.add((a, b))
    return pairs


def resolution_scores(pred_clusters: Iterable[Iterable[str]], gold: Dict[str, str]) -> dict:
    """Pairwise precision/recall/F1 and purity of predicted clusters.

    gold maps each mention to its entity. Only mentions in gold are scored;
    predicted clusters are restricted to them.
    """
    items = sorted(gold)
    item_set = set(items)
    pred = clusters_to_labels([[m for m in c if m in item_set] for c in pred_clusters], items)
    gold_labels = {m: gold[m] for m in items}
    pred_pairs = _pairs(pred, items)
    gold_pairs = _pairs(gold_labels, items)
    true_pairs = pred_pairs & gold_pairs
    precision = len(true_pairs) / len(pred_pairs) if pred_pairs else None
    recall = len(true_pairs) / len(gold_pairs) if gold_pairs else None
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    def _purity(a: Dict[str, object], b: Dict[str, object]) -> float:
        groups: Dict[object, Counter] = {}
        for m in items:
            groups.setdefault(a[m], Counter())[b[m]] += 1
        return sum(c.most_common(1)[0][1] for c in groups.values()) / len(items) if items else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "purity": _purity(pred, gold_labels),
        "inverse_purity": _purity(gold_labels, pred),
        "predicted_pairs": len(pred_pairs), "gold_pairs": len(gold_pairs), "true_pairs": len(true_pairs),
        "predicted_clusters": len(set(pred.values())), "gold_clusters": len(set(gold_labels.values())),
        "mentions": len(items),
    }


def exact_match_clusters(names: Dict[str, str]) -> List[Set[str]]:
    """Baseline: mentions whose normalised names are identical are merged."""
    groups: Dict[str, Set[str]] = {}
    for mid, name in names.items():
        key = normalize_name(name)
        if key:
            groups.setdefault(key, set()).add(mid)
    return [g for g in groups.values() if len(g) > 1]


def fuzzy_match_clusters(names: Dict[str, str], threshold: float = 90.0) -> List[Set[str]]:
    """Baseline: rapidfuzz token_sort_ratio >= threshold on normalised names,
    merged transitively (single link)."""
    from rapidfuzz import fuzz

    ids = sorted(names)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    norm = {i: normalize_name(names[i]) for i in ids}
    for a, b in combinations(ids, 2):
        if norm[a] and norm[b] and fuzz.token_sort_ratio(norm[a], norm[b]) >= threshold:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    groups: Dict[str, Set[str]] = {}
    for i in ids:
        groups.setdefault(find(i), set()).add(i)
    return [g for g in groups.values() if len(g) > 1]


# --------------------------------------------------------------------------
# Answer scoring
# --------------------------------------------------------------------------
_UNITS = {"crore": 1e7, "crores": 1e7, "cr": 1e7, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
          "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9, "thousand": 1e3}
_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(crores?|cr|lakhs?|lacs?|million|mn|billion|bn|thousand)?\b",
                     re.IGNORECASE)


def parse_numbers(text: str) -> List[float]:
    """Numbers in text, with Indian and Western digit grouping and unit words.

    "INR 1,26,00,000" -> 12600000, "12.6 million" -> 12600000, "62%" -> 62
    """
    values = []
    for match in _NUMBER.finditer(text or ""):
        raw, unit = match.group(1), (match.group(2) or "").lower()
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        values.append(value * _UNITS.get(unit, 1.0))
    return values


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def parse_yes_no(text: str) -> Optional[str]:
    tokens = normalize_text(text).split()
    if not tokens:
        return None
    if tokens[0] in ("yes", "no"):
        return tokens[0]
    if "not" in tokens or "no" in tokens or "neither" in tokens:
        return "no"
    if "yes" in tokens:
        return "yes"
    return None


def is_abstention(text: str) -> bool:
    norm = normalize_text(text)
    return not norm or norm.startswith(ABSTAIN_PREFIXES)


def set_f1(predicted: Set[str], gold: Set[str]) -> float:
    if not predicted or not gold:
        return 0.0
    tp = len(predicted & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(predicted), tp / len(gold)
    return 2 * p * r / (p + r)


def score_answer(question: dict, answer: str, index: AliasIndex) -> dict:
    """Score one answer against the gold answer.

    correct: lenient match: the gold entity is named (among others), the gold
             number or ID appears, the yes/no reading matches. For lists: the
             named entities are exactly the gold set.
    em:      the whole normalised answer is exactly a gold alias / number / ID /
             yes-no. For lists: same as correct.
    f1:      entity-set F1 between the entities named in the answer (minus
             those named in the question) and the gold set. Entity types only.
    score:   the headline metric, which penalises hedged answers: f1 for entity
             and list questions; for IDs, correct only if no other ID-like
             token is given; correct (0/1) for numbers and yes/no.
    """
    atype, gold = question["answer_type"], question["answer"]
    answer = answer or ""
    norm = normalize_text(answer).strip()
    abstained = is_abstention(answer)
    result = {"abstained": abstained, "correct": False, "em": False, "f1": None}
    if abstained:
        result["score"] = 0.0
        if atype.startswith("entity"):
            result["f1"] = 0.0
        return result

    if atype.startswith("entity"):
        in_question = set(index.recognize(question["question"]))
        named = [e for e in index.recognize(answer) if e not in in_question]
        result["named_entities"] = named
        gold_set = set(gold)
        named_set = set(named)
        if atype == "entity":
            result["correct"] = gold_set <= named_set
            result["em"] = any(norm == a for g in gold for a in index.aliases_of(g))
            result["f1"] = set_f1(named_set, gold_set)
        elif atype == "entity_any":
            result["correct"] = bool(gold_set & named_set)
            result["em"] = any(norm == a for g in gold for a in index.aliases_of(g))
            result["f1"] = max(set_f1(named_set, {g}) for g in gold)
        else:  # entity_list
            result["f1"] = set_f1(named_set, gold_set)
            result["correct"] = result["em"] = named_set == gold_set
    elif atype == "number":
        values = parse_numbers(answer)
        result["correct"] = any(_close(v, float(gold)) for v in values)
        leftover = [t for t in norm.split() if not re.fullmatch(r"[\d]+", t)
                    and t not in {"inr", "rs", "rupees", "percent", "per", "cent"}]
        result["em"] = len(values) == 1 and _close(values[0], float(gold)) and not leftover
    elif atype == "string":
        tokens = norm.split()
        target = normalize_text(str(gold))
        result["correct"] = target in tokens
        result["em"] = norm == target
        others = {t for t in tokens if t != target and re.fullmatch(r"\d{6,}", t)}
        result["strict"] = result["correct"] and not others
    elif atype == "yes_no":
        reading = parse_yes_no(answer)
        result["reading"] = reading
        result["correct"] = reading == gold
        result["em"] = norm in ("yes", "no") and norm == gold
    else:
        raise ValueError(f"unknown answer_type {atype}")

    if atype.startswith("entity"):
        result["score"] = result["f1"]
    elif atype == "string":
        result["score"] = float(result["strict"])
    else:
        result["score"] = float(result["correct"])
    return result


# --------------------------------------------------------------------------
# Retrieval diagnostics (no LLM)
# --------------------------------------------------------------------------
def context_coverage(question: dict, context: str, index: AliasIndex) -> dict:
    """Does the context handed to the answering model contain what is needed?

    evidence_recall: share of the question's evidence entities named in the context.
    answer_in_context: the gold answer (entities, number or ID) appears in the
    context; None for yes/no questions.
    """
    named = set(index.recognize(context))
    evidence = question["evidence_entities"]
    recall = sum(e in named for e in evidence) / len(evidence) if evidence else None
    atype, gold = question["answer_type"], question["answer"]
    if atype in ("entity", "entity_list"):
        in_ctx = set(gold) <= named
    elif atype == "entity_any":
        in_ctx = bool(set(gold) & named)
    elif atype == "number":
        in_ctx = any(_close(v, float(gold)) for v in parse_numbers(context))
    elif atype == "string":
        in_ctx = normalize_text(str(gold)) in normalize_text(context).split()
    else:
        in_ctx = None
    return {"evidence_recall": recall, "answer_in_context": in_ctx}


def subgraph_coverage(question: dict, node_names: Iterable[str], edges: Iterable[tuple],
                      index: AliasIndex) -> dict:
    """Structural hit rate for a returned subgraph.

    answer_entities_in_subgraph: all gold answer entities are returned nodes
    (entity questions only). evidence_recall: share of evidence entities among
    the nodes. path_hit: every pair of some alternative evidence path is joined
    by a returned edge (either direction). Relation types in the gold path are
    ignored: extracted relation names are free text.
    """
    node_entities: Set[str] = set()
    for name in node_names:
        node_entities.update(index.recognize(name))
    edge_pairs = set()
    for src, dst in edges:
        for a in index.recognize(src):
            for b in index.recognize(dst):
                edge_pairs.add((a, b))
                edge_pairs.add((b, a))
    atype, gold = question["answer_type"], question["answer"]
    if atype in ("entity", "entity_list"):
        answer_hit = set(gold) <= node_entities
    elif atype == "entity_any":
        answer_hit = bool(set(gold) & node_entities)
    else:
        answer_hit = None
    evidence = question["evidence_entities"]
    paths = question.get("evidence_paths") or []
    path_hit = any(all((p[0], p[1]) in edge_pairs for p in alt) for alt in paths) if paths else None
    return {
        "answer_entities_in_subgraph": answer_hit,
        "evidence_recall": sum(e in node_entities for e in evidence) / len(evidence) if evidence else None,
        "path_hit": path_hit,
    }


# --------------------------------------------------------------------------
# Basic-RAG helpers
# --------------------------------------------------------------------------
def chunk_words(text: str, size: int = 100, overlap: int = 20) -> List[str]:
    """Fixed-size word windows with overlap."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


def bm25_tokens(text: str) -> List[str]:
    return normalize_text(text).split()


class BM25:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [list(d) for d in docs]
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.n if self.n else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(t for d in self.docs for t in set(d))
        self.idf = {t: math.log((self.n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def scores(self, query: Sequence[str]) -> List[float]:
        out = []
        for tf, doc in zip(self.tf, self.docs):
            s = 0.0
            for t in query:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            out.append(s)
        return out

    def top_k(self, query: Sequence[str], k: int) -> List[int]:
        scores = self.scores(query)
        return sorted(range(self.n), key=lambda i: (-scores[i], i))[:k]
