"""Deterministic relationship scoring for documents and chunks."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
_ROMAN_NUMERAL_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "about",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "how",
    "what",
    "when",
    "where",
    "why",
    "who",
    "will",
    "would",
    "should",
    "can",
    "could",
    "may",
    "might",
    "use",
    "using",
    "used",
    "all",
    "abstract",
    "any",
    "appendix",
    "per",
    "via",
    "new",
    "not",
    "but",
    "you",
    "its",
    "our",
    "their",
    "them",
    "they",
    "than",
    "then",
    "also",
    "more",
    "less",
    "after",
    "before",
    "been",
    "com",
    "document",
    "documents",
    "guide",
    "her",
    "him",
    "his",
    "overview",
    "contents",
    "figure",
    "html",
    "index",
    "introduction",
    "like",
    "markdown",
    "notice",
    "notes",
    "section",
    "table",
    "text",
    "txt",
    "project",
    "gutenberg",
    "ebook",
    "ebooks",
    "copyright",
    "foundation",
    "license",
}


@dataclass(frozen=True)
class RelationshipNode:
    """Text item to compare against other items."""

    key: str
    label: str
    path: str
    body: str
    group: str | None = None


@dataclass(frozen=True)
class Relationship:
    """Similarity signal from one node to another."""

    target_key: str
    score: float
    shared_terms: list[str]


def tokenize(text: str) -> list[str]:
    """Return normalized content words from text."""
    tokens: list[str] = []
    for word in _WORD_RE.findall(text.lower()):
        if len(word) < 3 or word.isdigit() or word in _STOPWORDS:
            continue
        if _ROMAN_NUMERAL_RE.fullmatch(word):
            continue
        tokens.append(word)
    return tokens


def extract_keywords(*texts: str, limit: int = 8) -> list[str]:
    """Extract deduplicated keywords from one or more text fields."""
    seen: set[str] = set()
    keywords: list[str] = []
    for text in texts:
        for token in tokenize(text):
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= limit:
                return keywords
    return keywords


def build_relationships(
    nodes: list[RelationshipNode],
    *,
    limit: int = 3,
    min_score: float = 0.08,
    require_different_group: bool = False,
) -> dict[str, list[Relationship]]:
    """Build lexical similarity relationships for the given nodes."""
    if not nodes:
        return {}

    term_counts = [Counter(tokenize(node.body)) for node in nodes]
    document_frequency: Counter[str] = Counter()
    for counts in term_counts:
        document_frequency.update(counts.keys())

    corpus_size = len(nodes)
    inverse_document_frequency = {
        term: math.log((corpus_size + 1) / (freq + 1)) + 1.0
        for term, freq in document_frequency.items()
    }
    weighted_terms = [
        {term: count * inverse_document_frequency[term] for term, count in counts.items()}
        for counts in term_counts
    ]
    vector_norms = [
        math.sqrt(sum(weight * weight for weight in weights.values()))
        for weights in weighted_terms
    ]
    salient_terms = [
        [
            term
            for term, _ in sorted(
                weights.items(),
                key=lambda item: (-item[1], item[0]),
            )[:12]
        ]
        for weights in weighted_terms
    ]

    term_index: dict[str, set[int]] = defaultdict(set)
    for index, terms in enumerate(salient_terms):
        for term in terms:
            term_index[term].add(index)

    relationships: dict[str, list[Relationship]] = {}
    for index, node in enumerate(nodes):
        candidates: set[int] = set()
        for term in salient_terms[index]:
            candidates.update(term_index[term])
        candidates.discard(index)

        ranked: list[Relationship] = []
        for candidate_index in candidates:
            other = nodes[candidate_index]
            if require_different_group and node.group == other.group:
                continue

            score = _cosine_similarity(
                weighted_terms[index],
                weighted_terms[candidate_index],
                vector_norms[index],
                vector_norms[candidate_index],
            )
            if score < min_score:
                continue

            shared_terms = _shared_terms(
                weighted_terms[index],
                weighted_terms[candidate_index],
            )
            if not shared_terms:
                continue

            ranked.append(
                Relationship(
                    target_key=other.key,
                    score=round(score, 3),
                    shared_terms=shared_terms,
                )
            )

        ranked.sort(key=lambda relationship: (-relationship.score, relationship.target_key))
        relationships[node.key] = ranked[:limit]

    return relationships


def _cosine_similarity(
    left: dict[str, float],
    right: dict[str, float],
    left_norm: float,
    right_norm: float,
) -> float:
    if left_norm == 0 or right_norm == 0:
        return 0.0

    if len(left) > len(right):
        left, right = right, left

    dot_product = 0.0
    for term, weight in left.items():
        dot_product += weight * right.get(term, 0.0)

    return dot_product / (left_norm * right_norm)


def _shared_terms(
    left: dict[str, float],
    right: dict[str, float],
    *,
    limit: int = 4,
) -> list[str]:
    shared = [
        (term, min(weight, right[term]))
        for term, weight in left.items()
        if term in right
    ]
    shared.sort(key=lambda item: (-item[1], item[0]))
    return [term for term, _ in shared[:limit]]
