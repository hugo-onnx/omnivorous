"""Release-gate helpers for corpus validation and retrieval evaluation."""

from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnivorous.relationships import tokenize

_RETRIEVAL_TOKEN_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)
_RETRIEVAL_FRAGMENT_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
_RETRIEVAL_SHORT_TOKENS = {"ai", "api", "db", "eu", "id", "ml", "sdk", "ui", "uk", "us", "ux"}
_RETRIEVAL_NOISE_TOKENS = {"doc", "docx", "html", "jpeg", "jpg", "json", "md", "pdf", "png", "txt"}


@dataclass(frozen=True)
class RetrievalCase:
    """Expected retrieval target for a natural-language query."""

    case_id: str
    query: str
    expected_document: str
    expected_chunk_heading_contains: str | None = None
    expected_chunk_path: str | None = None
    top_k: int = 3


@dataclass(frozen=True)
class RetrievalResult:
    """Outcome of running one retrieval case against a generated manifest."""

    case_id: str
    query: str
    expected_document: str
    document_rank: int | None
    chunk_rank: int | None
    passed: bool
    top_documents: list[str]
    top_chunks: list[str]


@dataclass(frozen=True)
class _SearchIndex:
    term_counts: dict[str, Counter[str]]
    inverse_document_frequency: dict[str, float]
    average_length: float


def load_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    """Load a manifest from either a pack directory or a manifest path."""
    manifest_path = path / "manifest.json" if path.is_dir() else path
    output_dir = manifest_path.parent
    return output_dir, json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_manifest(output_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Validate pack artifact integrity."""
    errors: list[str] = []

    for rel_path in manifest.get("output_files", []):
        if not (output_dir / rel_path).exists():
            errors.append(f"missing_output:{rel_path}")

    chunk_paths = {chunk["path"] for doc in manifest["documents"] for chunk in doc["chunks"]}
    doc_paths = {doc["full_path"] for doc in manifest["documents"]}
    valid_targets = chunk_paths | doc_paths

    for doc in manifest["documents"]:
        if not (output_dir / doc["full_path"]).exists():
            errors.append(f"missing_document:{doc['full_path']}")

        chunks = doc["chunks"]
        for index, chunk in enumerate(chunks):
            if not (output_dir / chunk["path"]).exists():
                errors.append(f"missing_chunk:{chunk['path']}")

            previous = chunks[index - 1]["path"] if index > 0 else None
            nxt = chunks[index + 1]["path"] if index + 1 < len(chunks) else None
            if chunk.get("previous") != previous:
                errors.append(f"bad_previous:{chunk['path']}")
            if chunk.get("next") != nxt:
                errors.append(f"bad_next:{chunk['path']}")

            for edge in chunk.get("related_chunks", []):
                if edge["target_path"] not in valid_targets:
                    errors.append(f"bad_chunk_target:{chunk['path']}->{edge['target_path']}")

        for edge in doc.get("related_documents", []):
            if edge["target_path"] not in valid_targets:
                errors.append(f"bad_document_target:{doc['full_path']}->{edge['target_path']}")

    return errors


def collect_manifest_metrics(manifest: dict[str, Any]) -> dict[str, float | int]:
    """Return core pack metrics used by release gates."""
    chunks = [chunk for doc in manifest["documents"] for chunk in doc["chunks"]]
    chunk_tokens = sorted(chunk["tokens_estimate"] for chunk in chunks)
    if chunk_tokens:
        p95_index = max(0, int(len(chunk_tokens) * 0.95) - 1)
        p95 = chunk_tokens[p95_index]
        avg = statistics.mean(chunk_tokens)
        median = statistics.median(chunk_tokens)
        max_tokens = chunk_tokens[-1]
    else:
        p95 = avg = median = max_tokens = 0

    return {
        "documents": len(manifest["documents"]),
        "chunks": len(chunks),
        "total_tokens": manifest["total_tokens"],
        "avg_chunk_tokens": round(avg, 1),
        "median_chunk_tokens": round(median, 1),
        "p95_chunk_tokens": p95,
        "max_chunk_tokens": max_tokens,
        "docs_without_headings": sum(1 for doc in manifest["documents"] if not doc.get("headings")),
        "docs_without_related": sum(
            1 for doc in manifest["documents"] if not doc.get("related_documents")
        ),
        "chunks_without_related": sum(
            1 for chunk in chunks if not chunk.get("related_chunks")
        ),
    }


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    """Load retrieval evaluation cases from JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        RetrievalCase(
            case_id=item["id"],
            query=item["query"],
            expected_document=item["expected_document"],
            expected_chunk_heading_contains=item.get("expected_chunk_heading_contains"),
            expected_chunk_path=item.get("expected_chunk_path"),
            top_k=item.get("top_k", 3),
        )
        for item in raw
    ]


def evaluate_retrieval(
    manifest: dict[str, Any],
    cases: list[RetrievalCase],
    *,
    output_dir: Path | None = None,
) -> list[RetrievalResult]:
    """Evaluate whether the manifest surfaces the expected doc/chunk near the top."""
    file_cache: dict[str, str] = {}
    results: list[RetrievalResult] = []
    all_chunks = [
        {
            **chunk,
            "document_title": doc["title"],
            "original_source": doc["original_source"],
            "document_path": doc["full_path"],
        }
        for doc in manifest["documents"]
        for chunk in doc["chunks"]
    ]
    document_index = _build_search_index(
        {
            doc["full_path"]: _document_search_text(doc, output_dir=output_dir, file_cache=file_cache)
            for doc in manifest["documents"]
        }
    )
    chunk_index = _build_search_index(
        {
            chunk["path"]: _chunk_search_text(chunk, output_dir=output_dir, file_cache=file_cache)
            for chunk in all_chunks
        }
    )

    for case in cases:
        ranked_chunks = _rank_chunks(all_chunks, case.query, chunk_index)
        top_chunk_scores = _top_chunk_scores_by_document(ranked_chunks)
        ranked_documents = _rank_documents(
            manifest["documents"],
            case.query,
            document_index,
            top_chunk_scores=top_chunk_scores,
        )

        doc_sources = [doc["original_source"] for doc in ranked_documents]
        chunk_ids = [chunk["path"] for chunk in ranked_chunks]
        document_rank = _rank_of(doc_sources, case.expected_document)
        chunk_rank = _match_chunk_rank(ranked_chunks, case)

        passed = document_rank is not None and document_rank <= case.top_k
        if case.expected_chunk_heading_contains or case.expected_chunk_path:
            passed = passed and chunk_rank is not None and chunk_rank <= case.top_k

        results.append(
            RetrievalResult(
                case_id=case.case_id,
                query=case.query,
                expected_document=case.expected_document,
                document_rank=document_rank,
                chunk_rank=chunk_rank,
                passed=passed,
                top_documents=doc_sources[: case.top_k],
                top_chunks=chunk_ids[: case.top_k],
            )
        )

    return results


def _rank_documents(
    documents: list[dict[str, Any]],
    query: str,
    search_index: _SearchIndex,
    *,
    top_chunk_scores: dict[str, float],
) -> list[dict[str, Any]]:
    query_tokens = _query_terms(query)
    ranked = sorted(
        documents,
        key=lambda doc: (
            -_document_score(
                doc,
                query,
                query_tokens,
                search_index,
                top_chunk_score=top_chunk_scores.get(doc["full_path"], 0.0),
            ),
            doc["original_source"],
        ),
    )
    return ranked


def _rank_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    search_index: _SearchIndex,
) -> list[dict[str, Any]]:
    query_tokens = _query_terms(query)
    scored_chunks = [
        {
            **chunk,
            "_retrieval_score": _chunk_score(chunk, query, query_tokens, search_index),
        }
        for chunk in chunks
    ]
    ranked = sorted(
        scored_chunks,
        key=lambda chunk: (
            -chunk["_retrieval_score"],
            chunk["path"],
        ),
    )
    return ranked


def _document_score(
    doc: dict[str, Any],
    query: str,
    query_tokens: list[str],
    search_index: _SearchIndex,
    *,
    top_chunk_score: float,
) -> float:
    score = _bm25_score(query_tokens, search_index, doc["full_path"])
    score += _field_overlap_bonus(query_tokens, doc["title"], 4.0, search_index)
    score += _field_overlap_bonus(query_tokens, doc["original_source"], 6.0, search_index)
    score += _field_overlap_bonus(query_tokens, " ".join(doc.get("keywords", [])), 3.0, search_index)
    score += _field_overlap_bonus(query_tokens, " ".join(doc.get("heading_samples", [])), 3.0, search_index)
    score += _coverage_bonus(query_tokens, search_index.term_counts[doc["full_path"]], search_index) * 4.0
    score += top_chunk_score * 0.2

    if query.lower() in doc["title"].lower():
        score += 6.0
    return round(score, 6)


def _chunk_score(
    chunk: dict[str, Any],
    query: str,
    query_tokens: list[str],
    search_index: _SearchIndex,
) -> float:
    score = _bm25_score(query_tokens, search_index, chunk["path"])
    score += _field_overlap_bonus(query_tokens, chunk["document_title"], 2.0, search_index)
    score += _field_overlap_bonus(query_tokens, chunk["heading"], 6.0, search_index)
    score += _field_overlap_bonus(query_tokens, chunk["preview"], 4.0, search_index)
    score += _field_overlap_bonus(query_tokens, chunk["original_source"], 3.0, search_index)
    score += _coverage_bonus(query_tokens, search_index.term_counts[chunk["path"]], search_index) * 3.0
    if query.lower() in chunk["heading"].lower():
        score += 6.0
    return round(score, 6)


def _rank_of(values: list[str], expected: str) -> int | None:
    for index, value in enumerate(values, 1):
        if value == expected:
            return index
    return None


def _match_chunk_rank(ranked_chunks: list[dict[str, Any]], case: RetrievalCase) -> int | None:
    for index, chunk in enumerate(ranked_chunks, 1):
        if case.expected_chunk_path and chunk["path"] == case.expected_chunk_path:
            return index
        if case.expected_chunk_heading_contains:
            heading = _normalize_heading(chunk["heading"])
            if case.expected_chunk_heading_contains.lower() in heading.lower():
                return index
    return None


def _normalize_heading(heading: str) -> str:
    return heading.replace("\\.", ".")


def _normalize_retrieval_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 6:
        stem = token[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        if stem.endswith(("ach", "at", "iz", "ov")):
            return stem + "e"
        return stem
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith("es") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _query_terms(query: str) -> list[str]:
    return _tokenize_retrieval_text(query)


def _build_search_index(texts: dict[str, str]) -> _SearchIndex:
    term_counts: dict[str, Counter[str]] = {}
    document_frequency: Counter[str] = Counter()
    lengths: list[int] = []

    for key, text in texts.items():
        counts = Counter(_tokenize_retrieval_text(text))
        term_counts[key] = counts
        document_frequency.update(counts.keys())
        lengths.append(sum(counts.values()))

    corpus_size = max(1, len(texts))
    inverse_document_frequency = {
        term: math.log(1.0 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }
    average_length = statistics.mean(lengths) if lengths else 0.0
    return _SearchIndex(
        term_counts=term_counts,
        inverse_document_frequency=inverse_document_frequency,
        average_length=average_length,
    )


def _document_search_text(
    doc: dict[str, Any],
    *,
    output_dir: Path | None,
    file_cache: dict[str, str],
) -> str:
    parts = [
        doc["title"],
        doc["original_source"],
        " ".join(doc.get("keywords", [])),
        " ".join(doc.get("heading_samples", [])),
    ]
    if output_dir is not None:
        parts.append(_load_text(output_dir, doc["full_path"], file_cache))
    else:
        for chunk in doc.get("chunks", []):
            parts.append(chunk["heading"])
            parts.append(chunk["preview"])
    return "\n".join(part for part in parts if part)


def _chunk_search_text(
    chunk: dict[str, Any],
    *,
    output_dir: Path | None,
    file_cache: dict[str, str],
) -> str:
    parts = [
        chunk["document_title"],
        chunk["original_source"],
        chunk["heading"],
        chunk["preview"],
    ]
    if output_dir is not None:
        parts.append(_load_text(output_dir, chunk["path"], file_cache))
    return "\n".join(part for part in parts if part)


def _load_text(output_dir: Path, rel_path: str, file_cache: dict[str, str]) -> str:
    if rel_path not in file_cache:
        path = output_dir / rel_path
        file_cache[rel_path] = path.read_text(encoding="utf-8") if path.exists() else ""
    return file_cache[rel_path]


def _tokenize_retrieval_text(text: str) -> list[str]:
    tokens = [_normalize_retrieval_token(token) for token in tokenize(text)]
    normalized = unicodedata.normalize("NFKC", text).lower()

    for raw in _RETRIEVAL_TOKEN_RE.findall(normalized):
        if "-" not in raw and not any(char.isdigit() for char in raw):
            continue
        for fragment in _RETRIEVAL_FRAGMENT_RE.findall(raw):
            normalized_fragment = _normalize_retrieval_token(fragment)
            if _is_retrieval_fragment(normalized_fragment):
                tokens.append(normalized_fragment)

    return [token for token in tokens if token]


def _is_retrieval_fragment(token: str) -> bool:
    if not token or token in _RETRIEVAL_NOISE_TOKENS:
        return False
    if token.isdigit():
        return len(token) >= 2
    if token in _RETRIEVAL_SHORT_TOKENS:
        return True
    return bool(tokenize(token))


def _bm25_score(query_tokens: list[str], search_index: _SearchIndex, key: str) -> float:
    counts = search_index.term_counts.get(key, Counter())
    document_length = sum(counts.values())
    if not counts or not query_tokens:
        return 0.0

    average_length = search_index.average_length or 1.0
    score = 0.0
    for token in dict.fromkeys(query_tokens):
        frequency = counts.get(token, 0)
        if not frequency:
            continue
        inverse_document_frequency = search_index.inverse_document_frequency.get(token, 0.0)
        numerator = frequency * 2.2
        denominator = frequency + 1.2 * (1.0 - 0.75 + 0.75 * (document_length / average_length))
        score += inverse_document_frequency * (numerator / denominator)
    return score


def _field_overlap_bonus(
    query_tokens: list[str],
    text: str,
    weight: float,
    search_index: _SearchIndex,
) -> float:
    if not text:
        return 0.0
    field_terms = set(_tokenize_retrieval_text(text))
    return weight * sum(
        search_index.inverse_document_frequency.get(token, 0.0)
        for token in dict.fromkeys(query_tokens)
        if token in field_terms
    )


def _coverage_bonus(
    query_tokens: list[str],
    counts: Counter[str],
    search_index: _SearchIndex,
) -> float:
    if not query_tokens:
        return 0.0
    matched = {
        token
        for token in dict.fromkeys(query_tokens)
        if counts.get(token, 0) > 0
    }
    if not matched:
        return 0.0

    matched_weight = sum(search_index.inverse_document_frequency.get(token, 0.0) for token in matched)
    total_weight = sum(
        search_index.inverse_document_frequency.get(token, 0.0)
        for token in dict.fromkeys(query_tokens)
    )
    if total_weight == 0.0:
        return 0.0
    return matched_weight / total_weight


def _top_chunk_scores_by_document(ranked_chunks: list[dict[str, Any]]) -> dict[str, float]:
    top_scores: dict[str, float] = {}
    for chunk in ranked_chunks:
        current = top_scores.get(chunk["document_path"], 0.0)
        score = chunk.get("_retrieval_score", 0.0)
        if score > current:
            top_scores[chunk["document_path"]] = score
    return top_scores
