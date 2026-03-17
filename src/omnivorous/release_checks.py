"""Release-gate helpers for corpus validation and retrieval evaluation."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnivorous.relationships import tokenize


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
) -> list[RetrievalResult]:
    """Evaluate whether the manifest surfaces the expected doc/chunk near the top."""
    results: list[RetrievalResult] = []
    all_chunks = [
        {
            **chunk,
            "document_title": doc["title"],
            "original_source": doc["original_source"],
        }
        for doc in manifest["documents"]
        for chunk in doc["chunks"]
    ]

    for case in cases:
        ranked_documents = _rank_documents(manifest["documents"], case.query)
        ranked_chunks = _rank_chunks(all_chunks, case.query)

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


def _rank_documents(documents: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    query_tokens = set(tokenize(query))
    ranked = sorted(
        documents,
        key=lambda doc: (
            -_document_score(doc, query, query_tokens),
            doc["original_source"],
        ),
    )
    return ranked


def _rank_chunks(chunks: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    query_tokens = set(tokenize(query))
    ranked = sorted(
        chunks,
        key=lambda chunk: (
            -_chunk_score(chunk, query, query_tokens),
            chunk["path"],
        ),
    )
    return ranked


def _document_score(doc: dict[str, Any], query: str, query_tokens: set[str]) -> float:
    score = 0.0
    score += _weighted_overlap(query_tokens, doc["title"], 7)
    score += _weighted_overlap(query_tokens, doc["original_source"], 3)
    score += _weighted_overlap(query_tokens, " ".join(doc.get("keywords", [])), 5)
    score += _weighted_overlap(query_tokens, " ".join(doc.get("heading_samples", [])), 4)

    for chunk in doc.get("chunks", [])[:12]:
        score += _weighted_overlap(query_tokens, chunk["heading"], 2)
        score += _weighted_overlap(query_tokens, chunk["preview"], 1)

    if query.lower() in doc["title"].lower():
        score += 10
    return score


def _chunk_score(chunk: dict[str, Any], query: str, query_tokens: set[str]) -> float:
    score = 0.0
    score += _weighted_overlap(query_tokens, chunk["document_title"], 3)
    score += _weighted_overlap(query_tokens, chunk["heading"], 6)
    score += _weighted_overlap(query_tokens, chunk["preview"], 3)
    score += _weighted_overlap(query_tokens, chunk["original_source"], 2)
    if query.lower() in chunk["heading"].lower():
        score += 8
    return score


def _weighted_overlap(query_tokens: set[str], text: str, weight: int) -> float:
    if not text:
        return 0.0
    normalized_query = {_normalize_retrieval_token(token) for token in query_tokens}
    normalized_tokens = {_normalize_retrieval_token(token) for token in tokenize(text)}
    return float(weight * len(normalized_query & normalized_tokens))


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
        return token[:-3]
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith("es") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token
