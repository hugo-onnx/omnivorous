"""Optional local embeddings with a pluggable backend and on-disk cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class EmbeddingBackend(Protocol):
    """Backend protocol for local embedding providers."""

    def embed(self, texts: list[str], *, model_name: str | None = None) -> list[list[float]]: ...


class FastEmbedBackend:
    """Lazy wrapper around fastembed so semantic mode stays optional."""

    def __init__(self, model_name: str | None = None):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised via integration path
            raise ImportError(
                "Semantic mode requires local embedding support. "
                "Use `uv sync --extra semantic` or run once with "
                "`uv run --extra semantic omni pack ...`. "
                "With pip, install `omnivorous[semantic]`."
            ) from exc

        self._embedder = TextEmbedding(model_name=model_name) if model_name else TextEmbedding()

    def embed(self, texts: list[str], *, model_name: str | None = None) -> list[list[float]]:
        del model_name
        return [[float(value) for value in vector] for vector in self._embedder.embed(texts)]


@dataclass(frozen=True)
class EmbeddingNode:
    """Item to embed and compare against other items."""

    key: str
    text: str
    group: str | None = None


@dataclass(frozen=True)
class EmbeddingMatch:
    """Embedding-based similarity edge."""

    target_key: str
    score: float


class LocalEmbeddingService:
    """Compute and cache local embeddings, then derive similarity relationships."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        backend_name: str = "fastembed",
        model_name: str | None = None,
        backend: EmbeddingBackend | None = None,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend_name = backend_name
        self.model_name = model_name
        self._backend = backend

    def build_relationships(
        self,
        nodes: list[EmbeddingNode],
        *,
        limit: int = 3,
        min_score: float = 0.45,
        require_different_group: bool = False,
        candidate_groups: dict[str, set[str]] | None = None,
    ) -> dict[str, list[EmbeddingMatch]]:
        if not nodes:
            return {}

        vectors = self._embed_nodes(nodes)
        group_members: dict[str | None, list[int]] = {}
        if candidate_groups is not None:
            for index, node in enumerate(nodes):
                group_members.setdefault(node.group, []).append(index)
        relationships: dict[str, list[EmbeddingMatch]] = {}
        for index, node in enumerate(nodes):
            ranked: list[EmbeddingMatch] = []
            if candidate_groups is None:
                candidate_indexes = range(len(nodes))
            else:
                candidate_indexes_set: set[int] = set()
                for group in candidate_groups.get(node.group or "", set()):
                    candidate_indexes_set.update(group_members.get(group, []))
                candidate_indexes = sorted(candidate_indexes_set)

            for other_index in candidate_indexes:
                if index == other_index:
                    continue
                other = nodes[other_index]
                if require_different_group and node.group == other.group:
                    continue

                score = _cosine_similarity(vectors[node.key], vectors[other.key])
                if score < min_score:
                    continue
                ranked.append(EmbeddingMatch(target_key=other.key, score=round(score, 3)))

            ranked.sort(key=lambda match: (-match.score, match.target_key))
            relationships[node.key] = ranked[:limit]

        return relationships

    def _embed_nodes(self, nodes: list[EmbeddingNode]) -> dict[str, list[float]]:
        vectors: dict[str, list[float]] = {}
        missing_texts: list[str] = []
        missing_keys: list[str] = []

        for node in nodes:
            cache_path = self._cache_path(node.text)
            if cache_path.exists():
                vectors[node.key] = json.loads(cache_path.read_text(encoding="utf-8"))
                continue
            missing_texts.append(node.text)
            missing_keys.append(node.key)

        if missing_texts:
            backend = self._resolve_backend()
            generated = backend.embed(missing_texts, model_name=self.model_name)
            for key, text, vector in zip(missing_keys, missing_texts, generated, strict=True):
                normalized = [float(value) for value in vector]
                cache_path = self._cache_path(text)
                cache_path.write_text(json.dumps(normalized), encoding="utf-8")
                vectors[key] = normalized

        return vectors

    def _resolve_backend(self) -> EmbeddingBackend:
        if self._backend is not None:
            return self._backend
        if self.backend_name != "fastembed":
            raise ValueError(
                f"Unsupported embedding backend: {self.backend_name!r}. Valid: fastembed"
            )
        self._backend = FastEmbedBackend(model_name=self.model_name)
        return self._backend

    def _cache_path(self, text: str) -> Path:
        payload = f"{self.backend_name}:{self.model_name or 'default'}:{text}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return self.cache_dir / f"{digest}.json"


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_norm * right_norm)


def default_embedding_cache_dir() -> Path:
    """Return a user-level cache directory for local embeddings."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "omnivorous"

    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "omnivorous"
