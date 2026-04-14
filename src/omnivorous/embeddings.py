"""Always-on local embeddings with persistent on-disk caches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

FIXED_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


class EmbeddingBackend(Protocol):
    """Backend protocol for local embedding providers."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedBackend:
    """Lazy wrapper around fastembed for the fixed local embedding model."""

    def __init__(
        self,
        *,
        cache_dir: Path,
    ):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised via integration path
            raise ImportError(
                "Semantic relationships are unavailable because the omnivorous installation is incomplete. "
                "Reinstall omnivorous and try again."
            ) from exc

        try:
            self._embedder = TextEmbedding(
                model_name=FIXED_EMBEDDING_MODEL,
                cache_dir=str(cache_dir),
            )
        except ValueError as exc:
            message = str(exc)
            if "Could not load model" not in message:
                raise
            raise ImportError(
                "Omnivorous could not load its local semantic model. "
                "The first successful `omni <folder>` run needs network access to download the model once; "
                "later runs reuse the local cache."
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
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
        cache_dir: Path | None = None,
        model_cache_dir: Path | None = None,
        backend: EmbeddingBackend | None = None,
    ):
        resolved_cache_dir = cache_dir or default_embedding_cache_dir()
        resolved_model_cache_dir = model_cache_dir or default_embedding_model_cache_dir()
        self.cache_dir = _ensure_cache_dir(resolved_cache_dir, "vector cache")
        self.model_cache_dir = _ensure_cache_dir(resolved_model_cache_dir, "model cache")
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
            generated = backend.embed(missing_texts)
            for key, text, vector in zip(missing_keys, missing_texts, generated, strict=True):
                normalized = [float(value) for value in vector]
                cache_path = self._cache_path(text)
                cache_path.write_text(json.dumps(normalized), encoding="utf-8")
                vectors[key] = normalized

        return vectors

    def _resolve_backend(self) -> EmbeddingBackend:
        if self._backend is not None:
            return self._backend
        self._backend = FastEmbedBackend(
            cache_dir=self.model_cache_dir,
        )
        return self._backend

    def _cache_path(self, text: str) -> Path:
        payload = f"{FIXED_EMBEDDING_MODEL}:{text}".encode("utf-8")
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
    return default_embedding_root_dir() / "vectors"


def default_embedding_model_cache_dir() -> Path:
    """Return the cache directory used for local embedding model files."""
    return default_embedding_root_dir() / "models"


def default_embedding_root_dir() -> Path:
    """Return the persistent user-level cache root for semantic artifacts."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "omnivorous"

    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "omnivorous"


def _ensure_cache_dir(path: Path, purpose: str) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as exc:
        raise ValueError(
            f"Could not create the semantic {purpose} at {path}. "
            "Omnivorous needs a writable persistent cache directory for local semantic relationships."
        ) from exc
