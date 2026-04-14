"""Tests for always-on local embeddings."""

import hashlib
import sys
from pathlib import Path

import omnivorous.embeddings as embeddings_module
from omnivorous.embeddings import (
    EmbeddingNode,
    FIXED_EMBEDDING_MODEL,
    LocalEmbeddingService,
    default_embedding_cache_dir,
    default_embedding_model_cache_dir,
    default_embedding_root_dir,
)


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered or "beta" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class FakeFloat32:
    def __init__(self, value: float):
        self.value = value

    def __float__(self) -> float:
        return self.value


class FakeNumpyLikeBackend:
    def embed(self, texts: list[str]) -> list[list[FakeFloat32]]:
        return [[FakeFloat32(1.0), FakeFloat32(0.0)] for _ in texts]


def test_local_embedding_service_caches_vectors(tmp_path: Path):
    backend = FakeBackend()
    service = LocalEmbeddingService(
        cache_dir=tmp_path,
        backend=backend,
    )
    nodes = [
        EmbeddingNode(key="a", text="Alpha document"),
        EmbeddingNode(key="b", text="Beta document"),
    ]

    first = service.build_relationships(nodes, min_score=0.1)
    second = service.build_relationships(nodes, min_score=0.1)

    assert backend.calls == 1
    assert first["a"][0].target_key == "b"
    assert second["a"][0].target_key == "b"


def test_local_embedding_service_respects_groups(tmp_path: Path):
    backend = FakeBackend()
    service = LocalEmbeddingService(
        cache_dir=tmp_path,
        backend=backend,
    )
    nodes = [
        EmbeddingNode(key="a1", text="Alpha one", group="a"),
        EmbeddingNode(key="a2", text="Alpha two", group="a"),
        EmbeddingNode(key="b1", text="Beta one", group="b"),
    ]

    relationships = service.build_relationships(
        nodes,
        min_score=0.1,
        require_different_group=True,
    )

    assert relationships["a1"][0].target_key == "b1"


def test_local_embedding_service_respects_candidate_groups(tmp_path: Path):
    backend = FakeBackend()
    service = LocalEmbeddingService(
        cache_dir=tmp_path,
        backend=backend,
    )
    nodes = [
        EmbeddingNode(key="a1", text="Alpha one", group="a"),
        EmbeddingNode(key="b1", text="Beta one", group="b"),
        EmbeddingNode(key="c1", text="Beta two", group="c"),
    ]

    relationships = service.build_relationships(
        nodes,
        min_score=0.1,
        require_different_group=True,
        candidate_groups={"a": {"b"}, "b": {"a"}},
    )

    assert relationships["a1"][0].target_key == "b1"
    assert relationships["c1"] == []
    assert all(match.target_key != "c1" for match in relationships["a1"])


def test_local_embedding_service_normalizes_json_unsafe_scalars(tmp_path: Path):
    service = LocalEmbeddingService(
        cache_dir=tmp_path,
        backend=FakeNumpyLikeBackend(),
    )
    nodes = [
        EmbeddingNode(key="a", text="Alpha document"),
        EmbeddingNode(key="b", text="Alpha sibling"),
    ]

    relationships = service.build_relationships(nodes, min_score=0.1)

    assert relationships["a"][0].target_key == "b"
    cache_files = list(tmp_path.glob("*.json"))
    assert cache_files


def test_default_embedding_cache_dirs_use_platform_root(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setattr(embeddings_module, "default_embedding_root_dir", default_embedding_root_dir)

    if sys.platform == "darwin":
        expected_root = Path.home() / "Library" / "Caches" / "omnivorous"
    else:
        expected_root = tmp_path / "xdg-cache" / "omnivorous"

    assert default_embedding_root_dir() == expected_root
    assert default_embedding_cache_dir() == expected_root / "vectors"
    assert default_embedding_model_cache_dir() == expected_root / "models"


def test_local_embedding_service_cache_keys_include_fixed_model(tmp_path: Path):
    service = LocalEmbeddingService(cache_dir=tmp_path, backend=FakeBackend())

    path = service._cache_path("Alpha document")
    expected = hashlib.sha256(f"{FIXED_EMBEDDING_MODEL}:Alpha document".encode("utf-8")).hexdigest()

    assert path == tmp_path / f"{expected}.json"
