"""Tests for optional local embeddings."""

from pathlib import Path

from omnivorous.embeddings import EmbeddingNode, LocalEmbeddingService


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def embed(self, texts: list[str], *, model_name: str | None = None) -> list[list[float]]:
        del model_name
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
    def embed(self, texts: list[str], *, model_name: str | None = None) -> list[list[FakeFloat32]]:
        del model_name
        return [[FakeFloat32(1.0), FakeFloat32(0.0)] for _ in texts]


def test_local_embedding_service_caches_vectors(tmp_path: Path):
    backend = FakeBackend()
    service = LocalEmbeddingService(
        cache_dir=tmp_path,
        backend_name="fake",
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
        backend_name="fake",
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
        backend_name="fake",
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
        backend_name="fake",
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
