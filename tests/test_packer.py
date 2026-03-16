"""Tests for agent context packer."""

from __future__ import annotations

import json
from pathlib import Path

from omnivorous.embeddings import LocalEmbeddingService
from omnivorous.models import DocumentMetadata
from omnivorous.agents import AGENT_TARGETS
from omnivorous.packer import (
    _filter_document_relationships,
    generate_agent_instructions,
    generate_claude_md,
    generate_manifest,
    generate_project_context,
    pack_context,
    resolve_output_paths,
)


class _FakeEmbeddingBackend:
    def embed(self, texts: list[str], *, model_name: str | None = None) -> list[list[float]]:
        del model_name
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered or "beta" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class _ModerateSemanticBackend:
    def embed(self, texts: list[str], *, model_name: str | None = None) -> list[list[float]]:
        del model_name
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "cache" in lowered or "http" in lowered:
                vectors.append([1.0, 0.0])
            elif "tax" in lowered or "return" in lowered:
                vectors.append([0.6, 0.8])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class _GenericAnchorSemanticBackend:
    def embed(self, texts: list[str], *, model_name: str | None = None) -> list[list[float]]:
        del model_name
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "protocol" in lowered:
                vectors.append([1.0, 0.0])
            elif "cybersecurity" in lowered:
                vectors.append([0.6, 0.8])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def _meta(
    title: str = "Test Doc",
    headings: list[str] | None = None,
) -> DocumentMetadata:
    return DocumentMetadata(
        source="test.md",
        format="markdown",
        title=title,
        tokens_estimate=100,
        headings=headings or [],
    )


def test_generate_claude_md():
    result = generate_claude_md([_meta("Doc A"), _meta("Doc B")])
    assert "# Project Context" in result
    assert "Doc A" in result
    assert "Doc B" in result


def test_generate_project_context():
    result = generate_project_context([_meta()])
    assert "Total documents: 1" in result
    assert "Total chunks: 0" in result
    assert "Test Doc" in result


def test_generate_manifest():
    result = generate_manifest([_meta()], ["docs/test.md"])
    data = json.loads(result)
    assert data["version"] == "3.0"
    assert data["generator"] == "omnivorous"
    assert data["relationship_strategy"] == "hybrid_reference_tfidf"
    assert len(data["documents"]) == 1
    assert data["chunk_strategy"] == "heading"
    assert data["chunk_size"] == 500
    assert "docs/test.md" in data["output_files"]


def test_pack_context(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-context"
    pack_context(fixtures_dir, out)

    assert (out / "CLAUDE.md").exists()
    assert (out / "PROJECT_CONTEXT.md").exists()
    assert (out / "manifest.json").exists()
    assert (out / "docs" / "full").is_dir()
    assert (out / "docs" / "chunks").is_dir()

    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["documents"]) > 0
    assert manifest["total_tokens"] > 0
    assert manifest["total_chunks"] > 0
    assert "manifest.json" in manifest["output_files"]


def test_pack_context_disambiguates_stem_collisions(tmp_path: Path):
    """Files with the same stem but different extensions get disambiguated output names."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "readme.md").write_text("# Hello")
    (source / "readme.txt").write_text("Hello")

    out = tmp_path / "out"
    pack_context(source, out)

    full_docs = out / "docs" / "full"
    chunk_docs = out / "docs" / "chunks"
    assert (full_docs / "readme_md.md").exists()
    assert (full_docs / "readme_txt.md").exists()
    assert not (full_docs / "readme.md").exists()
    assert any(chunk_docs.glob("readme_md_*.md"))
    assert any(chunk_docs.glob("readme_txt_*.md"))


def test_pack_context_preserves_subdirectory_structure(tmp_path: Path):
    """Files in different subdirs with the same name get separate output directories."""
    source = tmp_path / "source"
    (source / "ch1").mkdir(parents=True)
    (source / "ch2").mkdir(parents=True)
    (source / "ch1" / "intro.txt").write_text("Chapter 1 intro")
    (source / "ch2" / "intro.txt").write_text("Chapter 2 intro")

    out = tmp_path / "out"
    pack_context(source, out)

    full_docs = out / "docs" / "full"
    chunk_docs = out / "docs" / "chunks"
    assert (full_docs / "ch1" / "intro.md").exists()
    assert (full_docs / "ch2" / "intro.md").exists()
    assert any(chunk_docs.glob("ch1/intro_*.md"))
    assert any(chunk_docs.glob("ch2/intro_*.md"))


def test_generate_agent_instructions():
    agent = AGENT_TARGETS["codex"]
    result = generate_agent_instructions([_meta("My Doc")], agent)
    assert "# Project Context" in result
    assert "My Doc" in result
    assert "Codex CLI" in result
    assert "Prefer files under `docs/chunks/`" in result


def test_generate_claude_md_backward_compat():
    """generate_claude_md still works as a backward-compatible wrapper."""
    result = generate_claude_md([_meta("Doc A")])
    assert "# Project Context" in result
    assert "Doc A" in result
    assert "Claude Code" in result


def test_pack_context_codex_agent(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-context"
    pack_context(fixtures_dir, out, agents=["codex"])

    assert (out / "AGENTS.md").exists()
    assert not (out / "CLAUDE.md").exists()
    assert (out / "PROJECT_CONTEXT.md").exists()
    assert (out / "manifest.json").exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert "AGENTS.md" in manifest["output_files"]
    assert any(path.startswith("docs/chunks/") for path in manifest["output_files"])
    assert any(path.startswith("docs/full/") for path in manifest["output_files"])


def test_pack_context_cursor_agent(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-context"
    pack_context(fixtures_dir, out, agents=["cursor"])

    assert (out / ".cursor" / "rules" / "omnivorous.md").exists()
    assert not (out / "CLAUDE.md").exists()


def test_pack_context_antigravity_agent(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-context"
    pack_context(fixtures_dir, out, agents=["antigravity"])

    assert (out / ".agent" / "skills" / "omnivorous.md").exists()


def test_pack_context_multiple_agents(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-context"
    pack_context(fixtures_dir, out, agents=["claude", "codex"])

    assert (out / "CLAUDE.md").exists()
    assert (out / "AGENTS.md").exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert "CLAUDE.md" in manifest["output_files"]
    assert "AGENTS.md" in manifest["output_files"]


def test_pack_context_all_agents(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-context"
    pack_context(fixtures_dir, out, agents=["all"])

    assert (out / "CLAUDE.md").exists()
    assert (out / "AGENTS.md").exists()
    assert (out / ".cursor" / "rules" / "omnivorous.md").exists()
    assert (out / ".github" / "copilot-instructions.md").exists()
    assert (out / ".agent" / "skills" / "omnivorous.md").exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["output_files"]) >= len(AGENT_TARGETS)


def test_pack_context_manifest_includes_chunk_navigation(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-context"
    pack_context(fixtures_dir, out, chunk_size=40, chunk_by="tokens")

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["chunk_strategy"] == "tokens"
    assert manifest["chunk_size"] == 40
    first_document = manifest["documents"][0]
    assert first_document["full_path"].startswith("docs/full/")
    assert first_document["chunk_count"] >= 1
    assert first_document["chunks"][0]["path"].startswith("docs/chunks/")
    assert "preview" in first_document["chunks"][0]
    assert "related_documents" in first_document


def test_pack_context_builds_cross_document_relationships(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "payments.md").write_text(
        "# Payments API\n\n"
        "## Refunds\n\n"
        "Refund invoice payments when retries fail.\n\n"
        "## Retries\n\n"
        "Retry failed invoice charges with backoff.\n"
    )
    (source / "billing.txt").write_text(
        "Billing runbook for invoice refunds and payment retries after failed charges."
    )
    (source / "hr.txt").write_text("Employee onboarding handbook and vacation policy.")

    out = tmp_path / "agent-context"
    pack_context(source, out, chunk_size=20, chunk_by="tokens")

    manifest = json.loads((out / "manifest.json").read_text())
    payments = next(doc for doc in manifest["documents"] if doc["title"] == "Payments API")
    related_titles = [item["target_title"] for item in payments["related_documents"]]
    assert "billing" in related_titles
    assert "hr" not in related_titles

    related_chunk = payments["chunks"][0]["related_chunks"][0]
    assert related_chunk["target_title"] == "billing"
    assert related_chunk["target_path"].startswith("docs/chunks/")
    assert related_chunk["evidence"]["shared_terms"]

    project_context = (out / "PROJECT_CONTEXT.md").read_text()
    assert "Cross-Document Bridges" in project_context


def test_pack_context_records_explicit_reference_evidence(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "overview.md").write_text(
        "# Overview\n\nSee `billing.txt` and RFC-101 for refund handling.\n"
    )
    (source / "billing.txt").write_text("RFC-101 billing runbook for refund handling.")

    out = tmp_path / "agent-context"
    pack_context(source, out, chunk_size=20, chunk_by="tokens")

    manifest = json.loads((out / "manifest.json").read_text())
    overview = next(doc for doc in manifest["documents"] if doc["title"] == "Overview")
    billing_edge = next(
        edge for edge in overview["related_documents"] if edge["target_title"] == "billing"
    )
    assert billing_edge["relationship_type"] in {"explicit_reference", "hybrid"}
    assert billing_edge["signal_scores"]["reference_match"] >= 0.95
    assert {"type": "path", "value": "billing.txt"} in billing_edge["evidence"]["references"]


def test_pack_context_ignores_low_confidence_reference_noise(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "story.txt").write_text(
        "CHAPTER IV\n\n"
        "See section 4, THE, and III before you continue.\n"
    )
    (source / "api.md").write_text("# API Guide\n\n## 4 Billing\n\n`PaymentClient` handles retries.\n")

    out = tmp_path / "agent-context"
    pack_context(source, out, chunk_size=20, chunk_by="tokens")

    manifest = json.loads((out / "manifest.json").read_text())
    story = next(doc for doc in manifest["documents"] if doc["title"] == "story")
    assert story["related_documents"] == []


def test_pack_context_excludes_low_signal_boilerplate_chunks_from_relationships(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "story-one.md").write_text(
        "# Story One\n\n"
        "## Plot\n\n"
        "Tea party, rabbit hole, and curious adventures.\n\n"
        "## License\n\n"
        "Project Gutenberg ebook license copyright foundation donations trademark archive.\n"
    )
    (source / "story-two.md").write_text(
        "# Story Two\n\n"
        "## Plot\n\n"
        "Battle plans, scouts, and strategic retreats.\n\n"
        "## License\n\n"
        "Project Gutenberg ebook license copyright foundation donations trademark archive.\n"
    )

    out = tmp_path / "agent-context"
    pack_context(source, out, chunk_size=20, chunk_by="heading")

    manifest = json.loads((out / "manifest.json").read_text())
    story_one = next(doc for doc in manifest["documents"] if doc["title"] == "Story One")
    license_chunk = next(chunk for chunk in story_one["chunks"] if chunk["heading"] == "License")
    assert license_chunk["related_chunks"] == []


def test_pack_context_can_add_semantic_relationships(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "alpha.md").write_text("# Alpha Guide\n\nWidget recovery procedures.")
    (source / "beta.txt").write_text("Subsystem restoration notes.")
    (source / "hr.txt").write_text("Vacation handbook and onboarding.")

    embedding_service = LocalEmbeddingService(
        cache_dir=tmp_path / "embeddings-cache",
        backend_name="fake",
        backend=_FakeEmbeddingBackend(),
    )

    out = tmp_path / "agent-context"
    pack_context(
        source,
        out,
        chunk_size=20,
        chunk_by="tokens",
        enable_semantic=True,
        embedding_service=embedding_service,
    )

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["relationship_strategy"] == "hybrid_reference_tfidf_embedding"
    alpha = next(doc for doc in manifest["documents"] if doc["title"] == "Alpha Guide")
    beta_edge = next(
        edge for edge in alpha["related_documents"] if edge["target_title"] == "beta"
    )
    assert beta_edge["relationship_type"] in {"semantic_similarity", "hybrid"}
    assert beta_edge["signal_scores"]["semantic_similarity"] >= 0.99
    assert not (out / ".omnivorous-cache").exists()


def test_pack_context_uses_external_default_embedding_cache(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "alpha.md").write_text("# Alpha\n\nWidget recovery procedures.")
    (source / "beta.md").write_text("# Beta\n\nSubsystem restoration notes.")

    recorded: dict[str, Path] = {}

    class RecordingEmbeddingService:
        def __init__(
            self,
            *,
            cache_dir: Path,
            backend_name: str = "fastembed",
            model_name: str | None = None,
            backend=None,
        ):
            del backend_name, model_name, backend
            recorded["cache_dir"] = cache_dir

        def build_relationships(self, nodes, **kwargs):
            del nodes, kwargs
            return {}

    monkeypatch.setattr("omnivorous.packer.LocalEmbeddingService", RecordingEmbeddingService)

    out = tmp_path / "agent-context"
    pack_context(source, out, chunk_size=20, chunk_by="tokens", enable_semantic=True)

    assert recorded["cache_dir"] != out / ".omnivorous-cache"
    assert out not in recorded["cache_dir"].parents
    assert not (out / ".omnivorous-cache").exists()


def test_pack_context_filters_moderate_semantic_false_positives(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "http-caching.md").write_text(
        "# HTTP caching\n\nCache-Control and ETag validation rules.\n"
    )
    (source / "tax-form.txt").write_text(
        "Tax return filing instructions and withholding details.\n"
    )

    embedding_service = LocalEmbeddingService(
        cache_dir=tmp_path / "embeddings-cache",
        backend_name="moderate",
        backend=_ModerateSemanticBackend(),
    )

    out = tmp_path / "agent-context"
    pack_context(
        source,
        out,
        chunk_size=20,
        chunk_by="tokens",
        enable_semantic=True,
        embedding_service=embedding_service,
    )

    manifest = json.loads((out / "manifest.json").read_text())
    caching = next(doc for doc in manifest["documents"] if doc["title"] == "HTTP caching")
    assert caching["related_documents"] == []


def test_pack_context_ignores_generic_anchor_terms_for_semantic_edges(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "protocol.md").write_text(
        "# Protocol Handbook\n\n## Abstract\n\nHTTP message framing details.\n"
    )
    (source / "cybersecurity.md").write_text(
        "# Cybersecurity Playbook\n\n## Abstract\n\nRisk assessment guidance.\n"
    )

    embedding_service = LocalEmbeddingService(
        cache_dir=tmp_path / "embeddings-cache",
        backend_name="generic-anchor",
        backend=_GenericAnchorSemanticBackend(),
    )

    out = tmp_path / "agent-context"
    pack_context(
        source,
        out,
        chunk_size=20,
        chunk_by="tokens",
        enable_semantic=True,
        embedding_service=embedding_service,
    )

    manifest = json.loads((out / "manifest.json").read_text())
    protocol = next(doc for doc in manifest["documents"] if doc["title"] == "Protocol Handbook")
    assert protocol["related_documents"] == []


def test_filter_document_relationships_requires_multiple_anchor_terms():
    documents = [
        {
            "full_path": "docs/full/source.md",
            "keywords": ["client", "sdk", "telemetry"],
        },
        {
            "full_path": "docs/full/target.md",
            "keywords": ["client", "http", "protocol"],
        },
    ]
    edges = {
        "docs/full/source.md": [
            {
                "target_path": "docs/full/target.md",
                "target_kind": "document",
                "target_title": "target",
                "target_heading": None,
                "relationship_type": "semantic_similarity",
                "score": 0.66,
                "signal_scores": {"semantic_similarity": 0.66},
                "evidence": {"shared_terms": [], "references": []},
            }
        ],
        "docs/full/target.md": [],
    }

    filtered = _filter_document_relationships(documents, edges)

    assert filtered["docs/full/source.md"] == []


# --- resolve_output_paths unit tests ---


def test_resolve_output_paths_no_conflicts(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    files = [source / "a.txt", source / "b.md"]
    for f in files:
        f.touch()

    result = resolve_output_paths(files, source)
    assert result[files[0]] == Path("a.md")
    assert result[files[1]] == Path("b.md")


def test_resolve_output_paths_same_stem_different_ext(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    f1 = source / "readme.md"
    f2 = source / "readme.txt"
    f1.touch()
    f2.touch()

    result = resolve_output_paths([f1, f2], source)
    assert result[f1] == Path("readme_md.md")
    assert result[f2] == Path("readme_txt.md")


def test_resolve_output_paths_subdirectories(tmp_path: Path):
    source = tmp_path / "src"
    (source / "ch1").mkdir(parents=True)
    (source / "ch2").mkdir(parents=True)
    f1 = source / "ch1" / "intro.txt"
    f2 = source / "ch2" / "intro.txt"
    f1.touch()
    f2.touch()

    result = resolve_output_paths([f1, f2], source)
    assert result[f1] == Path("ch1/intro.md")
    assert result[f2] == Path("ch2/intro.md")


def test_resolve_output_paths_mixed(tmp_path: Path):
    """Subdirectory separation prevents collision, same-dir collision is disambiguated."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "sub").mkdir()
    f1 = source / "notes.md"
    f2 = source / "notes.txt"
    f3 = source / "sub" / "notes.txt"
    for f in [f1, f2, f3]:
        f.touch()

    result = resolve_output_paths([f1, f2, f3], source)
    # f1 and f2 collide at root → disambiguated
    assert result[f1] == Path("notes_md.md")
    assert result[f2] == Path("notes_txt.md")
    # f3 is in a subdir → no collision
    assert result[f3] == Path("sub/notes.md")
