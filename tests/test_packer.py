"""Tests for agent context packer."""

from __future__ import annotations

import json
from pathlib import Path

from omnivorous.models import DocumentMetadata
from omnivorous.agents import AGENT_TARGETS
from omnivorous.packer import (
    generate_agent_instructions,
    generate_claude_md,
    generate_manifest,
    generate_project_context,
    pack_context,
    resolve_output_paths,
)


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
    assert data["version"] == "2.0"
    assert data["generator"] == "omnivorous"
    assert data["relationship_strategy"] == "deterministic_tfidf"
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
    related_titles = [item["title"] for item in payments["related_documents"]]
    assert "billing" in related_titles
    assert "hr" not in related_titles

    related_chunk = payments["chunks"][0]["related_chunks"][0]
    assert related_chunk["document_title"] == "billing"
    assert related_chunk["path"].startswith("docs/chunks/")
    assert related_chunk["shared_terms"]

    project_context = (out / "PROJECT_CONTEXT.md").read_text()
    assert "Cross-Document Bridges" in project_context


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
