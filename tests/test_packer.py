"""Tests for agent context packer."""

import json
from pathlib import Path

import pytest

from omnivorous.models import DocumentMetadata
from omnivorous.agents import AGENT_TARGETS
from omnivorous.packer import (
    generate_agent_instructions,
    generate_claude_md,
    generate_manifest,
    generate_project_context,
    pack_context,
)


def _meta(title: str = "Test Doc") -> DocumentMetadata:
    return DocumentMetadata(
        source="test.md", format="markdown", title=title, tokens_estimate=100
    )


def test_generate_claude_md():
    result = generate_claude_md([_meta("Doc A"), _meta("Doc B")])
    assert "# Project Context" in result
    assert "Doc A" in result
    assert "Doc B" in result


def test_generate_project_context():
    result = generate_project_context([_meta()])
    assert "Total documents: 1" in result
    assert "Test Doc" in result


def test_generate_manifest():
    result = generate_manifest([_meta()], ["docs/test.md"])
    data = json.loads(result)
    assert data["version"] == "1.0"
    assert data["generator"] == "omnivorous"
    assert len(data["documents"]) == 1
    assert "docs/test.md" in data["output_files"]


def test_pack_context(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-context"
    pack_context(fixtures_dir, out)

    assert (out / "CLAUDE.md").exists()
    assert (out / "PROJECT_CONTEXT.md").exists()
    assert (out / "manifest.json").exists()
    assert (out / "docs").is_dir()

    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["documents"]) > 0
    assert manifest["total_tokens"] > 0
    assert "manifest.json" in manifest["output_files"]


def test_pack_context_rejects_stem_collisions(tmp_path: Path):
    """Files with the same stem but different extensions should be rejected."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "readme.md").write_text("# Hello")
    (source / "readme.txt").write_text("Hello")

    out = tmp_path / "out"
    with pytest.raises(ValueError, match="same name"):
        pack_context(source, out)


def test_generate_agent_instructions():
    agent = AGENT_TARGETS["codex"]
    result = generate_agent_instructions([_meta("My Doc")], agent)
    assert "# Project Context" in result
    assert "My Doc" in result
    assert "Codex CLI" in result


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
