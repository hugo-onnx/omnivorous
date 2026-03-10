"""Tests for agent context packer."""

import json
from pathlib import Path

import pytest

from agentmd.models import DocumentMetadata
from agentmd.packer import (
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
    assert data["generator"] == "agentmd"
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
