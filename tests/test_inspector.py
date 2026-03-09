"""Tests for document inspector."""

from pathlib import Path

from agentmd.inspector import inspect_file


def test_inspect_markdown(fixtures_dir: Path):
    meta = inspect_file(fixtures_dir / "sample.md")
    assert meta.format == "markdown"
    assert meta.title == "Sample Document"
    assert len(meta.headings) >= 2
    assert meta.tokens_estimate > 0


def test_inspect_html(fixtures_dir: Path):
    meta = inspect_file(fixtures_dir / "sample.html")
    assert meta.format == "html"
    assert meta.tables == 1


def test_inspect_txt(fixtures_dir: Path):
    meta = inspect_file(fixtures_dir / "sample.txt")
    assert meta.format == "txt"
    assert meta.tokens_estimate > 0


def test_inspect_docx(sample_docx: Path):
    meta = inspect_file(sample_docx)
    assert meta.format == "docx"
    assert "Test Document" in meta.headings
