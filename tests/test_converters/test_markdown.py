"""Tests for markdown converter."""

from pathlib import Path

from agentmd.converters.markdown import MarkdownConverter


def test_convert_markdown(fixtures_dir: Path):
    converter = MarkdownConverter()
    result = converter.convert(fixtures_dir / "readme.md")

    assert "# Sample Document" in result.content
    assert result.metadata.format == "markdown"
    assert result.metadata.title == "Sample Document"
    assert len(result.metadata.headings) == 4  # Sample Document, Section One, Section Two, Subsection
    assert result.metadata.tables == 1
    assert result.metadata.tokens_estimate > 0


def test_markdown_name():
    assert MarkdownConverter().name == "Markdown"
