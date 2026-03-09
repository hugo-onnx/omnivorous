"""Tests for HTML converter."""

from pathlib import Path

from agentmd.converters.html import HtmlConverter


def test_convert_html(fixtures_dir: Path):
    converter = HtmlConverter()
    result = converter.convert(fixtures_dir / "sample.html")

    assert result.metadata.format == "html"
    assert result.metadata.title == "Sample HTML Document"
    assert result.metadata.tables == 1
    assert result.metadata.tokens_estimate > 0
    # Should contain converted headings
    assert "Main Heading" in result.content


def test_html_name():
    assert HtmlConverter().name == "HTML"
