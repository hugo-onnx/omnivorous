"""Tests for plain text converter."""

from pathlib import Path

from agentmd.converters.txt import TxtConverter


def test_convert_txt(fixtures_dir: Path):
    converter = TxtConverter()
    result = converter.convert(fixtures_dir / "sample.txt")

    assert result.content.startswith("# sample")
    assert "plain text document" in result.content
    assert result.metadata.format == "txt"
    assert result.metadata.title == "sample"
    assert result.metadata.tokens_estimate > 0


def test_txt_name():
    assert TxtConverter().name == "Text"
