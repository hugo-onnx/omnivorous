"""Tests for DOCX converter."""

from pathlib import Path

from agentmd.converters.docx import DocxConverter


def test_convert_docx(sample_docx: Path):
    converter = DocxConverter()
    result = converter.convert(sample_docx)

    assert result.metadata.format == "docx"
    assert "Test Document" in result.metadata.headings
    assert "Section One" in result.metadata.headings
    assert result.metadata.tables == 1
    assert result.metadata.tokens_estimate > 0

    # Check markdown heading formatting
    assert "# Test Document" in result.content
    assert "## Section One" in result.content
    # Check table conversion
    assert "| Name | Value |" in result.content


def test_docx_name():
    assert DocxConverter().name == "DOCX"
