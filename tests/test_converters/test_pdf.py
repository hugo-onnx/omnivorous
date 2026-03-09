"""Tests for PDF converter."""

from pathlib import Path

from agentmd.converters.pdf import PdfConverter


def test_convert_blank_pdf(sample_pdf: Path):
    """A blank PDF should still produce metadata."""
    converter = PdfConverter()
    result = converter.convert(sample_pdf)

    assert result.metadata.format == "pdf"
    assert result.metadata.pages == 1
    assert result.metadata.title == "sample"


def test_pdf_name():
    assert PdfConverter().name == "PDF"
