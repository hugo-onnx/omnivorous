"""Tests for PDF converter."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from omnivorous.converters.pdf import PdfConverter


def test_convert_blank_pdf(sample_pdf: Path):
    """A blank PDF should still produce metadata."""
    converter = PdfConverter()
    result = converter.convert(sample_pdf)

    assert result.metadata.format == "pdf"
    assert result.metadata.source == "sample.pdf"
    assert result.metadata.pages == 1
    assert result.metadata.title == "sample"


def test_pdf_name():
    assert PdfConverter().name == "PDF"


def test_pdf_no_synthetic_headings(sample_pdf: Path):
    result = PdfConverter().convert(sample_pdf)
    assert result.metadata.headings == []


def _mock_reader(page_texts: list[str]):
    """Create a mock PdfReader with given page texts."""
    reader = MagicMock()
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    reader.pages = pages
    return reader


def test_pdf_ligature_normalization():
    with patch("omnivorous.converters.pdf.PdfReader") as mock_cls:
        mock_cls.return_value = _mock_reader(["\ufb01nance and e\ufb03ciency"])
        result = PdfConverter().convert(Path("test.pdf"))
    assert "finance" in result.content
    assert "efficiency" in result.content
    assert "\ufb01" not in result.content
    assert "\ufb03" not in result.content


def test_pdf_drop_cap_fix():
    with patch("omnivorous.converters.pdf.PdfReader") as mock_cls:
        mock_cls.return_value = _mock_reader(["N\nIST Framework"])
        result = PdfConverter().convert(Path("test.pdf"))
    assert "NIST Framework" in result.content


def test_pdf_repeated_lines_stripped():
    header = "ACME Corp Confidential"
    pages = [
        f"{header}\nFirst page content.",
        f"{header}\nSecond page content.",
        f"{header}\nThird page content.",
        f"{header}\nFourth page content.",
    ]
    with patch("omnivorous.converters.pdf.PdfReader") as mock_cls:
        mock_cls.return_value = _mock_reader(pages)
        result = PdfConverter().convert(Path("test.pdf"))
    assert "First page content." in result.content
    assert "Fourth page content." in result.content
    assert header not in result.content
