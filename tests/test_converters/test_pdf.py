"""Tests for PDF converter."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnivorous.converters.pdf import PdfConverter, get_pdf_engine, set_pdf_engine


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
    # Blank PDF should produce no headings in content
    assert result.metadata.pages == 1


# --- Engine selection tests ---


def test_default_engine_is_pymupdf():
    assert get_pdf_engine() == "pymupdf"


def test_set_engine_to_marker():
    original = get_pdf_engine()
    try:
        set_pdf_engine("marker")
        assert get_pdf_engine() == "marker"
    finally:
        set_pdf_engine(original)


def test_set_invalid_engine():
    with pytest.raises(ValueError, match="Unknown PDF engine"):
        set_pdf_engine("nonexistent")


# --- PyMuPDF engine tests (mocked) ---


def _mock_pymupdf4llm_convert(text: str, page_count: int = 1, images_per_page: int = 0):
    """Set up mocks for pymupdf4llm.to_markdown and pymupdf.open."""
    mock_to_md = MagicMock(return_value=text)

    mock_page = MagicMock()
    mock_page.get_text.return_value = text
    mock_page.get_images.return_value = [MagicMock()] * images_per_page

    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=page_count)
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page] * page_count))

    mock_open = MagicMock(return_value=mock_doc)

    return mock_to_md, mock_open


def test_pdf_ligature_normalization():
    mock_to_md, mock_open = _mock_pymupdf4llm_convert("\ufb01nance and e\ufb03ciency")
    with patch("omnivorous.converters.pdf._pymupdf.pymupdf4llm.to_markdown", mock_to_md), \
         patch("omnivorous.converters.pdf._pymupdf.pymupdf.open", mock_open):
        result = PdfConverter().convert(Path("test.pdf"))
    assert "finance" in result.content
    assert "efficiency" in result.content
    assert "\ufb01" not in result.content
    assert "\ufb03" not in result.content


def test_pdf_drop_cap_fix():
    mock_to_md, mock_open = _mock_pymupdf4llm_convert("N\nIST Framework")
    with patch("omnivorous.converters.pdf._pymupdf.pymupdf4llm.to_markdown", mock_to_md), \
         patch("omnivorous.converters.pdf._pymupdf.pymupdf.open", mock_open):
        result = PdfConverter().convert(Path("test.pdf"))
    assert "NIST Framework" in result.content


def test_pdf_repeated_lines_stripped():
    header = "ACME Corp Confidential"
    page_texts = [
        f"{header}\nFirst page content.",
        f"{header}\nSecond page content.",
        f"{header}\nThird page content.",
        f"{header}\nFourth page content.",
    ]
    combined = "\n\n".join(page_texts)

    mock_to_md = MagicMock(return_value=combined)

    # Each mock page returns text including the header
    mock_pages = []
    for text in page_texts:
        mp = MagicMock()
        mp.get_text.return_value = text
        mp.get_images.return_value = []
        mock_pages.append(mp)

    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=4)
    mock_doc.__iter__ = MagicMock(return_value=iter(mock_pages))
    mock_open = MagicMock(return_value=mock_doc)

    with patch("omnivorous.converters.pdf._pymupdf.pymupdf4llm.to_markdown", mock_to_md), \
         patch("omnivorous.converters.pdf._pymupdf.pymupdf.open", mock_open):
        result = PdfConverter().convert(Path("test.pdf"))
    assert "First page content." in result.content
    assert "Fourth page content." in result.content
    assert header not in result.content


def test_pdf_image_count():
    mock_to_md, mock_open = _mock_pymupdf4llm_convert(
        "Page with images.", page_count=1, images_per_page=3
    )
    with patch("omnivorous.converters.pdf._pymupdf.pymupdf4llm.to_markdown", mock_to_md), \
         patch("omnivorous.converters.pdf._pymupdf.pymupdf.open", mock_open):
        result = PdfConverter().convert(Path("test.pdf"))
    assert result.metadata.images == 3


def test_pdf_table_count():
    md_with_table = "Some text\n\n| Name | Value |\n| --- | --- |\n| foo | bar |\n\nMore text"
    mock_to_md, mock_open = _mock_pymupdf4llm_convert(md_with_table)
    with patch("omnivorous.converters.pdf._pymupdf.pymupdf4llm.to_markdown", mock_to_md), \
         patch("omnivorous.converters.pdf._pymupdf.pymupdf.open", mock_open):
        result = PdfConverter().convert(Path("test.pdf"))
    assert result.metadata.tables == 1


# --- Marker engine tests ---


def test_marker_not_installed_error():
    """Marker engine should raise ImportError with install instructions."""
    original = get_pdf_engine()
    try:
        set_pdf_engine("marker")
        with patch.dict("sys.modules", {"marker": None}):
            with pytest.raises(ImportError, match="omnivorous\\[scientific\\]"):
                from omnivorous.converters.pdf._marker import MarkerEngine
                MarkerEngine().extract(Path("test.pdf"))
    finally:
        set_pdf_engine(original)
