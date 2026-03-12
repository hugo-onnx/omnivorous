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


def _mock_page(text: str, tables=None, images=None):
    """Create a mock pdfplumber page."""
    page = MagicMock()
    page.extract_text.return_value = text
    page.find_tables.return_value = tables or []
    page.images = images or []
    page.height = 792
    page.width = 612
    return page


def _mock_pdf(page_texts: list[str], page_tables=None, page_images=None):
    """Create a mock pdfplumber PDF context manager."""
    pages = []
    for i, text in enumerate(page_texts):
        tables = page_tables[i] if page_tables else None
        images = page_images[i] if page_images else None
        pages.append(_mock_page(text, tables, images))

    pdf = MagicMock()
    pdf.pages = pages
    pdf.__enter__ = MagicMock(return_value=pdf)
    pdf.__exit__ = MagicMock(return_value=False)
    return pdf


def test_pdf_ligature_normalization():
    with patch("omnivorous.converters.pdf.pdfplumber") as mock_plumber:
        mock_plumber.open.return_value = _mock_pdf(["\ufb01nance and e\ufb03ciency"])
        result = PdfConverter().convert(Path("test.pdf"))
    assert "finance" in result.content
    assert "efficiency" in result.content
    assert "\ufb01" not in result.content
    assert "\ufb03" not in result.content


def test_pdf_drop_cap_fix():
    with patch("omnivorous.converters.pdf.pdfplumber") as mock_plumber:
        mock_plumber.open.return_value = _mock_pdf(["N\nIST Framework"])
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
    with patch("omnivorous.converters.pdf.pdfplumber") as mock_plumber:
        mock_plumber.open.return_value = _mock_pdf(pages)
        result = PdfConverter().convert(Path("test.pdf"))
    assert "First page content." in result.content
    assert "Fourth page content." in result.content
    assert header not in result.content


def test_pdf_table_extraction():
    """Tables should be converted to markdown format."""
    table_data = [["Name", "Value"], ["foo", "bar"]]
    mock_table = MagicMock()
    mock_table.bbox = (0, 100, 612, 200)
    mock_table.extract.return_value = table_data

    page = _mock_page("Some text\nwith table below", tables=[mock_table])
    # When cropping for text above the table, return the text
    cropped_above = MagicMock()
    cropped_above.extract_text.return_value = "Some text\nwith table below"
    cropped_below = MagicMock()
    cropped_below.extract_text.return_value = ""
    page.crop.side_effect = [cropped_above, cropped_below]

    with patch("omnivorous.converters.pdf.pdfplumber") as mock_plumber:
        mock_plumber.open.return_value = _mock_pdf(["Some text\nwith table below"])
        # Override pages with our custom page
        mock_plumber.open.return_value.pages = [page]
        result = PdfConverter().convert(Path("test.pdf"))

    assert "| Name | Value |" in result.content
    assert "| foo | bar |" in result.content
    assert result.metadata.tables == 1


def test_pdf_image_count():
    """Images should be counted in metadata."""
    images = [{"x0": 0, "y0": 0, "x1": 100, "y1": 100}]
    with patch("omnivorous.converters.pdf.pdfplumber") as mock_plumber:
        mock_plumber.open.return_value = _mock_pdf(
            ["Page with images."],
            page_images=[images],
        )
        result = PdfConverter().convert(Path("test.pdf"))
    assert result.metadata.images == 1
