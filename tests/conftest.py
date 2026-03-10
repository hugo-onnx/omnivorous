"""Shared test fixtures."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal PDF fixture using pypdf."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    # Add text content via page annotations isn't ideal —
    # use a pre-built PDF fixture instead
    out = tmp_path / "sample.pdf"
    with open(out, "wb") as f:
        writer.write(f)
    return out


@pytest.fixture
def sample_pdf_with_text(fixtures_dir: Path) -> Path:
    """Return path to the pre-built PDF fixture if it exists, else skip."""
    p = fixtures_dir / "document.pdf"
    if not p.exists():
        pytest.skip("document.pdf fixture not available")
    return p


@pytest.fixture
def sample_docx(tmp_path: Path) -> Path:
    """Create a DOCX fixture with headings, paragraphs, and a table."""
    from docx import Document

    doc = Document()
    doc.add_heading("Test Document", level=1)
    doc.add_paragraph("This is the first paragraph.")
    doc.add_heading("Section One", level=2)
    doc.add_paragraph("Content of section one.")

    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "foo"
    table.cell(1, 1).text = "bar"

    out = tmp_path / "sample.docx"
    doc.save(str(out))
    return out
