"""Shared test fixtures."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

import omnivorous.embeddings as embeddings_module

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


class _DefaultFakeEmbeddingBackend:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            buckets = [0.0] * 16
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                buckets[hash(token) % len(buckets)] += 1.0
            norm = math.sqrt(sum(value * value for value in buckets))
            if norm:
                buckets = [value / norm for value in buckets]
            vectors.append(buckets)
        return vectors


@pytest.fixture(autouse=True)
def fake_embedding_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_root = tmp_path / ".cache" / "omnivorous"

    def _fake_resolve_backend(self):
        if self._backend is None:
            self._backend = _DefaultFakeEmbeddingBackend()
        return self._backend

    monkeypatch.setattr(embeddings_module, "default_embedding_root_dir", lambda: cache_root)
    monkeypatch.setattr(
        embeddings_module.LocalEmbeddingService,
        "_resolve_backend",
        _fake_resolve_backend,
    )


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal PDF fixture using pypdf."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

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
