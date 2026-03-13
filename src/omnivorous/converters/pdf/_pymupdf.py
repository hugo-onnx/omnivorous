"""PyMuPDF + pymupdf4llm engine (default, no ML)."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pymupdf4llm

from omnivorous.converters.pdf._engine import PdfExtractionResult
from omnivorous.converters.pdf._postprocess import (
    detect_repeated_lines,
    fix_drop_caps,
    normalize_ligatures,
    strip_lines,
)

# Regex to count markdown tables (header + separator row)
_TABLE_RE = re.compile(r"^\|.+\|\n\|[\s\-:|]+\|", re.MULTILINE)

# Regex to extract markdown headings
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class PyMuPdfEngine:
    """PDF extraction using pymupdf4llm for markdown and pymupdf for metadata."""

    name: str = "pymupdf"

    def extract(self, path: Path) -> PdfExtractionResult:
        # Extract markdown via pymupdf4llm
        raw_md: str = pymupdf4llm.to_markdown(str(path))

        # Post-process: ligatures, drop caps
        content = normalize_ligatures(raw_md)
        content = fix_drop_caps(content)

        # Open with pymupdf for metadata
        doc = pymupdf.open(str(path))
        num_pages = len(doc)

        # Detect repeated headers/footers using per-page text
        pages_text: list[str] = []
        total_images = 0
        for page in doc:
            pages_text.append(page.get_text())
            total_images += len(page.get_images())
        doc.close()

        # Strip repeated header/footer lines
        repeated = detect_repeated_lines(pages_text)
        if repeated:
            content = strip_lines(content, repeated)

        # Count tables from markdown output
        total_tables = len(_TABLE_RE.findall(content))

        # Extract headings from markdown
        headings = [m.group(0) for m in _HEADING_RE.finditer(content)]

        return PdfExtractionResult(
            content=content,
            pages=num_pages,
            headings=headings,
            tables=total_tables,
            images=total_images,
        )
