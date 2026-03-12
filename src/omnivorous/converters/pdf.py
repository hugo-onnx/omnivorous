"""PDF converter using pdfplumber."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pdfplumber

from omnivorous.converters.base import BaseConverter
from omnivorous.models import ConvertResult, DocumentMetadata
from omnivorous.tokens import count_tokens, get_encoding_name

_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
}

_DROP_CAP_RE = re.compile(r"^([A-Z])\n([A-Z])", re.MULTILINE)


def _normalize_ligatures(text: str) -> str:
    for lig, replacement in _LIGATURES.items():
        text = text.replace(lig, replacement)
    return text


def _detect_repeated_lines(pages_text: list[str], threshold: float = 0.5) -> set[str]:
    line_counts: Counter[str] = Counter()
    for text in pages_text:
        for line in set(text.strip().splitlines()):
            stripped = line.strip()
            if stripped and len(stripped) < 80:
                line_counts[stripped] += 1
    min_count = max(2, int(len(pages_text) * threshold))
    return {line for line, count in line_counts.items() if count >= min_count}


def _strip_lines(text: str, lines_to_remove: set[str]) -> str:
    return "\n".join(ln for ln in text.splitlines() if ln.strip() not in lines_to_remove)


def _table_to_markdown(table_data: list[list[str | None]]) -> str:
    """Convert pdfplumber table output (list of rows) to markdown table syntax."""
    if not table_data:
        return ""

    rows = [[cell or "" for cell in row] for row in table_data]

    lines: list[str] = []
    header = rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_page_content(
    page: pdfplumber.page.Page, lines_to_remove: set[str]
) -> tuple[str, int, int]:
    """Extract content from a single pdfplumber page, interleaving text and tables.

    Returns (page_content, table_count, image_count).
    """
    tables = page.find_tables()
    image_count = len(page.images) if page.images else 0

    if not tables:
        # No tables — extract full page text
        text = page.extract_text() or ""
        text = _normalize_ligatures(text)
        text = _DROP_CAP_RE.sub(r"\1\2", text)
        if lines_to_remove:
            text = _strip_lines(text, lines_to_remove)
        return (text.strip(), 0, image_count)

    # Sort tables by vertical position (top of bounding box)
    sorted_tables = sorted(tables, key=lambda t: t.bbox[1])

    parts: list[str] = []
    table_count = 0
    page_height = float(page.height)
    page_width = float(page.width)

    # Track the current vertical position
    current_y = 0.0

    for table in sorted_tables:
        table_top = table.bbox[1]
        table_bottom = table.bbox[3]

        # Extract text above this table
        if table_top > current_y:
            crop_box = (0, current_y, page_width, table_top)
            try:
                cropped = page.crop(crop_box)
                text = cropped.extract_text() or ""
            except Exception:
                text = ""
            text = _normalize_ligatures(text)
            text = _DROP_CAP_RE.sub(r"\1\2", text)
            if lines_to_remove:
                text = _strip_lines(text, lines_to_remove)
            if text.strip():
                parts.append(text.strip())

        # Convert table to markdown
        table_data = table.extract()
        md = _table_to_markdown(table_data)
        if md:
            table_count += 1
            parts.append(md)

        current_y = table_bottom

    # Extract text below the last table
    if current_y < page_height:
        crop_box = (0, current_y, page_width, page_height)
        try:
            cropped = page.crop(crop_box)
            text = cropped.extract_text() or ""
        except Exception:
            text = ""
        text = _normalize_ligatures(text)
        text = _DROP_CAP_RE.sub(r"\1\2", text)
        if lines_to_remove:
            text = _strip_lines(text, lines_to_remove)
        if text.strip():
            parts.append(text.strip())

    return ("\n\n".join(parts), table_count, image_count)


class PdfConverter(BaseConverter):
    @property
    def name(self) -> str:
        return "PDF"

    def convert(self, path: Path) -> ConvertResult:
        with pdfplumber.open(path) as pdf:
            # First pass: extract raw text per page for header/footer detection
            raw_pages: list[str] = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                raw_pages.append(text)

            repeated = _detect_repeated_lines(raw_pages)

            # Second pass: extract content with tables interleaved
            pages: list[str] = []
            total_tables = 0
            total_images = 0

            for i, page in enumerate(pdf.pages, 1):
                page_content, table_count, image_count = _extract_page_content(
                    page, repeated
                )
                total_tables += table_count
                total_images += image_count
                if page_content.strip():
                    pages.append(f"## Page {i}\n\n{page_content.strip()}")

            content = "\n\n".join(pages)
            num_pages = len(pdf.pages)

        metadata = DocumentMetadata(
            source=path.name,
            format="pdf",
            title=path.stem,
            pages=num_pages,
            headings=[],
            tables=total_tables,
            images=total_images,
            tokens_estimate=count_tokens(content),
            encoding=get_encoding_name(),
        )
        return ConvertResult(content=content, metadata=metadata)
