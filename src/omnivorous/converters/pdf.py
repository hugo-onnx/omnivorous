"""PDF converter using pypdf."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

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


class PdfConverter(BaseConverter):
    @property
    def name(self) -> str:
        return "PDF"

    def convert(self, path: Path) -> ConvertResult:
        reader = PdfReader(path)

        # Extract and normalize text per page
        raw_pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            text = _normalize_ligatures(text)
            text = _DROP_CAP_RE.sub(r"\1\2", text)
            raw_pages.append(text)

        # Detect and strip repeated header/footer lines
        repeated = _detect_repeated_lines(raw_pages)
        if repeated:
            raw_pages = [_strip_lines(t, repeated) for t in raw_pages]

        # Format with page markers
        pages: list[str] = []
        for i, text in enumerate(raw_pages, 1):
            if text.strip():
                pages.append(f"## Page {i}\n\n{text.strip()}")

        content = "\n\n".join(pages)

        metadata = DocumentMetadata(
            source=path.name,
            format="pdf",
            title=path.stem,
            pages=len(reader.pages),
            headings=[],
            tokens_estimate=count_tokens(content),
            encoding=get_encoding_name(),
        )
        return ConvertResult(content=content, metadata=metadata)
