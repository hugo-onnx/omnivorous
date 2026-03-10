"""Markdown passthrough converter."""

from __future__ import annotations

import re
from pathlib import Path

from omnivorous.converters.base import BaseConverter
from omnivorous.models import ConvertResult, DocumentMetadata
from omnivorous.tokens import count_tokens, get_encoding_name

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_SETEXT_RE = re.compile(r"^(.+)\n(=+|-+)\s*$", re.MULTILINE)
_TABLE_RE = re.compile(r"(?:^\|.+\|$\n?){2,}", re.MULTILINE)
_MDBOOK_RE = re.compile(r"\{\{#\w+[^}]*\}\}")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


def _normalize_quotes(text: str) -> str:
    return text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')


class MarkdownConverter(BaseConverter):
    @property
    def name(self) -> str:
        return "Markdown"

    def convert(self, path: Path) -> ConvertResult:
        content = path.read_text(encoding="utf-8", errors="replace").strip()

        # Strip mdbook directives and raw HTML tags
        content = _MDBOOK_RE.sub("", content)
        content = _HTML_TAG_RE.sub("", content)

        # Extract ATX and setext headings, sorted by position
        atx = [(m.start(), m.group(2)) for m in _HEADING_RE.finditer(content)]
        setext = [(m.start(), m.group(1).strip()) for m in _SETEXT_RE.finditer(content)]
        all_headings = sorted(atx + setext, key=lambda x: x[0])
        headings = [_normalize_quotes(h[1]) for h in all_headings]

        tables = len(_TABLE_RE.findall(content))

        # Use first heading as title, fall back to filename
        title = headings[0] if headings else path.stem

        metadata = DocumentMetadata(
            source=path.name,
            format="markdown",
            title=title,
            headings=headings,
            tables=tables,
            tokens_estimate=count_tokens(content),
            encoding=get_encoding_name(),
        )
        return ConvertResult(content=content, metadata=metadata)
