"""Markdown passthrough converter."""

from __future__ import annotations

import re
from pathlib import Path

from agentmd.converters.base import BaseConverter
from agentmd.models import ConvertResult, DocumentMetadata
from agentmd.tokens import count_tokens, get_encoding_name

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_RE = re.compile(r"(?:^\|.+\|$\n?){2,}", re.MULTILINE)


class MarkdownConverter(BaseConverter):
    @property
    def name(self) -> str:
        return "Markdown"

    def convert(self, path: Path) -> ConvertResult:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        headings = [m.group(2) for m in _HEADING_RE.finditer(content)]
        tables = len(_TABLE_RE.findall(content))

        # Use first heading as title, fall back to filename
        title = headings[0] if headings else path.stem

        metadata = DocumentMetadata(
            source=str(path),
            format="markdown",
            title=title,
            headings=headings,
            tables=tables,
            tokens_estimate=count_tokens(content),
            encoding=get_encoding_name(),
        )
        return ConvertResult(content=content, metadata=metadata)
