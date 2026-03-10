"""Plain text converter."""

from __future__ import annotations

import re
from pathlib import Path

from agentmd.converters.base import BaseConverter
from agentmd.models import ConvertResult, DocumentMetadata
from agentmd.tokens import count_tokens, get_encoding_name

_GUTENBERG_START_RE = re.compile(r"^\*\*\*\s*START OF.*$", re.MULTILINE | re.IGNORECASE)
_GUTENBERG_END_RE = re.compile(r"^\*\*\*\s*END OF.*$", re.MULTILINE | re.IGNORECASE)


def _strip_gutenberg_boilerplate(text: str) -> str:
    start = _GUTENBERG_START_RE.search(text)
    if start:
        text = text[start.end():]
    end = _GUTENBERG_END_RE.search(text)
    if end:
        text = text[:end.start()]
    return text.strip()


class TxtConverter(BaseConverter):
    @property
    def name(self) -> str:
        return "Text"

    def convert(self, path: Path) -> ConvertResult:
        raw = path.read_text(encoding="utf-8-sig", errors="replace").strip()
        raw = _strip_gutenberg_boilerplate(raw)
        title = path.stem
        content = f"# {title}\n\n{raw}"

        metadata = DocumentMetadata(
            source=str(path),
            format="txt",
            title=title,
            tokens_estimate=count_tokens(content),
            encoding=get_encoding_name(),
        )
        return ConvertResult(content=content, metadata=metadata)
