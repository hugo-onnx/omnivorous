"""Plain text converter."""

from __future__ import annotations

from pathlib import Path

from agentmd.converters.base import BaseConverter
from agentmd.models import ConvertResult, DocumentMetadata
from agentmd.tokens import count_tokens, get_encoding_name


class TxtConverter(BaseConverter):
    @property
    def name(self) -> str:
        return "Text"

    def convert(self, path: Path) -> ConvertResult:
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
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
