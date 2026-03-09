"""PDF converter using pypdf."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from agentmd.converters.base import BaseConverter
from agentmd.models import ConvertResult, DocumentMetadata
from agentmd.tokens import count_tokens, get_encoding_name


class PdfConverter(BaseConverter):
    @property
    def name(self) -> str:
        return "PDF"

    def convert(self, path: Path) -> ConvertResult:
        reader = PdfReader(path)
        pages: list[str] = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"## Page {i}\n\n{text.strip()}")

        content = "\n\n".join(pages)
        headings = [f"Page {i}" for i in range(1, len(reader.pages) + 1)]

        metadata = DocumentMetadata(
            source=str(path),
            format="pdf",
            title=path.stem,
            pages=len(reader.pages),
            headings=headings,
            tokens_estimate=count_tokens(content),
            encoding=get_encoding_name(),
        )
        return ConvertResult(content=content, metadata=metadata)
