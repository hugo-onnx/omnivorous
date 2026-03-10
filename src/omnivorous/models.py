"""Data models for omnivorous."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document during conversion."""

    source: str
    format: str
    title: str = ""
    pages: int = 0
    headings: list[str] = field(default_factory=list)
    tables: int = 0
    tokens_estimate: int = 0
    encoding: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "format": self.format,
            "title": self.title,
            "pages": self.pages,
            "headings": self.headings,
            "tables": self.tables,
            "tokens_estimate": self.tokens_estimate,
            "encoding": self.encoding,
        }


@dataclass
class ConvertResult:
    """Result of converting a document to markdown."""

    content: str
    metadata: DocumentMetadata


@dataclass
class ChunkResult:
    """Result of chunking a markdown document."""

    chunks: list[str]
    metadata: DocumentMetadata
    output_files: list[Path] = field(default_factory=list)
