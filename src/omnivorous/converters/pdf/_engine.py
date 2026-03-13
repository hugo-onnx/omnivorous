"""PDF engine protocol and extraction result."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class PdfExtractionResult:
    """Result from a PDF engine extraction."""

    content: str
    pages: int
    headings: list[str] = field(default_factory=list)
    tables: int = 0
    images: int = 0


@runtime_checkable
class PdfEngine(Protocol):
    """Protocol for PDF extraction engines."""

    name: str

    def extract(self, path: Path) -> PdfExtractionResult: ...
