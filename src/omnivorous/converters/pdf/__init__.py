"""PDF converter package with pluggable engine support.

Engines:
- ``pymupdf`` (default): Fast extraction via pymupdf4llm, no ML.
- ``marker``: Scientific mode with LaTeX formula reconstruction (requires ``[scientific]`` extra).
"""

from __future__ import annotations

from pathlib import Path

from omnivorous.converters.base import BaseConverter
from omnivorous.converters.pdf._engine import PdfEngine, PdfExtractionResult
from omnivorous.models import ConvertResult, DocumentMetadata
from omnivorous.tokens import count_tokens, get_encoding_name

_pdf_engine_name: str = "pymupdf"

_VALID_ENGINES = {"pymupdf", "marker"}


def set_pdf_engine(name: str) -> None:
    """Set which PDF engine to use for extraction."""
    global _pdf_engine_name
    if name not in _VALID_ENGINES:
        raise ValueError(f"Unknown PDF engine: {name!r}. Valid: {', '.join(sorted(_VALID_ENGINES))}")
    _pdf_engine_name = name


def get_pdf_engine() -> str:
    """Return the current PDF engine name."""
    return _pdf_engine_name


def _resolve_engine() -> PdfEngine:
    """Instantiate the currently selected PDF engine."""
    if _pdf_engine_name == "marker":
        from omnivorous.converters.pdf._marker import MarkerEngine

        return MarkerEngine()

    from omnivorous.converters.pdf._pymupdf import PyMuPdfEngine

    return PyMuPdfEngine()


class PdfConverter(BaseConverter):
    """PDF converter that delegates to the active engine."""

    @property
    def name(self) -> str:
        return "PDF"

    def convert(self, path: Path) -> ConvertResult:
        engine = _resolve_engine()
        result: PdfExtractionResult = engine.extract(path)

        metadata = DocumentMetadata(
            source=path.name,
            format="pdf",
            title=path.stem,
            pages=result.pages,
            headings=result.headings,
            tables=result.tables,
            images=result.images,
            tokens_estimate=count_tokens(result.content),
            encoding=get_encoding_name(),
        )
        return ConvertResult(content=result.content, metadata=metadata)


__all__ = [
    "PdfConverter",
    "PdfExtractionResult",
    "get_pdf_engine",
    "set_pdf_engine",
]
