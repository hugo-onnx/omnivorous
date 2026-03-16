"""Marker-pdf engine for scientific documents with LaTeX formula extraction."""

from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path

from omnivorous.converters.pdf._engine import PdfExtractionResult
from omnivorous.converters.pdf._postprocess import (
    fix_drop_caps,
    normalize_ligatures,
    strip_toc,
)

# Regex to count markdown tables (header + separator row)
_TABLE_RE = re.compile(r"^\|.+\|\n\|[\s\-:|]+\|", re.MULTILINE)

# Regex to extract markdown headings
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _ensure_marker() -> None:
    """Check that marker-pdf is installed, raising a clear error if not."""
    try:
        import marker  # noqa: F401
    except ImportError:
        raise ImportError(
            "marker-pdf is required for scientific mode. "
            "Install it with: pip install omnivorous[scientific]"
        )


@contextlib.contextmanager
def _suppress_tqdm():
    """Monkey-patch tqdm to force disable on all progress bars."""
    import tqdm as tqdm_module

    orig_init = tqdm_module.tqdm.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["disable"] = True
        orig_init(self, *args, **kwargs)

    tqdm_module.tqdm.__init__ = _patched_init
    try:
        yield
    finally:
        tqdm_module.tqdm.__init__ = orig_init


class MarkerEngine:
    """PDF extraction using marker-pdf for LaTeX formula reconstruction."""

    name: str = "marker"

    def extract(self, path: Path) -> PdfExtractionResult:
        _ensure_marker()

        # Silence noisy loggers before importing marker internals
        for name in ("marker", "surya", "datalab", "texify"):
            logging.getLogger(name).setLevel(logging.WARNING)

        from marker.converters.pdf import PdfConverter as MarkerPdfConverter
        from marker.models import create_model_dict

        with _suppress_tqdm():
            models = create_model_dict()
            converter = MarkerPdfConverter(artifact_dict=models)
            rendered = converter(str(path))

        content: str = rendered.markdown
        content = normalize_ligatures(content)
        content = fix_drop_caps(content)
        content = strip_toc(content)

        # Count tables and headings from the markdown output
        total_tables = len(_TABLE_RE.findall(content))
        headings = [m.group(0) for m in _HEADING_RE.finditer(content)]

        # Extract page count from metadata if available
        num_pages = rendered.metadata.get("pages", 0) if hasattr(rendered, "metadata") else 0

        return PdfExtractionResult(
            content=content,
            pages=num_pages,
            headings=headings,
            tables=total_tables,
            images=0,
        )
