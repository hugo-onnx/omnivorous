"""Marker-pdf engine for scientific documents with LaTeX formula extraction."""

from __future__ import annotations

import contextlib
import logging
import re
import threading
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

_MARKER_MODELS = None
_MARKER_CONVERTER = None
_MARKER_MODELS_LOCK = threading.Lock()
_MARKER_CONVERTER_LOCK = threading.Lock()


def _ensure_marker() -> None:
    """Check that marker-pdf is installed, raising a clear error if not."""
    try:
        import marker  # noqa: F401
    except ImportError:
        raise ImportError(
            "marker-pdf is required for scientific mode. "
            "Install it with: pip install omnivorous[scientific]"
        )


def _load_marker_components():
    """Load marker components lazily so tests can patch them without the dependency."""
    from marker.converters.pdf import PdfConverter as MarkerPdfConverter
    from marker.models import create_model_dict

    return MarkerPdfConverter, create_model_dict


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


def _get_marker_models():
    """Build the marker model dictionary once per process."""
    global _MARKER_MODELS
    if _MARKER_MODELS is None:
        with _MARKER_MODELS_LOCK:
            if _MARKER_MODELS is None:
                _, create_model_dict = _load_marker_components()
                with _suppress_tqdm():
                    _MARKER_MODELS = create_model_dict()
    return _MARKER_MODELS


def _get_marker_converter():
    """Build the marker converter once per process."""
    global _MARKER_CONVERTER
    if _MARKER_CONVERTER is None:
        with _MARKER_CONVERTER_LOCK:
            if _MARKER_CONVERTER is None:
                MarkerPdfConverter, _ = _load_marker_components()
                with _suppress_tqdm():
                    _MARKER_CONVERTER = MarkerPdfConverter(artifact_dict=_get_marker_models())
    return _MARKER_CONVERTER


class MarkerEngine:
    """PDF extraction using marker-pdf for LaTeX formula reconstruction."""

    name: str = "marker"

    def extract(self, path: Path) -> PdfExtractionResult:
        _ensure_marker()

        # Silence noisy loggers before importing marker internals
        for name in ("marker", "surya", "datalab", "texify"):
            logging.getLogger(name).setLevel(logging.WARNING)

        with _suppress_tqdm():
            converter = _get_marker_converter()
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
