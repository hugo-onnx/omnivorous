"""Document inspection — extract and display metadata."""

from __future__ import annotations

from pathlib import Path

from omnivorous.models import DocumentMetadata
from omnivorous.registry import ensure_registry_loaded, get_converter


def inspect_file(path: Path) -> DocumentMetadata:
    """Inspect a file and return its metadata without writing output."""
    ensure_registry_loaded()
    ext = path.suffix.lower()
    converter = get_converter(ext)
    result = converter.convert(path)
    return result.metadata
