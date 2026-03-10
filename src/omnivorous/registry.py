"""Converter registry — maps file extensions to converter classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnivorous.converters.base import BaseConverter

_registry: dict[str, type[BaseConverter]] = {}


def register_converter(extensions: list[str], cls: type[BaseConverter]) -> None:
    """Register a converter class for the given file extensions."""
    for ext in extensions:
        ext = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        _registry[ext] = cls


def get_converter(ext: str) -> BaseConverter:
    """Get an instantiated converter for the given file extension."""
    ext = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
    cls = _registry.get(ext)
    if cls is None:
        raise ValueError(f"No converter registered for extension: {ext}")
    return cls()


def supported_extensions() -> list[str]:
    """Return a sorted list of supported file extensions."""
    return sorted(_registry.keys())


def ensure_registry_loaded() -> None:
    """Ensure converters are registered by importing the converters package."""
    if not _registry:
        import omnivorous.converters  # noqa: F401
