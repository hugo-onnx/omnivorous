"""omnivorous — Convert documents into agent-ready Markdown context."""

__version__ = "0.1.0"

from omnivorous.models import ChunkResult, ConvertResult, DocumentMetadata
from omnivorous.registry import get_converter, supported_extensions

__all__ = [
    "ChunkResult",
    "ConvertResult",
    "DocumentMetadata",
    "get_converter",
    "supported_extensions",
]
