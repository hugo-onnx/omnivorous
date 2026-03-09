"""agentmd — Convert documents into agent-ready Markdown context."""

__version__ = "0.1.0"

from agentmd.models import ChunkResult, ConvertResult, DocumentMetadata
from agentmd.registry import get_converter, supported_extensions

__all__ = [
    "ChunkResult",
    "ConvertResult",
    "DocumentMetadata",
    "get_converter",
    "supported_extensions",
]
