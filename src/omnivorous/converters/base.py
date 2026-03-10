"""Base converter abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from omnivorous.models import ConvertResult


class BaseConverter(ABC):
    """Abstract base class for document converters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this converter."""

    @abstractmethod
    def convert(self, path: Path) -> ConvertResult:
        """Convert a document file to markdown. Returns ConvertResult."""
