"""Tests for converter registry."""

from agentmd.converters.html import HtmlConverter
from agentmd.converters.markdown import MarkdownConverter
from agentmd.converters.pdf import PdfConverter
from agentmd.converters.txt import TxtConverter
from agentmd.registry import ensure_registry_loaded, get_converter, supported_extensions


def test_supported_extensions():
    ensure_registry_loaded()
    exts = supported_extensions()
    assert ".pdf" in exts
    assert ".docx" in exts
    assert ".html" in exts
    assert ".htm" in exts
    assert ".md" in exts
    assert ".txt" in exts


def test_get_converter():
    ensure_registry_loaded()
    assert isinstance(get_converter(".pdf"), PdfConverter)
    assert isinstance(get_converter(".html"), HtmlConverter)
    assert isinstance(get_converter(".md"), MarkdownConverter)
    assert isinstance(get_converter(".txt"), TxtConverter)


def test_get_converter_case_insensitive():
    ensure_registry_loaded()
    assert isinstance(get_converter(".PDF"), PdfConverter)
    assert isinstance(get_converter(".Html"), HtmlConverter)


def test_get_converter_unknown():
    ensure_registry_loaded()
    import pytest

    with pytest.raises(ValueError, match="No converter registered"):
        get_converter(".xyz")
