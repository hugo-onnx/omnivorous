"""Converter registry initialization — imports and registers all converters."""

from agentmd.converters.docx import DocxConverter
from agentmd.converters.html import HtmlConverter
from agentmd.converters.markdown import MarkdownConverter
from agentmd.converters.pdf import PdfConverter
from agentmd.converters.txt import TxtConverter
from agentmd.registry import register_converter

register_converter([".pdf"], PdfConverter)
register_converter([".docx"], DocxConverter)
register_converter([".html", ".htm"], HtmlConverter)
register_converter([".md", ".markdown"], MarkdownConverter)
register_converter([".txt"], TxtConverter)

__all__ = [
    "DocxConverter",
    "HtmlConverter",
    "MarkdownConverter",
    "PdfConverter",
    "TxtConverter",
]
