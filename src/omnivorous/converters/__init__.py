"""Converter registry initialization — imports and registers all converters."""

from omnivorous.converters.docx import DocxConverter
from omnivorous.converters.html import HtmlConverter
from omnivorous.converters.markdown import MarkdownConverter
from omnivorous.converters.pdf import PdfConverter
from omnivorous.converters.txt import TxtConverter
from omnivorous.registry import register_converter

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
