"""Golden-output snapshots for representative fixture conversions."""

from pathlib import Path

from omnivorous.converters.html import HtmlConverter
from omnivorous.converters.markdown import MarkdownConverter
from omnivorous.converters.pdf import PdfConverter, set_pdf_engine
from omnivorous.converters.txt import TxtConverter


def test_converter_goldens(fixtures_dir: Path):
    set_pdf_engine("pymupdf")
    cases = [
        ("readme.md", MarkdownConverter(), Path("tests/golden/readme.md")),
        ("notes.txt", TxtConverter(), Path("tests/golden/notes.md")),
        ("web.html", HtmlConverter(), Path("tests/golden/web.md")),
        ("document.pdf", PdfConverter(), Path("tests/golden/document.md")),
    ]

    for fixture_name, converter, golden_path in cases:
        result = converter.convert(fixtures_dir / fixture_name)
        assert result.content.strip() == golden_path.read_text(encoding="utf-8").strip()
