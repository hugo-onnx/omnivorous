"""Tests for plain text converter."""

from pathlib import Path

from omnivorous.converters.txt import TxtConverter


def test_convert_txt(fixtures_dir: Path):
    converter = TxtConverter()
    result = converter.convert(fixtures_dir / "notes.txt")

    assert result.content.startswith("# notes")
    assert "plain text document" in result.content
    assert result.metadata.format == "txt"
    assert result.metadata.source == "notes.txt"
    assert result.metadata.title == "notes"
    assert result.metadata.tokens_estimate > 0


def test_txt_name():
    assert TxtConverter().name == "Text"


def test_bom_stripped(tmp_path: Path):
    f = tmp_path / "bom.txt"
    f.write_bytes(b"\xef\xbb\xbfHello BOM")
    result = TxtConverter().convert(f)
    assert "\ufeff" not in result.content
    assert "Hello BOM" in result.content


def test_gutenberg_boilerplate_stripped(tmp_path: Path):
    text = (
        "The Project Gutenberg eBook\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK ***\n"
        "Actual content here.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK ***\n"
        "Small print license stuff."
    )
    f = tmp_path / "book.txt"
    f.write_text(text, encoding="utf-8")
    result = TxtConverter().convert(f)
    assert "Actual content here." in result.content
    assert "START OF" not in result.content
    assert "END OF" not in result.content
    assert "Small print" not in result.content
    assert "Project Gutenberg eBook" not in result.content


def test_non_gutenberg_unchanged(tmp_path: Path):
    text = "Just a normal document.\nWith multiple lines."
    f = tmp_path / "normal.txt"
    f.write_text(text, encoding="utf-8")
    result = TxtConverter().convert(f)
    assert "Just a normal document." in result.content
    assert "With multiple lines." in result.content
