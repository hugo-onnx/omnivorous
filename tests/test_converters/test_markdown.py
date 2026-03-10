"""Tests for markdown converter."""

from pathlib import Path

from agentmd.converters.markdown import MarkdownConverter


def test_convert_markdown(fixtures_dir: Path):
    converter = MarkdownConverter()
    result = converter.convert(fixtures_dir / "readme.md")

    assert "# Sample Document" in result.content
    assert result.metadata.format == "markdown"
    assert result.metadata.title == "Sample Document"
    assert len(result.metadata.headings) == 4  # Sample Document, Section One, Section Two, Subsection
    assert result.metadata.tables == 1
    assert result.metadata.tokens_estimate > 0


def test_markdown_name():
    assert MarkdownConverter().name == "Markdown"


def test_mdbook_directives_stripped(tmp_path: Path):
    content = "# Title\n\n{{#rustdoc_include path/to/file.rs}}\n\nSome text."
    f = tmp_path / "book.md"
    f.write_text(content, encoding="utf-8")
    result = MarkdownConverter().convert(f)
    assert "{{#rustdoc_include" not in result.content
    assert "Some text." in result.content


def test_html_tags_stripped(tmp_path: Path):
    content = '# Title\n\n<span class="x">important</span> text <br/> here.'
    f = tmp_path / "tagged.md"
    f.write_text(content, encoding="utf-8")
    result = MarkdownConverter().convert(f)
    assert "<span" not in result.content
    assert "</span>" not in result.content
    assert "<br/>" not in result.content
    assert "important" in result.content
    assert "text" in result.content


def test_setext_headings_detected(tmp_path: Path):
    content = "My Title\n========\n\nSome content.\n\nSubtitle\n--------\n\nMore content."
    f = tmp_path / "setext.md"
    f.write_text(content, encoding="utf-8")
    result = MarkdownConverter().convert(f)
    assert "My Title" in result.metadata.headings
    assert "Subtitle" in result.metadata.headings


def test_curly_quotes_normalized(tmp_path: Path):
    content = "# The \u201cBig\u201d Idea\n\nSome text."
    f = tmp_path / "quotes.md"
    f.write_text(content, encoding="utf-8")
    result = MarkdownConverter().convert(f)
    assert 'The "Big" Idea' in result.metadata.headings
