"""Tests for markdown converter."""

from pathlib import Path

from omnivorous.converters.markdown import MarkdownConverter


def test_convert_markdown(fixtures_dir: Path):
    converter = MarkdownConverter()
    result = converter.convert(fixtures_dir / "readme.md")

    assert "# Sample Document" in result.content
    assert result.metadata.format == "markdown"
    assert result.metadata.source == "readme.md"
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
    assert "# My Title" in result.metadata.headings
    assert "## Subtitle" in result.metadata.headings


def test_curly_quotes_normalized(tmp_path: Path):
    content = "# The \u201cBig\u201d Idea\n\nSome text."
    f = tmp_path / "quotes.md"
    f.write_text(content, encoding="utf-8")
    result = MarkdownConverter().convert(f)
    assert '# The "Big" Idea' in result.metadata.headings


def test_image_count(tmp_path: Path):
    content = "# Doc\n\n![photo](img.png)\n\nText.\n\n![diagram](diag.svg)\n"
    f = tmp_path / "images.md"
    f.write_text(content, encoding="utf-8")
    result = MarkdownConverter().convert(f)
    assert result.metadata.images == 2
    # Images should be preserved in content
    assert "![photo](img.png)" in result.content
    assert "![diagram](diag.svg)" in result.content


def test_no_images(tmp_path: Path):
    content = "# Doc\n\nNo images here."
    f = tmp_path / "noimages.md"
    f.write_text(content, encoding="utf-8")
    result = MarkdownConverter().convert(f)
    assert result.metadata.images == 0


def test_frontmatter_stripped_from_content(tmp_path: Path):
    content = "---\ntitle: Test\nformat: markdown\n---\n\n# Real Title\n\nBody text."
    f = tmp_path / "fm.md"
    f.write_text(content, encoding="utf-8")
    result = MarkdownConverter().convert(f)
    assert not result.content.startswith("---")
    assert "# Real Title" in result.content
    assert result.metadata.title == "Real Title"


def test_frontmatter_not_detected_as_setext_heading(tmp_path: Path):
    content = "---\ntitle: and more.\n---\n\n# Actual Title\n\nBody."
    f = tmp_path / "fm_setext.md"
    f.write_text(content, encoding="utf-8")
    result = MarkdownConverter().convert(f)
    # "and more." should NOT appear as a heading
    for h in result.metadata.headings:
        assert "and more." not in h
    assert result.metadata.title == "Actual Title"
