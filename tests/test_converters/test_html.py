"""Tests for HTML converter."""

from pathlib import Path

from omnivorous.converters.html import HtmlConverter


def test_convert_html(fixtures_dir: Path):
    converter = HtmlConverter()
    result = converter.convert(fixtures_dir / "web.html")

    assert result.metadata.format == "html"
    assert result.metadata.source == "web.html"
    assert result.metadata.title == "Sample HTML Document"
    assert result.metadata.tables == 1
    assert result.metadata.tokens_estimate > 0
    # Should contain converted headings
    assert "Main Heading" in result.content


def test_html_name():
    assert HtmlConverter().name == "HTML"


def test_heading_link_stripped(tmp_path: Path):
    html = "<html><body><h1><a href='/url'>Linked Title</a></h1><p>Body.</p></body></html>"
    f = tmp_path / "links.html"
    f.write_text(html, encoding="utf-8")
    result = HtmlConverter().convert(f)
    assert "# Linked Title" in result.metadata.headings
    # The heading text should not contain markdown link syntax
    for h in result.metadata.headings:
        assert "](/" not in h


def test_curly_quotes_normalized_html(tmp_path: Path):
    html = "<html><body><h2>The \u2018Smart\u2019 Way</h2><p>Content.</p></body></html>"
    f = tmp_path / "quotes.html"
    f.write_text(html, encoding="utf-8")
    result = HtmlConverter().convert(f)
    assert "## The 'Smart' Way" in result.metadata.headings


def test_html_image_placeholder_with_alt(tmp_path: Path):
    html = '<html><body><p>Text</p><img alt="diagram" src="x.png"><p>More.</p></body></html>'
    f = tmp_path / "img.html"
    f.write_text(html, encoding="utf-8")
    result = HtmlConverter().convert(f)
    assert "![diagram]()" in result.content
    assert result.metadata.images == 1


def test_html_image_placeholder_no_alt(tmp_path: Path):
    html = '<html><body><p>Text</p><img src="x.png"><p>More.</p></body></html>'
    f = tmp_path / "img_noalt.html"
    f.write_text(html, encoding="utf-8")
    result = HtmlConverter().convert(f)
    assert "![image]()" in result.content
    assert result.metadata.images == 1


def test_html_cleaner_handles_decomposed_children_with_classes(tmp_path: Path):
    html = (
        "<html><body>"
        "<nav class='breadcrumb'><a class='headerlink' href='#'>skip</a></nav>"
        "<main><h1>HTTP Caching</h1><p>Cache-Control directives.</p></main>"
        "</body></html>"
    )
    f = tmp_path / "nested_classes.html"
    f.write_text(html, encoding="utf-8")

    result = HtmlConverter().convert(f)

    assert "# HTTP Caching" in result.content
    assert "Cache\\-Control directives." in result.content


def test_html_cleaner_keeps_main_layout_container_with_sidebars_in_class(tmp_path: Path):
    html = (
        "<html><body>"
        "<div class='page-layout__main'>"
        "<div class='layout__2-sidebars-inline reference-layout'>"
        "<main><h1>HTTP caching</h1><p>The HTTP cache stores a response.</p></main>"
        "</div>"
        "<aside class='left-sidebar'>Navigation</aside>"
        "</div>"
        "</body></html>"
    )
    f = tmp_path / "main_layout.html"
    f.write_text(html, encoding="utf-8")

    result = HtmlConverter().convert(f)

    assert "# HTTP caching" in result.content
    assert "The HTTP cache stores a response." in result.content


def test_html_promotes_heading_lists_to_sections(tmp_path: Path):
    html = (
        "<html><body><main><ol>"
        "<li><h2><span>1.</span> Understand users and their needs</h2>"
        "<p><a href='/point-1'>Read more about point 1</a></p></li>"
        "<li><h2><span>2.</span> Solve a whole problem for users</h2>"
        "<p><a href='/point-2'>Read more about point 2</a></p></li>"
        "</ol></main></body></html>"
    )
    f = tmp_path / "service_points.html"
    f.write_text(html, encoding="utf-8")

    result = HtmlConverter().convert(f)

    assert "## 1\\. Understand users and their needs" in result.content
    assert "## 2\\. Solve a whole problem for users" in result.content
    assert "1. ##" not in result.content
