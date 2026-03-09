"""Tests for frontmatter utilities."""

from agentmd.frontmatter import add_frontmatter, parse_frontmatter


def test_add_frontmatter():
    result = add_frontmatter("# Hello", {"title": "Test", "format": "md"})
    assert result.startswith("---\n")
    assert "title: Test" in result
    assert "format: md" in result
    assert result.endswith("\n\n# Hello")


def test_add_frontmatter_empty():
    result = add_frontmatter("# Hello", {})
    assert result == "# Hello"


def test_add_frontmatter_filters_empty_values():
    result = add_frontmatter("# Hello", {"title": "Test", "pages": 0, "headings": []})
    assert "title: Test" in result
    assert "pages" not in result
    assert "headings" not in result


def test_parse_frontmatter():
    text = "---\ntitle: Test\nformat: md\n---\n\n# Hello"
    meta, body = parse_frontmatter(text)
    assert meta == {"title": "Test", "format": "md"}
    assert body == "# Hello"


def test_parse_frontmatter_none():
    meta, body = parse_frontmatter("# Hello")
    assert meta == {}
    assert body == "# Hello"


def test_roundtrip():
    original = "# Hello\n\nContent here."
    metadata = {"title": "Test", "source": "file.md"}
    with_fm = add_frontmatter(original, metadata)
    parsed_meta, parsed_body = parse_frontmatter(with_fm)
    assert parsed_meta == metadata
    assert parsed_body == original
