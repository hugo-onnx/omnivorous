"""Tests for markdown chunking."""

from pathlib import Path

from omnivorous.chunker import chunk_by_headings, chunk_by_tokens, chunk_markdown, write_chunks
from omnivorous.models import DocumentMetadata


def _meta() -> DocumentMetadata:
    return DocumentMetadata(source="test.md", format="markdown", title="Test")


def test_chunk_by_headings():
    content = "# Heading 1\n\nParagraph.\n\n## Heading 2\n\nMore text."
    chunks = chunk_by_headings(content)
    assert len(chunks) == 2
    assert chunks[0].startswith("# Heading 1")
    assert chunks[1].startswith("## Heading 2")


def test_chunk_by_headings_no_headings():
    content = "Just some text without headings."
    chunks = chunk_by_headings(content)
    assert len(chunks) == 1


def test_chunk_by_tokens():
    # Create content with many paragraphs
    content = "\n\n".join([f"Paragraph {i} with some words." for i in range(50)])
    chunks = chunk_by_tokens(content, chunk_size=20)
    assert len(chunks) > 1
    # Each chunk should be non-empty
    for c in chunks:
        assert c.strip()


def test_chunk_markdown_heading_strategy():
    content = "# A\n\nText.\n\n## B\n\nMore."
    result = chunk_markdown(content, _meta(), strategy="heading", chunk_size=5000)
    assert len(result.chunks) == 2


def test_chunk_markdown_tokens_strategy():
    content = "\n\n".join([f"Paragraph {i} with some content." for i in range(50)])
    result = chunk_markdown(content, _meta(), strategy="tokens", chunk_size=20)
    assert len(result.chunks) > 1


def test_write_chunks(tmp_path: Path):
    chunks = ["# Chunk 1\n\nContent.", "# Chunk 2\n\nMore content."]
    source = Path("test.md")
    paths = write_chunks(chunks, source, tmp_path)
    assert len(paths) == 2
    assert paths[0].name == "test_001.md"
    assert paths[1].name == "test_002.md"
    assert paths[0].read_text() == chunks[0]
    assert paths[1].read_text() == chunks[1]


def test_chunk_by_tokens_preserves_code_fences():
    code_block = "```python\ndef hello():\n    print('hi')\n```"
    content = f"Intro text.\n\n{code_block}\n\nAfter code."
    # Use a very small chunk size to force splitting attempts
    chunks = chunk_by_tokens(content, chunk_size=5)
    for chunk in chunks:
        fence_count = chunk.count("```")
        assert fence_count % 2 == 0, f"Chunk has unclosed fence: {chunk!r}"


def test_chunk_by_headings_merges_heading_only():
    content = "# A\n\n## B\n\nText under B."
    chunks = chunk_by_headings(content)
    # "# A" alone would be heading-only, so it should be merged with "## B\n\nText under B."
    assert len(chunks) == 1
    assert "# A" in chunks[0]
    assert "## B" in chunks[0]
    assert "Text under B." in chunks[0]


def test_chunk_by_headings_setext():
    content = "Title\n=====\n\nSome content.\n\nSubtitle\n--------\n\nMore content."
    chunks = chunk_by_headings(content)
    assert len(chunks) == 2
    assert "Title" in chunks[0]
    assert "Subtitle" in chunks[1]
