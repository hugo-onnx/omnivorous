"""Markdown chunking strategies."""

from __future__ import annotations

import re
from pathlib import Path

from agentmd.models import ChunkResult, DocumentMetadata
from agentmd.tokens import count_tokens

_HEADING_SPLIT_RE = re.compile(r"(?=^#{1,6}\s)", re.MULTILINE)
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s")


def chunk_by_headings(content: str) -> list[str]:
    """Split markdown at heading boundaries."""
    parts = _HEADING_SPLIT_RE.split(content)
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks if chunks else [content]


def chunk_by_tokens(content: str, chunk_size: int = 500) -> list[str]:
    """Split content into chunks of approximately `chunk_size` tokens."""
    paragraphs = content.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current and current_tokens + para_tokens > chunk_size:
            # Don't flush a chunk that is only a heading — keep it with the next paragraph
            current_text = "\n\n".join(current)
            if _HEADING_LINE_RE.match(current_text) and current_text.count("\n") == 0:
                current.append(para)
                current_tokens += para_tokens
            else:
                chunks.append(current_text)
                current = [para]
                current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_markdown(
    content: str,
    metadata: DocumentMetadata,
    strategy: str = "heading",
    chunk_size: int = 500,
) -> ChunkResult:
    """Main chunking entry point.

    strategy: "heading" splits at headings first, then by tokens if any chunk is too large.
              "tokens" splits purely by token count.
    """
    if strategy == "tokens":
        chunks = chunk_by_tokens(content, chunk_size)
    else:
        # heading-first with token fallback
        heading_chunks = chunk_by_headings(content)
        no_headings_found = len(heading_chunks) == 1
        chunks = []
        for chunk in heading_chunks:
            if count_tokens(chunk) > chunk_size:
                fallback_size = max(chunk_size, 4000) if no_headings_found else chunk_size
                chunks.extend(chunk_by_tokens(chunk, fallback_size))
            else:
                chunks.append(chunk)

    return ChunkResult(chunks=chunks, metadata=metadata)


def write_chunks(chunks: list[str], source_path: Path, output_dir: Path) -> list[Path]:
    """Write chunks to numbered files. Returns list of written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    written: list[Path] = []
    for i, chunk in enumerate(chunks, 1):
        out_path = output_dir / f"{stem}_{i:03d}.md"
        out_path.write_text(chunk, encoding="utf-8")
        written.append(out_path)
    return written
