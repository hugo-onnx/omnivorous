"""Markdown chunking strategies."""

from __future__ import annotations

import re
from pathlib import Path

from omnivorous.models import ChunkResult, DocumentMetadata
from omnivorous.tokens import count_tokens

_HEADING_SPLIT_RE = re.compile(r"(?=^#{1,6}\s)|(?=^.+\n[=-]{2,}\s*$)", re.MULTILINE)
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s")
_FENCE_RE = re.compile(r"^\s*`{3,}", re.MULTILINE)


def _has_open_fence(text: str) -> bool:
    return len(_FENCE_RE.findall(text)) % 2 != 0


def _is_heading_only(chunk: str) -> bool:
    return "\n" not in chunk and _HEADING_LINE_RE.match(chunk) is not None


def chunk_by_headings(content: str) -> list[str]:
    """Split markdown at heading boundaries.

    Note: reference-style links (e.g. [text][ref]) defined in one section but
    referenced in another will be separated from their definitions by this split.
    This is a known structural limitation of heading-based chunking.
    """
    parts = _HEADING_SPLIT_RE.split(content)
    chunks = [p.strip() for p in parts if p.strip()]

    # Merge heading-only chunks forward so they attach to the next section
    merged: list[str] = []
    for chunk in chunks:
        if merged and _is_heading_only(merged[-1]):
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)
    # If the last chunk is heading-only, merge it backward
    if len(merged) > 1 and _is_heading_only(merged[-1]):
        merged[-2] = merged[-2] + "\n\n" + merged[-1]
        merged.pop()

    return merged if merged else [content]


def chunk_by_tokens(content: str, chunk_size: int = 500) -> list[str]:
    """Split content into chunks of approximately `chunk_size` tokens."""
    paragraphs = content.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current and current_tokens + para_tokens > chunk_size:
            current_text = "\n\n".join(current)
            # Don't flush inside an open code fence
            if _has_open_fence(current_text):
                current.append(para)
                current_tokens += para_tokens
            # Don't flush a chunk that is only a heading
            elif _HEADING_LINE_RE.match(current_text) and current_text.count("\n") == 0:
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
