"""Shared post-processing utilities for PDF extraction."""

from __future__ import annotations

import re
from collections import Counter

_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

_DROP_CAP_RE = re.compile(r"^([A-Z])\n([A-Z])", re.MULTILINE)


def normalize_ligatures(text: str) -> str:
    """Replace Unicode ligature characters with their ASCII equivalents."""
    for lig, replacement in _LIGATURES.items():
        text = text.replace(lig, replacement)
    return text


def fix_drop_caps(text: str) -> str:
    """Rejoin drop caps that got split across lines."""
    return _DROP_CAP_RE.sub(r"\1\2", text)


def detect_repeated_lines(pages_text: list[str], threshold: float = 0.5) -> set[str]:
    """Detect lines repeated across many pages (headers/footers)."""
    line_counts: Counter[str] = Counter()
    for text in pages_text:
        for line in set(text.strip().splitlines()):
            stripped = line.strip()
            if stripped and len(stripped) < 80:
                line_counts[stripped] += 1
    min_count = max(2, int(len(pages_text) * threshold))
    return {line for line, count in line_counts.items() if count >= min_count}


def strip_lines(text: str, lines_to_remove: set[str]) -> str:
    """Remove specific lines from text."""
    return "\n".join(ln for ln in text.splitlines() if ln.strip() not in lines_to_remove)


# --- TOC stripping ---

_TOC_START_RE = re.compile(
    r"^(#{1,6}\s+)?(Table of Contents|Contents)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_DOTTED_LEADER_RE = re.compile(r"\.{3,}")
_PAGE_NUM_ONLY_RE = re.compile(r"^\d{1,4}\s*$")
_SECTION_ENTRY_RE = re.compile(r"^\d+(\.\d+)*\s+\S.*\d+\s*$")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_NUMERIC_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\s+\S.*$")
_PAGE_MARKER_RE = re.compile(r"\[Page\s+\d+\]", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```")


def _is_toc_line(line: str) -> bool:
    """Return True if *line* looks like a TOC entry (or blank)."""
    stripped = line.strip()
    if not stripped:
        return True
    if _DOTTED_LEADER_RE.search(stripped):
        return True
    if stripped.startswith("|"):
        return True
    if _PAGE_NUM_ONLY_RE.match(stripped):
        return True
    if _SECTION_ENTRY_RE.match(stripped):
        return True
    return False


def _is_prose(line: str) -> bool:
    """Return True if *line* looks like real prose (40+ chars, not TOC-like)."""
    return len(line.strip()) >= 40 and not _is_toc_line(line)


def _is_ignorable_boundary_line(line: str) -> bool:
    """Return True for structural lines that should not end a TOC block."""
    stripped = line.strip()
    if not stripped:
        return True
    if _CODE_FENCE_RE.match(stripped):
        return True
    if _PAGE_MARKER_RE.search(stripped):
        return True
    return False


def _is_content_heading(line: str) -> bool:
    """Return True if *line* looks like a real section heading, not a TOC entry."""
    stripped = line.strip()
    if not stripped or _is_ignorable_boundary_line(stripped):
        return False
    if _HEADING_RE.match(stripped):
        return True
    return bool(_NUMERIC_HEADING_RE.match(stripped) and not _SECTION_ENTRY_RE.match(stripped))


def _is_content_boundary(lines: list[str], start: int) -> bool:
    """Return True if *start* begins real content after a TOC block."""
    prose_count = 0
    heading_seen = False
    examined = 0

    for line in lines[start:]:
        stripped = line.strip()
        if _is_ignorable_boundary_line(stripped):
            continue
        examined += 1
        if _is_toc_line(stripped):
            return False
        if _is_content_heading(stripped):
            heading_seen = True
        elif _is_prose(stripped):
            prose_count += 1
        else:
            return False

        if heading_seen and prose_count >= 1:
            return True
        if prose_count >= 2:
            return True
        if examined >= 8:
            break

    return False


def strip_toc(text: str) -> str:
    """Detect and remove a Table of Contents section from *text*."""
    m = _TOC_START_RE.search(text)
    if m is None:
        return text

    toc_start = m.start()
    rest = text[m.end():]
    lines = rest.split("\n")

    end_offset = len(rest)  # default: strip to end
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_content_heading(line) or _is_prose(line):
            if _is_content_boundary(lines, i):
                end_offset = sum(len(ln) + 1 for ln in lines[:i])
                return text[:toc_start].rstrip("\n") + "\n\n" + rest[end_offset:]
        i += 1

    # TOC runs to end of document
    return text[:toc_start].rstrip("\n") + "\n"
