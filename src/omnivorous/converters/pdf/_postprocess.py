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
