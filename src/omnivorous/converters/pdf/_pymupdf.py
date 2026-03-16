"""PyMuPDF + pymupdf4llm engine (default, no ML)."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pymupdf4llm

from omnivorous.converters.pdf._engine import PdfExtractionResult
from omnivorous.converters.pdf._postprocess import (
    detect_repeated_lines,
    fix_drop_caps,
    normalize_ligatures,
    strip_lines,
    strip_toc,
)
from omnivorous.tokens import count_tokens

# Regex to count markdown tables (header + separator row)
_TABLE_RE = re.compile(r"^\|.+\|\n\|[\s\-:|]+\|", re.MULTILINE)
_TABLE_LINE_RE = re.compile(r"^\|.*\|$", re.MULTILINE)
_PLACEHOLDER_COLUMN_RE = re.compile(r"\bCol\d+\b")
_NUMERIC_HEADING_RE = re.compile(r"^(?P<section>\d+(?:\.\d+)*\.?)\s+(?P<title>[A-Z].+)$")
_NAMED_HEADING_RE = re.compile(
    r"^(Abstract|Background|Conclusion|Conclusions|Copyright Notice|Introduction|"
    r"References|Status of This Memo|Acknowledg(?:e)?ments?|Appendix(?:es)?)$"
)
_TITLE_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'./-]*")
_TITLE_NOISE_RE = re.compile(
    r"\b(attribution|copyright|grants|journalistic|license|permission|provided|"
    r"reproduce|reserved|rights|scholarly|solely)\b",
    re.IGNORECASE,
)
_OUTER_CODE_FENCE_RE = re.compile(r"^\s*```[^\n]*\n(?P<body>.*)\n```\s*$", re.DOTALL)

# Regex to extract markdown headings
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class PyMuPdfEngine:
    """PDF extraction using pymupdf4llm for markdown and pymupdf for metadata."""

    name: str = "pymupdf"

    def extract(self, path: Path) -> PdfExtractionResult:
        # Extract markdown via pymupdf4llm
        raw_md: str = pymupdf4llm.to_markdown(str(path))

        # Open with pymupdf for metadata
        doc = pymupdf.open(str(path))
        num_pages = len(doc)

        # Detect repeated headers/footers using per-page text
        pages_text: list[str] = []
        total_images = 0
        for page in doc:
            pages_text.append(page.get_text())
            total_images += len(page.get_images())
        doc.close()

        repeated = detect_repeated_lines(pages_text)
        markdown_content = _postprocess_pdf_text(raw_md, repeated)
        plain_text = _postprocess_pdf_text("\n\n".join(pages_text), repeated)

        use_plain_text_fallback = _should_fallback_to_plain_text(markdown_content, plain_text)
        content = (
            _plain_text_to_markdown(plain_text)
            if use_plain_text_fallback
            else markdown_content
        )

        # Count tables from markdown output
        total_tables = 0 if use_plain_text_fallback else len(_TABLE_RE.findall(content))

        # Extract headings from markdown
        headings = [m.group(0) for m in _HEADING_RE.finditer(content)]

        return PdfExtractionResult(
            content=content,
            pages=num_pages,
            headings=headings,
            tables=total_tables,
            images=total_images,
        )


def _postprocess_pdf_text(text: str, repeated_lines: set[str]) -> str:
    """Apply shared cleanup to PDF-derived text."""
    content = normalize_ligatures(text)
    content = fix_drop_caps(content)
    content = strip_toc(content)
    if repeated_lines:
        content = strip_lines(content, repeated_lines)
    content = _strip_outer_code_fence(content)
    return content.strip()


def _should_fallback_to_plain_text(markdown_content: str, plain_text: str) -> bool:
    """Detect pathological markdown extraction and prefer plain text when it is clearly better."""
    if not markdown_content:
        return True
    if not plain_text:
        return False

    placeholder_columns = len(_PLACEHOLDER_COLUMN_RE.findall(markdown_content))
    table_lines = len(_TABLE_LINE_RE.findall(markdown_content))
    max_line_length = max((len(line) for line in markdown_content.splitlines()), default=0)
    markdown_tokens = count_tokens(markdown_content)
    plain_text_tokens = count_tokens(plain_text)
    markdown_heading_count = len(_HEADING_RE.findall(markdown_content))
    plain_text_heading_count = _count_plain_text_headings(plain_text)

    if placeholder_columns >= 2 and table_lines >= 4:
        return True
    if max_line_length >= 1500:
        return True
    if plain_text_tokens and markdown_tokens > plain_text_tokens * 2 and table_lines >= 4:
        return True
    if (
        markdown_heading_count == 0
        and plain_text_heading_count >= 3
        and plain_text_tokens
        and markdown_tokens <= plain_text_tokens * 1.25
    ):
        return True
    return False


def _plain_text_to_markdown(text: str) -> str:
    """Convert plain PDF text into lightweight markdown with inferred headings."""
    lines = text.splitlines()
    title = _infer_plain_text_title(lines)
    output: list[str] = []
    if title:
        output.extend([f"# {title}", ""])

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            output.append("")
            continue

        if title and stripped == title:
            continue

        numeric_match = _NUMERIC_HEADING_RE.match(stripped)
        if numeric_match:
            level = min(6, numeric_match.group("section").rstrip(".").count(".") + 2)
            output.append(f"{'#' * level} {stripped}")
            continue

        if _NAMED_HEADING_RE.fullmatch(stripped):
            output.append(f"## {stripped}")
            continue

        output.append(stripped)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def _infer_plain_text_title(lines: list[str]) -> str | None:
    """Infer a title from the first title-like line in plain text extraction."""
    best_title: str | None = None
    best_score = float("-inf")

    for line in lines[:40]:
        stripped = line.strip()
        if not stripped or _NUMERIC_HEADING_RE.match(stripped) or _NAMED_HEADING_RE.fullmatch(stripped):
            continue
        if stripped.endswith((".", ":", ";")):
            continue
        words = _TITLE_WORD_RE.findall(stripped)
        if not (3 <= len(words) <= 14 and 10 <= len(stripped) <= 140):
            continue

        score = 0
        if 3 <= len(words) <= 8:
            score += 2
        if re.search(r"\s{3,}", stripped):
            score -= 3
        if "@" in stripped or "http://" in stripped or "https://" in stripped:
            score -= 4
        if _TITLE_NOISE_RE.search(stripped):
            score -= 5

        titled_words = sum(
            1 for word in words if word[:1].isupper() or word.isupper()
        )
        lowercase_words = sum(1 for word in words if word.islower())
        if titled_words / len(words) >= 0.6:
            score += 3
        if lowercase_words / len(words) > 0.5:
            score -= 2
        if stripped.isupper():
            score += 1

        if score > best_score:
            best_score = score
            best_title = stripped

    return best_title


def _count_plain_text_headings(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if _NUMERIC_HEADING_RE.match(stripped) or _NAMED_HEADING_RE.fullmatch(stripped):
            count += 1
    return count


def _strip_outer_code_fence(text: str) -> str:
    match = _OUTER_CODE_FENCE_RE.match(text.strip())
    if not match:
        return text
    body = match.group("body")
    if body.count("```") > 0:
        return text
    return body
