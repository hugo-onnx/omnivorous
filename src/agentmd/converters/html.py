"""HTML converter using BeautifulSoup + markdownify."""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

from agentmd.converters.base import BaseConverter
from agentmd.models import ConvertResult, DocumentMetadata
from agentmd.tokens import count_tokens, get_encoding_name

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_RE = re.compile(r"^\|.+\|$", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# Tags to remove entirely (content and all)
_REMOVE_TAGS = {"script", "style", "nav", "header", "footer", "noscript"}

# role attribute values that indicate non-content elements
_REMOVE_ROLES = {"navigation", "search", "banner", "contentinfo"}

# class substrings that indicate non-content elements
_REMOVE_CLASS_KEYWORDS = {"sidebar", "breadcrumb", "headerlink", "permalink"}


def _normalize_quotes(text: str) -> str:
    return text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')


def _clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove non-content elements from the soup before markdown conversion."""
    # Remove unwanted tags
    for tag in soup.find_all(_REMOVE_TAGS):
        tag.decompose()

    # Remove elements by role
    for tag in soup.find_all(attrs={"role": _REMOVE_ROLES}):
        tag.decompose()

    # Remove elements whose class contains known non-content keywords
    for tag in soup.find_all(class_=True):
        if not isinstance(tag, Tag):
            continue
        classes = " ".join(tag.get("class", []))
        if any(kw in classes.lower() for kw in _REMOVE_CLASS_KEYWORDS):
            tag.decompose()

    return soup


class HtmlConverter(BaseConverter):
    @property
    def name(self) -> str:
        return "HTML"

    def convert(self, path: Path) -> ConvertResult:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        title = ""
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()

        # Count tables in the content area before cleaning
        # Try to scope to main content first
        main = soup.find("main") or soup.find(attrs={"role": "main"})
        table_count = len((main or soup).find_all("table"))

        # Clean non-content elements and convert
        _clean_soup(soup)

        # If a <main> or role="main" element exists, convert only that
        main = soup.find("main") or soup.find(attrs={"role": "main"})
        convert_html = str(main) if main else str(soup)

        content = markdownify(convert_html, heading_style="ATX", strip=["img"])
        # Clean up excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        headings = [
            _normalize_quotes(_MD_LINK_RE.sub(r"\1", m.group(2)))
            for m in _HEADING_RE.finditer(content)
        ]

        metadata = DocumentMetadata(
            source=str(path),
            format="html",
            title=title or path.stem,
            headings=headings,
            tables=table_count,
            tokens_estimate=count_tokens(content),
            encoding=get_encoding_name(),
        )
        return ConvertResult(content=content, metadata=metadata)
