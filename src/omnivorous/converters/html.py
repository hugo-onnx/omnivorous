"""HTML converter using BeautifulSoup + markdownify."""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, FeatureNotFound, NavigableString, Tag
from markdownify import markdownify

from omnivorous.converters.base import BaseConverter
from omnivorous.models import ConvertResult, DocumentMetadata
from omnivorous.tokens import count_tokens, get_encoding_name

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_RE = re.compile(r"^\|.+\|$", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# Tags to remove entirely (content and all)
_REMOVE_TAGS = {"script", "style", "nav", "header", "footer", "noscript"}

# role attribute values that indicate non-content elements
_REMOVE_ROLES = {"navigation", "search", "banner", "contentinfo"}

# class substrings that indicate non-content elements
_REMOVE_CLASS_KEYWORDS = {"sidebar", "breadcrumb", "breadcrumbs", "headerlink", "permalink"}
_MIN_MEANINGFUL_SCOPE_CHARS = 200


def _normalize_quotes(text: str) -> str:
    return text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')


def _make_soup(html: str) -> BeautifulSoup:
    """Prefer the more robust lxml parser when available."""
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(html, "html.parser")


def _class_matches_keyword(token: str, keyword: str) -> bool:
    lowered = token.lower()
    if lowered == keyword:
        return True
    separators = ("-", "_")
    return any(
        lowered.startswith(f"{keyword}{separator}")
        or lowered.endswith(f"{separator}{keyword}")
        or f"{separator}{keyword}{separator}" in lowered
        for separator in separators
    )


def _is_non_content_class(classes: str) -> bool:
    tokens = classes.lower().split()
    for token in tokens:
        if any(_class_matches_keyword(token, keyword) for keyword in _REMOVE_CLASS_KEYWORDS):
            return True
    return False


def _text_length(tag: Tag | BeautifulSoup) -> int:
    return len(tag.get_text(" ", strip=True))


def _select_scope(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    """Prefer article-like content regions over the whole page shell."""
    candidates = [
        soup.find("article"),
        soup.find("main"),
        soup.find(attrs={"role": "main"}),
        soup.body,
        soup,
    ]
    ranked = [(candidate, _text_length(candidate)) for candidate in candidates if candidate is not None]
    for candidate, length in ranked:
        if length >= _MIN_MEANINGFUL_SCOPE_CHARS:
            return candidate
    return max(ranked, key=lambda item: item[1])[0] if ranked else soup


def _fallback_shell_content(soup: BeautifulSoup, title: str) -> str:
    """Return a minimal but informative fallback for JS-heavy or shell-like pages."""
    description = ""
    description_tag = soup.find("meta", attrs={"name": "description"})
    if description_tag and description_tag.get("content"):
        description = description_tag["content"].strip()

    lines = [f"# {title}"]
    if description:
        lines.extend(["", description])
    return "\n".join(lines).strip()


def _looks_like_shell_content(content: str) -> bool:
    """Return True when converted HTML still looks like a page shell."""
    lowered = content.lower()
    return "skip to main content" in lowered or "skip to search" in lowered


def _clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove non-content elements from the soup before markdown conversion."""
    # Remove unwanted tags
    for tag in list(soup.find_all(_REMOVE_TAGS)):
        tag.decompose()

    # Remove elements by role
    for tag in list(soup.find_all(attrs={"role": _REMOVE_ROLES})):
        if not isinstance(tag, Tag) or tag.attrs is None:
            continue
        tag.decompose()

    # Remove elements whose class contains known non-content keywords
    for tag in list(soup.find_all(class_=True)):
        if not isinstance(tag, Tag) or tag.attrs is None:
            continue
        class_attr = tag.attrs.get("class") or []
        classes = class_attr if isinstance(class_attr, str) else " ".join(class_attr)
        if _is_non_content_class(classes):
            tag.decompose()

    return soup


class HtmlConverter(BaseConverter):
    @property
    def name(self) -> str:
        return "HTML"

    def convert(self, path: Path) -> ConvertResult:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = _make_soup(html)

        title = ""
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()

        # Count tables in the content area before cleaning
        # Try to scope to main content first
        table_count = len(_select_scope(soup).find_all("table"))

        # Clean non-content elements and convert
        _clean_soup(soup)

        scope = _select_scope(soup)

        # Replace <img> tags with markdown image placeholders
        image_count = 0
        image_placeholders: dict[str, str] = {}
        for img in scope.find_all("img"):
            alt = img.get("alt") or img.get("title") or "image"
            token = f"OMNIVOROUSIMAGE{image_count}TOKEN"
            image_placeholders[token] = f"![{alt}]()"
            # Use plain-text sentinels and restore markdown after conversion so
            # markdownify does not escape the placeholder syntax.
            img.replace_with(NavigableString(f"\n\n{token}\n\n"))
            image_count += 1

        convert_html = str(scope)

        content = markdownify(convert_html, heading_style="ATX")
        for token, placeholder in image_placeholders.items():
            content = content.replace(token, placeholder)
        # Clean up excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        if count_tokens(content) < 80 and _looks_like_shell_content(content):
            content = _fallback_shell_content(soup, title or path.stem)

        headings = []
        for m in _HEADING_RE.finditer(content):
            prefix = m.group(1)
            text = _normalize_quotes(_MD_LINK_RE.sub(r"\1", m.group(2)))
            headings.append(f"{prefix} {text}")

        metadata = DocumentMetadata(
            source=path.name,
            format="html",
            title=title or path.stem,
            headings=headings,
            tables=table_count,
            images=image_count,
            tokens_estimate=count_tokens(content),
            encoding=get_encoding_name(),
        )
        return ConvertResult(content=content, metadata=metadata)
