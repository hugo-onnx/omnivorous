"""HTML converter using BeautifulSoup + markdownify."""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify

from agentmd.converters.base import BaseConverter
from agentmd.models import ConvertResult, DocumentMetadata
from agentmd.tokens import count_tokens, get_encoding_name

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_RE = re.compile(r"^\|.+\|$", re.MULTILINE)


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

        # Count tables in HTML before conversion
        table_count = len(soup.find_all("table"))

        content = markdownify(html, heading_style="ATX", strip=["img", "script", "style"])
        # Clean up excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        headings = [m.group(2) for m in _HEADING_RE.finditer(content)]

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
