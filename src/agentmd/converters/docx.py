"""DOCX converter using python-docx."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.table import Table

from agentmd.converters.base import BaseConverter
from agentmd.models import ConvertResult, DocumentMetadata
from agentmd.tokens import count_tokens, get_encoding_name

_HEADING_MAP = {
    "Heading 1": "#",
    "Heading 2": "##",
    "Heading 3": "###",
    "Heading 4": "####",
    "Heading 5": "#####",
    "Heading 6": "######",
}


def _table_to_markdown(table: Table) -> str:
    """Convert a docx Table to a markdown table."""
    rows: list[list[str]] = []
    for row in table.rows:
        rows.append([cell.text.strip() for cell in row.cells])
    if not rows:
        return ""

    lines: list[str] = []
    header = rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


class DocxConverter(BaseConverter):
    @property
    def name(self) -> str:
        return "DOCX"

    def convert(self, path: Path) -> ConvertResult:
        doc = Document(str(path))
        parts: list[str] = []
        headings: list[str] = []
        table_count = 0

        # Interleave paragraphs and tables in document order
        # python-docx exposes doc.element.body which contains all elements in order
        from docx.oxml.ns import qn

        for element in doc.element.body:
            if element.tag == qn("w:p"):
                # It's a paragraph
                from docx.text.paragraph import Paragraph

                para = Paragraph(element, doc)
                style_name = para.style.name if para.style else ""
                text = para.text.strip()
                if not text:
                    continue
                prefix = _HEADING_MAP.get(style_name, "")
                if prefix:
                    headings.append(text)
                    parts.append(f"{prefix} {text}")
                else:
                    parts.append(text)
            elif element.tag == qn("w:tbl"):
                table = Table(element, doc)
                md = _table_to_markdown(table)
                if md:
                    table_count += 1
                    parts.append(md)

        content = "\n\n".join(parts)
        metadata = DocumentMetadata(
            source=str(path),
            format="docx",
            title=path.stem,
            headings=headings,
            tables=table_count,
            tokens_estimate=count_tokens(content),
            encoding=get_encoding_name(),
        )
        return ConvertResult(content=content, metadata=metadata)
