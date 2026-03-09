"""YAML frontmatter utilities."""

from __future__ import annotations

import re

import yaml


def add_frontmatter(content: str, metadata: dict) -> str:
    """Prepend YAML frontmatter to markdown content."""
    # Filter out empty values
    filtered = {k: v for k, v in metadata.items() if v}
    if not filtered:
        return content
    frontmatter = yaml.dump(filtered, default_flow_style=False, sort_keys=False).strip()
    return f"---\n{frontmatter}\n---\n\n{content}"


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n\n?", re.DOTALL)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content. Returns (metadata, body)."""
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content
    raw = match.group(1)
    metadata = yaml.safe_load(raw) or {}
    body = content[match.end() :]
    return metadata, body
