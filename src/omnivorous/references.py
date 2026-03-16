"""Deterministic explicit reference extraction and resolution."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

_PATH_RE = re.compile(
    r"(?<!\w)(?:\.{0,2}/)?(?:[\w.-]+/)*[\w.-]+\.(?:"
    r"md|markdown|txt|pdf|docx|html?|py|js|jsx|ts|tsx|json|ya?ml|toml|ini|cfg|csv|sql|ipynb"
    r")\b",
    re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,15}-\d{1,5}\b")
_SECTION_RE = re.compile(r"\b(?:section|sec\.?)\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)
_BACKTICK_RE = re.compile(r"`([^`\n]{2,120})`")
_SECTION_PREFIX_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\b")
_SYMBOL_RE = re.compile(
    r"\b(?:"
    r"[A-Z][a-zA-Z0-9]+"
    r"|[a-z_][a-z0-9_]{2,}"
    r"|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
    r")\b"
)


@dataclass(frozen=True)
class ReferenceTarget:
    """Canonical target that explicit references can resolve to."""

    key: str
    path: str
    kind: str
    label: str
    group: str | None = None
    path_aliases: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    section_numbers: tuple[str, ...] = ()
    headings: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceMatch:
    """Resolved explicit reference signal."""

    target_key: str
    target_path: str
    target_kind: str
    target_label: str
    signal_type: str
    score: float
    matched_text: str


@dataclass
class ReferenceIndex:
    """Lookup tables for explicit reference resolution."""

    targets: dict[str, ReferenceTarget] = field(default_factory=dict)
    paths: dict[str, list[str]] = field(default_factory=dict)
    identifiers: dict[str, list[str]] = field(default_factory=dict)
    sections: dict[str, list[str]] = field(default_factory=dict)
    headings: dict[str, list[str]] = field(default_factory=dict)
    symbols: dict[str, list[str]] = field(default_factory=dict)


def extract_identifiers(text: str) -> list[str]:
    """Extract uppercase identifier references like RFC-001."""
    return _dedupe(_IDENTIFIER_RE.findall(text.upper()))


def extract_section_numbers(text: str) -> list[str]:
    """Extract section references like section 3.2."""
    return _dedupe(match.group(1) for match in _SECTION_RE.finditer(text))


def extract_section_prefix(text: str) -> str | None:
    """Return a leading numeric section prefix from a heading if present."""
    match = _SECTION_PREFIX_RE.match(text)
    return match.group(1) if match else None


def extract_backticked_phrases(text: str) -> list[str]:
    """Extract inline code spans that may refer to headings or symbols."""
    return _dedupe(match.group(1).strip() for match in _BACKTICK_RE.finditer(text))


def extract_symbols(text: str, *, limit: int = 24) -> list[str]:
    """Extract code-like symbols from text."""
    backticked = []
    for phrase in extract_backticked_phrases(text):
        if " " not in phrase:
            backticked.append(phrase)

    symbols = list(backticked)
    for match in _SYMBOL_RE.finditer(text):
        symbol = match.group(0)
        if len(symbol) < 3:
            continue
        symbols.append(symbol)

    normalized = []
    for symbol in _dedupe(symbols):
        lowered = symbol.lower()
        if lowered in {"section", "table", "figure", "appendix"}:
            continue
        if "." not in symbol and "_" not in symbol and not any(ch.isupper() for ch in symbol[1:]):
            continue
        normalized.append(symbol)
        if len(normalized) >= limit:
            break
    return normalized


def extract_reference_candidates(text: str) -> dict[str, list[str]]:
    """Extract explicit reference candidates from arbitrary text."""
    paths = _dedupe(match.group(0) for match in _PATH_RE.finditer(text))
    identifiers = extract_identifiers(text)
    sections = extract_section_numbers(text)

    headings: list[str] = []
    symbols: list[str] = []
    for phrase in extract_backticked_phrases(text):
        if _PATH_RE.fullmatch(phrase):
            paths.append(phrase)
            continue
        if _IDENTIFIER_RE.fullmatch(phrase.upper()):
            identifiers.append(phrase.upper())
            continue
        if " " in phrase:
            headings.append(phrase)
        else:
            symbols.append(phrase)

    return {
        "path": _dedupe(paths),
        "identifier": _dedupe(identifiers),
        "section": sections,
        "heading": _dedupe(headings),
        "symbol": _dedupe(symbols + extract_symbols(text)),
    }


def build_reference_index(targets: list[ReferenceTarget]) -> ReferenceIndex:
    """Build lookup tables for explicit reference resolution."""
    index = ReferenceIndex()
    path_index: defaultdict[str, list[str]] = defaultdict(list)
    identifier_index: defaultdict[str, list[str]] = defaultdict(list)
    section_index: defaultdict[str, list[str]] = defaultdict(list)
    heading_index: defaultdict[str, list[str]] = defaultdict(list)
    symbol_index: defaultdict[str, list[str]] = defaultdict(list)

    for target in targets:
        index.targets[target.key] = target

        for alias in target.path_aliases:
            path_index[_normalize_path(alias)].append(target.key)
        for identifier in target.identifiers:
            identifier_index[_normalize_identifier(identifier)].append(target.key)
        for section in target.section_numbers:
            section_index[section].append(target.key)
        for heading in target.headings:
            normalized = _normalize_heading(heading)
            if normalized:
                heading_index[normalized].append(target.key)
        for symbol in target.symbols:
            normalized = _normalize_symbol(symbol)
            if normalized:
                symbol_index[normalized].append(target.key)

    index.paths = {key: _dedupe(values) for key, values in path_index.items()}
    index.identifiers = {key: _dedupe(values) for key, values in identifier_index.items()}
    index.sections = {key: _dedupe(values) for key, values in section_index.items()}
    index.headings = {key: _dedupe(values) for key, values in heading_index.items()}
    index.symbols = {key: _dedupe(values) for key, values in symbol_index.items()}
    return index


def resolve_references(
    text: str,
    index: ReferenceIndex,
    *,
    source_key: str,
    source_group: str | None = None,
    different_group_only: bool = True,
    limit: int = 5,
) -> dict[str, list[ReferenceMatch]]:
    """Resolve explicit references in text against a canonical target index."""
    candidates = extract_reference_candidates(text)
    matches: defaultdict[str, list[ReferenceMatch]] = defaultdict(list)

    for path in candidates["path"]:
        _append_matches(
            matches,
            _resolve_target_keys(index.paths, _normalize_path(path)),
            index,
            "path",
            1.0,
            path,
            source_key=source_key,
            source_group=source_group,
            different_group_only=different_group_only,
        )
    for identifier in candidates["identifier"]:
        _append_matches(
            matches,
            _resolve_target_keys(index.identifiers, _normalize_identifier(identifier)),
            index,
            "identifier",
            0.95,
            identifier,
            source_key=source_key,
            source_group=source_group,
            different_group_only=different_group_only,
        )
    for section in candidates["section"]:
        _append_matches(
            matches,
            _resolve_target_keys(index.sections, section),
            index,
            "section",
            0.9,
            section,
            source_key=source_key,
            source_group=source_group,
            different_group_only=different_group_only,
        )
    for heading in candidates["heading"]:
        _append_matches(
            matches,
            _resolve_target_keys(index.headings, _normalize_heading(heading)),
            index,
            "heading",
            0.84,
            heading,
            source_key=source_key,
            source_group=source_group,
            different_group_only=different_group_only,
        )
    for symbol in candidates["symbol"]:
        _append_matches(
            matches,
            _resolve_target_keys(index.symbols, _normalize_symbol(symbol)),
            index,
            "symbol",
            0.78,
            symbol,
            source_key=source_key,
            source_group=source_group,
            different_group_only=different_group_only,
        )

    resolved: dict[str, list[ReferenceMatch]] = {}
    for key, values in matches.items():
        values.sort(key=lambda match: (-match.score, match.signal_type, match.matched_text.lower()))
        resolved[key] = values[:limit]
    return resolved


def _append_matches(
    container: defaultdict[str, list[ReferenceMatch]],
    target_keys: list[str],
    index: ReferenceIndex,
    signal_type: str,
    score: float,
    matched_text: str,
    *,
    source_key: str,
    source_group: str | None,
    different_group_only: bool,
) -> None:
    for target_key in target_keys:
        target = index.targets[target_key]
        if target.key == source_key:
            continue
        if different_group_only and source_group is not None and target.group == source_group:
            continue
        container[target.key].append(
            ReferenceMatch(
                target_key=target.key,
                target_path=target.path,
                target_kind=target.kind,
                target_label=target.label,
                signal_type=signal_type,
                score=score,
                matched_text=matched_text,
            )
        )


def _resolve_target_keys(index: dict[str, list[str]], key: str) -> list[str]:
    if not key:
        return []
    return index.get(key, [])


def _normalize_path(path: str) -> str:
    return path.strip().lstrip("./").replace("\\", "/").lower()


def _normalize_identifier(identifier: str) -> str:
    return identifier.strip().upper()


def _normalize_heading(heading: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", heading.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().rstrip("()").replace("`", "")
    return normalized.lower()


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped
