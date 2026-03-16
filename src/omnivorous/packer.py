"""Agent context pack generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from omnivorous.agents import AgentTarget, resolve_agents
from omnivorous.chunker import chunk_markdown
from omnivorous.frontmatter import add_frontmatter
from omnivorous.models import DocumentMetadata
from omnivorous.pipeline import (
    discover_source_files,
    iter_converted_documents,
    resolve_output_paths as pipeline_resolve_output_paths,
)
from omnivorous.references import (
    ReferenceMatch,
    ReferenceTarget,
    build_reference_index,
    extract_identifiers,
    extract_section_prefix,
    extract_symbols,
    resolve_references,
)
from omnivorous.relationships import RelationshipNode, build_relationships, extract_keywords
from omnivorous.registry import ensure_registry_loaded
from omnivorous.tokens import count_tokens

_ATX_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_SETEXT_HEADING_RE = re.compile(r"^(.+)\n([=-]{2,})\s*$", re.MULTILINE)


def _normalize_heading(heading: str) -> str:
    return heading.lstrip("#").strip()


def _heading_samples(headings: list[str], limit: int = 4) -> list[str]:
    return [_normalize_heading(heading) for heading in headings[:limit]]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _chunk_heading(chunk: str, fallback: str) -> str:
    atx = _ATX_HEADING_RE.search(chunk)
    if atx:
        return atx.group(1).strip()

    setext = _SETEXT_HEADING_RE.search(chunk)
    if setext:
        return setext.group(1).strip()

    for line in chunk.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return fallback


def _chunk_preview(chunk: str, fallback: str) -> str:
    for line in chunk.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        return stripped[:160]
    return fallback[:160]


def _coerce_document_entries(
    documents: list[DocumentMetadata] | list[dict[str, Any]],
    original_sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not documents:
        return []

    first = documents[0]
    if isinstance(first, dict):
        return [dict(doc) for doc in documents]

    entries: list[dict[str, Any]] = []
    for index, meta in enumerate(documents):
        assert isinstance(meta, DocumentMetadata)
        entry = meta.to_dict()
        entry["full_path"] = meta.source
        entry["chunk_count"] = 0
        entry["chunks"] = []
        entry["keywords"] = extract_keywords(
            meta.title,
            *[_normalize_heading(heading) for heading in meta.headings[:12]],
        )
        entry["heading_samples"] = _heading_samples(meta.headings)
        entry["related_documents"] = []
        if original_sources and index < len(original_sources):
            entry["original_source"] = original_sources[index]
        entries.append(entry)
    return entries


def _document_relationship_text(entry: dict[str, Any]) -> str:
    parts = [
        entry["title"],
        *entry.get("heading_samples", []),
        *entry.get("keywords", []),
    ]
    for chunk in entry.get("chunks", [])[:6]:
        parts.append(chunk["heading"])
        parts.append(chunk["preview"])
    return "\n".join(parts)


def _bridge_summaries(document_entries: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    bridges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for document in document_entries:
        for chunk in document.get("chunks", []):
            for related in chunk.get("related_chunks", []):
                pair = tuple(sorted((chunk["path"], related["target_path"])))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                bridges.append({
                    "left_path": chunk["path"],
                    "left_heading": chunk["heading"],
                    "right_path": related["target_path"],
                    "right_heading": related.get("target_heading"),
                    "score": related["score"],
                    "relationship_type": related["relationship_type"],
                    "shared_terms": related["evidence"].get("shared_terms", []),
                    "references": related["evidence"].get("references", []),
                })

    bridges.sort(
        key=lambda bridge: (-bridge["score"], bridge["left_path"], bridge["right_path"])
    )
    return bridges[:limit]


def _path_aliases(*paths: str) -> tuple[str, ...]:
    aliases: list[str] = []
    for path in paths:
        if not path:
            continue
        aliases.append(path)
        aliases.append(Path(path).name)
    return tuple(_dedupe_preserve_order(aliases))


def _document_target_lookup(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        document["full_path"]: {
            "path": document["full_path"],
            "kind": "document",
            "title": document["title"],
            "heading": None,
        }
        for document in documents
    }


def _chunk_target_lookup(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        chunk["path"]: {
            "path": chunk["path"],
            "kind": "chunk",
            "title": document["title"],
            "heading": chunk["heading"],
        }
        for document in documents
        for chunk in document["chunks"]
    }


def _fuse_relationship_sets(
    source_keys: list[str],
    lexical_relationships: dict[str, list[Any]],
    explicit_relationships: dict[str, dict[str, list[ReferenceMatch]]],
    target_lookup: dict[str, dict[str, Any]],
    *,
    limit: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    fused: dict[str, list[dict[str, Any]]] = {}

    for source_key in source_keys:
        merged: dict[str, dict[str, Any]] = {}

        for explicit in explicit_relationships.get(source_key, {}).values():
            if not explicit:
                continue
            target_key = explicit[0].target_key
            target = target_lookup[target_key]
            edge = merged.setdefault(
                target_key,
                {
                    "target_path": target["path"],
                    "target_kind": target["kind"],
                    "target_title": target["title"],
                    "target_heading": target["heading"],
                    "relationship_type": "explicit_reference",
                    "score": 0.0,
                    "signal_scores": {},
                    "evidence": {
                        "shared_terms": [],
                        "references": [],
                    },
                },
            )
            best_reference = max(match.score for match in explicit)
            edge["signal_scores"]["reference_match"] = max(
                best_reference,
                edge["signal_scores"].get("reference_match", 0.0),
            )
            for match in explicit:
                ref = {"type": match.signal_type, "value": match.matched_text}
                if ref not in edge["evidence"]["references"]:
                    edge["evidence"]["references"].append(ref)

        for lexical in lexical_relationships.get(source_key, []):
            target = target_lookup[lexical.target_key]
            edge = merged.setdefault(
                lexical.target_key,
                {
                    "target_path": target["path"],
                    "target_kind": target["kind"],
                    "target_title": target["title"],
                    "target_heading": target["heading"],
                    "relationship_type": "lexical_similarity",
                    "score": 0.0,
                    "signal_scores": {},
                    "evidence": {
                        "shared_terms": [],
                        "references": [],
                    },
                },
            )
            edge["signal_scores"]["lexical_similarity"] = lexical.score
            edge["evidence"]["shared_terms"] = lexical.shared_terms

        ranked: list[dict[str, Any]] = []
        for edge in merged.values():
            reference_score = edge["signal_scores"].get("reference_match", 0.0)
            lexical_score = edge["signal_scores"].get("lexical_similarity", 0.0)
            if reference_score and lexical_score:
                edge["relationship_type"] = "hybrid"
                edge["score"] = round(min(1.0, reference_score + lexical_score * 0.2), 3)
            elif reference_score:
                edge["relationship_type"] = "explicit_reference"
                edge["score"] = round(reference_score, 3)
            else:
                edge["relationship_type"] = "lexical_similarity"
                edge["score"] = round(lexical_score, 3)
            ranked.append(edge)

        ranked.sort(
            key=lambda edge: (
                -edge["score"],
                edge["target_kind"],
                edge["target_path"],
            )
        )
        fused[source_key] = ranked[:limit]

    return fused


def generate_agent_instructions(
    docs_metadata: list[DocumentMetadata] | list[dict[str, Any]],
    agent: AgentTarget,
) -> str:
    """Generate an instruction file for the given agent target."""
    document_entries = _coerce_document_entries(docs_metadata)

    lines = [
        "# Project Context",
        "",
        f"This pack was generated by omnivorous for {agent.display_name}.",
        "",
        "## Workflow",
        "",
        "1. Read `PROJECT_CONTEXT.md` first for the high-level document map.",
        "2. Use `manifest.json` to locate the most relevant document and chunk paths.",
        "3. Prefer files under `docs/chunks/` for focused context.",
        "4. Open `docs/full/` only when chunk context is insufficient or full layout matters.",
        "5. If a chunk is relevant, check adjacent chunks before making assumptions about missing context.",
        "",
        "## Available Documents",
        "",
    ]

    for entry in document_entries:
        line = (
            f"- **{entry['title']}** "
            f"({entry.get('chunk_count', 0)} chunks, ~{entry['tokens_estimate']} tokens)"
        )
        lines.append(line)

    lines.extend([
        "",
        "Use repo-relative file paths when you reference the documentation in your answer.",
    ])
    return "\n".join(lines) + "\n"


def generate_claude_md(docs_metadata: list[DocumentMetadata] | list[dict[str, Any]]) -> str:
    """Generate a CLAUDE.md file with agent instructions. Kept for backward compatibility."""
    from omnivorous.agents import AGENT_TARGETS

    return generate_agent_instructions(docs_metadata, AGENT_TARGETS["claude"])


def generate_project_context(docs_metadata: list[DocumentMetadata] | list[dict[str, Any]]) -> str:
    """Generate a PROJECT_CONTEXT.md summary."""
    document_entries = _coerce_document_entries(docs_metadata)
    total_chunks = sum(entry.get("chunk_count", 0) for entry in document_entries)
    bridges = _bridge_summaries(document_entries)

    lines = [
        "# Project Documentation Map",
        "",
        f"Total documents: {len(document_entries)}",
        f"Total chunks: {total_chunks}",
        f"Total estimated tokens: {sum(entry['tokens_estimate'] for entry in document_entries)}",
        "",
        "## How To Use This Pack",
        "",
        "Start with the most relevant chunk under `docs/chunks/`.",
        "Use `manifest.json` for machine-readable chunk metadata and related-document hints.",
        "Escalate to `docs/full/` only when you need broader context or exact original structure.",
        "",
        "## Documents",
        "",
    ]

    for entry in document_entries:
        lines.append(f"### {entry['title']}")
        if entry.get("original_source"):
            lines.append(f"- Original source: `{entry['original_source']}`")
        lines.append(f"- Full document: `{entry.get('full_path', entry['source'])}`")
        if entry.get("chunk_count"):
            lines.append(f"- Chunks: {entry['chunk_count']}")
            first_chunk = entry["chunks"][0]["path"]
            lines.append(f"- Start with: `{first_chunk}`")
        if entry.get("pages"):
            lines.append(f"- Pages: {entry['pages']}")
        lines.append(f"- Format: {entry['format']}")
        lines.append(f"- Tokens: {entry['tokens_estimate']}")
        if entry.get("heading_samples"):
            lines.append(f"- Key headings: {', '.join(entry['heading_samples'])}")
        if entry.get("keywords"):
            lines.append(f"- Keywords: {', '.join(entry['keywords'])}")
        if entry.get("related_documents"):
            related = ", ".join(
                f"{item['target_title']} [{item['relationship_type']}]"
                for item in entry["related_documents"]
            )
            lines.append(f"- Related: {related}")
        lines.append("")

    if bridges:
        lines.extend([
            "## Cross-Document Bridges",
            "",
        ])
        for bridge in bridges:
            evidence_parts: list[str] = []
            if bridge["shared_terms"]:
                evidence_parts.append(", ".join(bridge["shared_terms"]))
            if bridge["references"]:
                evidence_parts.append(
                    ", ".join(ref["value"] for ref in bridge["references"])
                )
            evidence = "; ".join(evidence_parts) or "no extra evidence"
            lines.append(
                f"- `{bridge['left_path']}` <-> `{bridge['right_path']}` "
                f"[{bridge['relationship_type']}] (score {bridge['score']}) via {evidence}"
            )
        lines.append("")

    return "\n".join(lines)


def generate_manifest(
    docs_metadata: list[DocumentMetadata] | list[dict[str, Any]],
    output_files: list[str],
    original_sources: list[str] | None = None,
    *,
    chunk_by: str = "heading",
    chunk_size: int = 500,
) -> str:
    """Generate a manifest.json for the agent context pack."""
    documents = _coerce_document_entries(docs_metadata, original_sources)
    manifest = {
        "version": "3.0",
        "generator": "omnivorous",
        "chunk_strategy": chunk_by,
        "chunk_size": chunk_size,
        "relationship_strategy": "hybrid_reference_tfidf",
        "documents": documents,
        "output_files": output_files,
        "total_tokens": sum(doc["tokens_estimate"] for doc in documents),
        "total_chunks": sum(doc.get("chunk_count", 0) for doc in documents),
    }
    return json.dumps(manifest, indent=2) + "\n"


def resolve_output_paths(
    source_files: list[Path], source_dir: Path
) -> dict[Path, Path]:
    """Backward-compatible wrapper for the shared output path resolver."""
    return pipeline_resolve_output_paths(source_files, source_dir)


def _chunk_relative_path(doc_relative_path: Path, index: int) -> str:
    return (Path("docs/chunks") / doc_relative_path.parent / f"{doc_relative_path.stem}_{index:03d}.md").as_posix()


def pack_context(
    source_dir: Path,
    output_dir: Path,
    agents: list[str] | None = None,
    *,
    chunk_size: int = 500,
    chunk_by: str = "heading",
) -> Path:
    """Orchestrate full pipeline: convert all docs and generate agent context pack."""
    ensure_registry_loaded()

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if chunk_by not in {"heading", "tokens"}:
        raise ValueError("chunk_by must be one of: heading, tokens")

    full_docs_dir = output_dir / "docs" / "full"
    full_docs_dir.mkdir(parents=True, exist_ok=True)

    source_files = discover_source_files(source_dir)
    output_map = resolve_output_paths(source_files, source_dir)

    document_entries: list[dict[str, Any] | None] = [None] * len(source_files)
    document_nodes: list[RelationshipNode] = []
    chunk_nodes: list[RelationshipNode] = []
    document_reference_targets: list[ReferenceTarget] = []
    chunk_reference_targets: list[ReferenceTarget] = []
    document_reference_texts: dict[str, str] = {}
    chunk_reference_texts: dict[str, str] = {}
    output_files: list[str] = []

    for index, source_file, result in iter_converted_documents(source_files):
        original_source = source_file.relative_to(source_dir).as_posix()
        out_rel = output_map[source_file]

        full_doc_path = f"docs/full/{out_rel.as_posix()}"
        result.metadata.source = full_doc_path
        full_doc_with_frontmatter = add_frontmatter(result.content, result.metadata.to_dict())

        full_out_path = full_docs_dir / out_rel
        full_out_path.parent.mkdir(parents=True, exist_ok=True)
        full_out_path.write_text(full_doc_with_frontmatter, encoding="utf-8")
        output_files.append(full_doc_path)

        chunk_result = chunk_markdown(
            result.content,
            result.metadata,
            strategy=chunk_by,
            chunk_size=chunk_size,
        )
        chunk_count = len(chunk_result.chunks)
        chunk_entries: list[dict[str, Any]] = []
        normalized_headings = [_normalize_heading(heading) for heading in result.metadata.headings]
        document_identifiers = extract_identifiers(
            "\n".join([result.metadata.title, *normalized_headings, result.content])
        )
        document_symbols = extract_symbols(
            "\n".join([result.metadata.title, *normalized_headings]),
            limit=16,
        )
        document_sections = _dedupe_preserve_order(
            [
                section
                for heading in normalized_headings
                if (section := extract_section_prefix(heading)) is not None
            ]
        )
        document_reference_texts[full_doc_path] = result.content

        for chunk_index, chunk_content in enumerate(chunk_result.chunks, 1):
            chunk_path = _chunk_relative_path(out_rel, chunk_index)
            chunk_heading = _chunk_heading(chunk_content, result.metadata.title)
            chunk_tokens = count_tokens(chunk_content)
            chunk_preview = _chunk_preview(chunk_content, chunk_heading)
            chunk_metadata = {
                "source": full_doc_path,
                "original_source": original_source,
                "title": result.metadata.title,
                "format": result.metadata.format,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "chunk_heading": chunk_heading,
                "tokens_estimate": chunk_tokens,
                "encoding": result.metadata.encoding,
            }
            chunk_out_path = output_dir / chunk_path
            chunk_out_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_out_path.write_text(
                add_frontmatter(chunk_content, chunk_metadata),
                encoding="utf-8",
            )
            output_files.append(chunk_path)

            chunk_entries.append({
                "path": chunk_path,
                "index": chunk_index,
                "tokens_estimate": chunk_tokens,
                "heading": chunk_heading,
                "preview": chunk_preview,
                "previous": (
                    _chunk_relative_path(out_rel, chunk_index - 1)
                    if chunk_index > 1
                    else None
                ),
                "next": (
                    _chunk_relative_path(out_rel, chunk_index + 1)
                    if chunk_index < chunk_count
                    else None
                ),
                "related_chunks": [],
            })
            chunk_nodes.append(
                RelationshipNode(
                    key=chunk_path,
                    label=chunk_heading,
                    path=chunk_path,
                    body="\n".join([
                        result.metadata.title,
                        chunk_heading,
                        chunk_preview,
                        chunk_content,
                    ]),
                    group=full_doc_path,
                )
            )
            chunk_reference_texts[chunk_path] = chunk_content
            chunk_sections = []
            section_prefix = extract_section_prefix(chunk_heading)
            if section_prefix is not None:
                chunk_sections.append(section_prefix)
            chunk_alias_inputs = [chunk_path]
            if chunk_index == 1:
                chunk_alias_inputs.extend([original_source, full_doc_path])
            chunk_reference_targets.append(
                ReferenceTarget(
                    key=chunk_path,
                    path=chunk_path,
                    kind="chunk",
                    label=chunk_heading,
                    group=full_doc_path,
                    path_aliases=_path_aliases(*chunk_alias_inputs),
                    identifiers=tuple(extract_identifiers(f"{chunk_heading}\n{chunk_content}")),
                    section_numbers=tuple(chunk_sections),
                    headings=(chunk_heading,),
                    symbols=tuple(extract_symbols(f"{chunk_heading}\n{chunk_content}", limit=12)),
                )
            )

        document_entries[index] = {
            **result.metadata.to_dict(),
            "original_source": original_source,
            "full_path": full_doc_path,
            "chunk_count": chunk_count,
            "chunks": chunk_entries,
            "keywords": extract_keywords(
                result.metadata.title,
                *[_normalize_heading(heading) for heading in result.metadata.headings[:12]],
            ),
            "heading_samples": _heading_samples(result.metadata.headings),
            "related_documents": [],
        }
        document_reference_targets.append(
            ReferenceTarget(
                key=full_doc_path,
                path=full_doc_path,
                kind="document",
                label=result.metadata.title,
                group=full_doc_path,
                path_aliases=_path_aliases(original_source, full_doc_path),
                identifiers=tuple(document_identifiers),
                section_numbers=tuple(document_sections),
                headings=tuple([result.metadata.title, *normalized_headings]),
                symbols=tuple(document_symbols),
            )
        )
        document_nodes.append(
            RelationshipNode(
                key=full_doc_path,
                label=result.metadata.title,
                path=full_doc_path,
                body=_document_relationship_text(document_entries[index]),
                group=full_doc_path,
            )
        )

    documents = [entry for entry in document_entries if entry is not None]
    document_lookup = {entry["full_path"]: entry for entry in documents}
    document_relationships = build_relationships(document_nodes, limit=3, min_score=0.06)
    document_reference_index = build_reference_index(document_reference_targets)
    document_explicit_relationships = {
        path: resolve_references(
            document_reference_texts[path],
            document_reference_index,
            source_key=path,
            source_group=path,
            different_group_only=True,
        )
        for path in document_lookup
    }
    document_edges = _fuse_relationship_sets(
        list(document_lookup),
        document_relationships,
        document_explicit_relationships,
        _document_target_lookup(documents),
    )
    for path, edges in document_edges.items():
        document_lookup[path]["related_documents"] = edges

    chunk_lookup = {
        chunk["path"]: (document, chunk)
        for document in documents
        for chunk in document["chunks"]
    }
    chunk_relationships = build_relationships(
        chunk_nodes,
        limit=3,
        min_score=0.05,
        require_different_group=True,
    )
    chunk_reference_index = build_reference_index(chunk_reference_targets)
    chunk_explicit_relationships = {
        path: resolve_references(
            chunk_reference_texts[path],
            chunk_reference_index,
            source_key=path,
            source_group=chunk_lookup[path][0]["full_path"],
            different_group_only=True,
        )
        for path in chunk_lookup
    }
    chunk_edges = _fuse_relationship_sets(
        list(chunk_lookup),
        chunk_relationships,
        chunk_explicit_relationships,
        _chunk_target_lookup(documents),
    )
    for path, edges in chunk_edges.items():
        _, chunk = chunk_lookup[path]
        chunk["related_chunks"] = edges

    resolved = resolve_agents(agents or ["claude"])
    for agent in resolved:
        instructions = generate_agent_instructions(documents, agent)
        agent_path = output_dir / agent.file_path
        agent_path.parent.mkdir(parents=True, exist_ok=True)
        agent_path.write_text(instructions, encoding="utf-8")
        output_files.append(agent.file_path)

    project_ctx = generate_project_context(documents)
    (output_dir / "PROJECT_CONTEXT.md").write_text(project_ctx, encoding="utf-8")
    output_files.append("PROJECT_CONTEXT.md")

    output_files.append("manifest.json")
    output_files = sorted(output_files)
    manifest = generate_manifest(
        documents,
        output_files,
        chunk_by=chunk_by,
        chunk_size=chunk_size,
    )
    (output_dir / "manifest.json").write_text(manifest, encoding="utf-8")

    return output_dir
