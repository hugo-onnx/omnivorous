"""CLI commands for omnivorous."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from omnivorous.output import console, get_progress, print_error, print_info, print_success, unique_output_dir

app = typer.Typer(
    name="omni",
    help="Convert documents into agent-ready Markdown context.",
    no_args_is_help=True,
)


def _apply_encoding(encoding: str) -> None:
    """Validate and set the tiktoken encoding, exiting on error."""
    from omnivorous.tokens import set_encoding

    try:
        set_encoding(encoding)
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(1)


@app.command()
def convert(
    file: Path = typer.Argument(..., help="Path to the document to convert."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file path."),
    encoding: str = typer.Option("o200k_base", "--encoding", help="Tiktoken encoding name."),
) -> None:
    """Convert a single document to Markdown."""
    from omnivorous.frontmatter import add_frontmatter
    from omnivorous.registry import ensure_registry_loaded, get_converter

    _apply_encoding(encoding)
    ensure_registry_loaded()

    if not file.exists():
        print_error(f"File not found: {file}")
        raise typer.Exit(1)

    ext = file.suffix.lower()
    try:
        converter = get_converter(ext)
    except ValueError:
        print_error(f"Unsupported format: {ext}")
        raise typer.Exit(1)

    with get_progress() as progress:
        progress.add_task(f"Converting {file.name}...", total=None)
        result = converter.convert(file)

    md = add_frontmatter(result.content, result.metadata.to_dict())

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(md, encoding="utf-8")
        print_success(f"Converted {file.name} → {output}")
    else:
        typer.echo(md)

    print_info(f"{result.metadata.tokens_estimate} tokens")


@app.command()
def ingest(
    folder: Path = typer.Argument(..., help="Folder containing documents to convert."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory."),
    encoding: str = typer.Option("o200k_base", "--encoding", help="Tiktoken encoding name."),
) -> None:
    """Scan a folder and convert all supported documents to Markdown."""
    from omnivorous.frontmatter import add_frontmatter
    from omnivorous.registry import ensure_registry_loaded, get_converter, supported_extensions

    _apply_encoding(encoding)
    ensure_registry_loaded()

    if not folder.is_dir():
        print_error(f"Not a directory: {folder}")
        raise typer.Exit(1)

    out_dir = unique_output_dir(output or Path("output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = set(supported_extensions())
    files = sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in exts)

    if not files:
        print_error(f"No supported files found in {folder}")
        raise typer.Exit(1)

    from omnivorous.packer import resolve_output_paths

    output_map = resolve_output_paths(files, folder)

    print_info(f"Found {len(files)} file(s)")

    with get_progress() as progress:
        task = progress.add_task("Converting...", total=len(files))
        for f in files:
            converter = get_converter(f.suffix.lower())
            result = converter.convert(f)
            md = add_frontmatter(result.content, result.metadata.to_dict())
            out_rel = output_map[f]
            out_path = out_dir / out_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md, encoding="utf-8")
            progress.update(task, advance=1)

    print_success(f"Converted {len(files)} file(s) → {out_dir}/")


@app.command()
def inspect(
    file: Path = typer.Argument(..., help="File to inspect."),
    encoding: str = typer.Option("o200k_base", "--encoding", help="Tiktoken encoding name."),
) -> None:
    """Display metadata for a document."""
    from omnivorous.inspector import inspect_file

    _apply_encoding(encoding)

    if not file.exists():
        print_error(f"File not found: {file}")
        raise typer.Exit(1)

    with get_progress() as progress:
        progress.add_task(f"Inspecting {file.name}...", total=None)
        meta = inspect_file(file)

    table = Table(title=f"Document: {file.name}")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Source", meta.source)
    table.add_row("Format", meta.format)
    table.add_row("Title", meta.title)
    table.add_row("Pages", str(meta.pages) if meta.pages else "N/A")
    table.add_row("Headings", str(len(meta.headings)))
    table.add_row("Tables", str(meta.tables))
    table.add_row("Images", str(meta.images))
    table.add_row("Tokens (est.)", str(meta.tokens_estimate))
    table.add_row("Encoding", meta.encoding or "N/A")

    console.print(table)

    if meta.headings:
        headings_table = Table(title="Headings")
        headings_table.add_column("#", style="dim")
        headings_table.add_column("Heading")
        for i, h in enumerate(meta.headings, 1):
            # Strip prefix for display, show indentation for hierarchy
            stripped = h.lstrip("#").strip()
            level = len(h) - len(h.lstrip("#"))
            indent = "  " * max(0, level - 1)
            headings_table.add_row(str(i), f"{indent}{stripped}")
        console.print(headings_table)


@app.command()
def pack(
    folder: Path = typer.Argument(..., help="Folder containing documents to pack."),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output directory for agent context."
    ),
    encoding: str = typer.Option("o200k_base", "--encoding", help="Tiktoken encoding name."),
    agent: Optional[list[str]] = typer.Option(
        None, "--agent", "-a", help="Target agent(s): claude, codex, cursor, copilot, antigravity, or all."
    ),
) -> None:
    """Generate an agent context pack from a folder of documents."""
    from omnivorous.agents import resolve_agents
    from omnivorous.packer import pack_context

    _apply_encoding(encoding)

    if not folder.is_dir():
        print_error(f"Not a directory: {folder}")
        raise typer.Exit(1)

    agent_names = agent or ["claude"]
    try:
        resolved = resolve_agents(agent_names)
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    out_dir = unique_output_dir(output or Path("agent-context"))

    with get_progress() as progress:
        progress.add_task("Packing agent context...", total=None)
        try:
            pack_context(folder, out_dir, agents=agent_names)
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1)

    print_success(f"Agent context pack created in {out_dir}/")
    for a in resolved:
        print_info(f"  {a.file_path} — {a.display_name} instructions")
    print_info("  PROJECT_CONTEXT.md — documentation summary")
    print_info("  manifest.json — file manifest")
    print_info("  docs/ — converted documents")
