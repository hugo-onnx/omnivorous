"""CLI entrypoint for omnivorous."""

from __future__ import annotations

from pathlib import Path

import click
import typer
from typer.core import TyperCommand

from omnivorous.output import get_progress, print_error, print_info, print_success, unique_output_dir

app = typer.Typer(
    name="omni",
    help="Generate agent-ready Markdown context packs from documents.",
)


class HelpOnEmptyCommand(TyperCommand):
    """Return help with exit code 0 when the root command receives no arguments."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not args and self.no_args_is_help and not ctx.resilient_parsing:
            click.echo(ctx.get_help(), color=ctx.color)
            ctx.exit()
        return super().parse_args(ctx, args)


def _apply_encoding(encoding: str) -> None:
    """Validate and set the tiktoken encoding, exiting on error."""
    from omnivorous.tokens import set_encoding

    try:
        set_encoding(encoding)
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(1)


def _apply_mode(mode: str) -> None:
    """Validate and set the PDF engine based on conversion mode."""
    from omnivorous.converters.pdf import set_pdf_engine

    mode_to_engine = {"fast": "pymupdf", "scientific": "marker"}
    if mode not in mode_to_engine:
        print_error(f"Unknown mode: {mode!r}. Valid: fast, scientific")
        raise typer.Exit(1)

    engine = mode_to_engine[mode]
    if engine == "marker":
        try:
            import marker  # noqa: F401
        except ImportError:
            print_error(
                "Scientific mode is unavailable because the omnivorous installation is incomplete. "
                "Reinstall omnivorous and try again."
            )
            raise typer.Exit(1)

    try:
        set_pdf_engine(engine)
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(1)


def _run_pack(
    *,
    folder: Path,
    output: Path | None,
    encoding: str,
    agent: list[str] | None,
    mode: str,
    chunk_size: int,
    chunk_by: str,
    semantic: bool,
    embedding_backend: str,
    embedding_model: str | None,
    embedding_cache_dir: Path | None,
    embedding_model_cache_dir: Path | None,
    semantic_offline: bool,
) -> None:
    from omnivorous.agents import resolve_agents
    from omnivorous.packer import pack_context

    _apply_encoding(encoding)
    _apply_mode(mode)

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
            pack_context(
                folder,
                out_dir,
                agents=agent_names,
                chunk_size=chunk_size,
                chunk_by=chunk_by,
                enable_semantic=semantic,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
                embedding_cache_dir=embedding_cache_dir,
                embedding_model_cache_dir=embedding_model_cache_dir,
                semantic_offline=semantic_offline,
            )
        except (ImportError, ValueError) as exc:
            print_error(str(exc))
            raise typer.Exit(1)

    print_success(f"Agent context pack created in {out_dir}/")
    for resolved_agent in resolved:
        print_info(f"  {resolved_agent.file_path} — {resolved_agent.display_name} instructions")
    print_info("  PROJECT_CONTEXT.md — documentation summary")
    print_info("  manifest.json — file manifest")
    print_info("  docs/chunks/ — chunked documents for focused reading")
    print_info("  docs/full/ — full converted documents")


@app.command(cls=HelpOnEmptyCommand, no_args_is_help=True)
def main(
    folder: Path = typer.Argument(..., help="Folder containing documents to pack."),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output directory for agent context.",
        rich_help_panel="Basic",
    ),
    agent: list[str] | None = typer.Option(
        None,
        "--agent",
        "-a",
        help="Target agent(s): claude, codex, cursor, copilot, antigravity, or all.",
        rich_help_panel="Basic",
    ),
    mode: str = typer.Option(
        "fast",
        "--mode",
        "-m",
        help="PDF mode: fast (default) or scientific (LaTeX formulas).",
        rich_help_panel="Basic",
    ),
    chunk_size: int = typer.Option(
        500,
        "--chunk-size",
        min=1,
        help="Target chunk size in tokens.",
        rich_help_panel="Chunking",
    ),
    chunk_by: str = typer.Option(
        "heading",
        "--chunk-by",
        help="Chunking strategy: heading or tokens.",
        rich_help_panel="Chunking",
    ),
    semantic: bool = typer.Option(
        False,
        "--semantic",
        help="Enable optional local-embedding relationships.",
        rich_help_panel="Semantic",
    ),
    embedding_backend: str = typer.Option(
        "fastembed",
        "--embedding-backend",
        help="Local embedding backend to use when --semantic is enabled.",
        rich_help_panel="Semantic",
    ),
    embedding_model: str | None = typer.Option(
        None,
        "--embedding-model",
        help="Optional local embedding model name.",
        rich_help_panel="Semantic",
    ),
    embedding_cache_dir: Path | None = typer.Option(
        None,
        "--embedding-cache-dir",
        help="Optional cache directory for local embeddings.",
        rich_help_panel="Semantic",
    ),
    embedding_model_cache_dir: Path | None = typer.Option(
        None,
        "--embedding-model-cache-dir",
        help="Optional cache directory for local embedding model files.",
        rich_help_panel="Semantic",
    ),
    semantic_offline: bool = typer.Option(
        False,
        "--semantic-offline",
        help="Require semantic mode to use pre-cached local model files only.",
        rich_help_panel="Semantic",
    ),
    encoding: str = typer.Option(
        "o200k_base",
        "--encoding",
        help="Tiktoken encoding name.",
        rich_help_panel="Advanced",
    ),
) -> None:
    """Generate an agent context pack from a folder of documents."""
    _run_pack(
        folder=folder,
        output=output,
        encoding=encoding,
        agent=agent,
        mode=mode,
        chunk_size=chunk_size,
        chunk_by=chunk_by,
        semantic=semantic,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_cache_dir=embedding_cache_dir,
        embedding_model_cache_dir=embedding_model_cache_dir,
        semantic_offline=semantic_offline,
    )
