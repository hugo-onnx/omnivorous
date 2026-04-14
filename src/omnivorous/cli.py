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
    agent: list[str] | None,
    mode: str,
    chunk_size: int,
    chunk_by: str,
) -> None:
    from omnivorous.agents import resolve_agents
    from omnivorous.packer import pack_context

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
) -> None:
    """Generate an agent context pack from a folder of documents."""
    _run_pack(
        folder=folder,
        output=output,
        agent=agent,
        mode=mode,
        chunk_size=chunk_size,
        chunk_by=chunk_by,
    )
