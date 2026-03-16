"""Rich console output helpers. All output goes to stderr."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console(stderr=True)


def print_success(message: str) -> None:
    console.print(f"[green]{escape(message)}[/green]", soft_wrap=True)


def print_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {escape(message)}", soft_wrap=True)


def print_warning(message: str) -> None:
    console.print(f"[yellow]Warning:[/yellow] {escape(message)}", soft_wrap=True)


def print_info(message: str) -> None:
    console.print(f"[dim]{escape(message)}[/dim]", soft_wrap=True)


def get_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


def unique_output_dir(base: Path) -> Path:
    """Return *base* if it doesn't exist, otherwise append -1, -2, ... until free."""
    if not base.exists():
        return base
    parent = base.parent
    name = base.name
    counter = 1
    while True:
        candidate = parent / f"{name}-{counter}"
        if not candidate.exists():
            print_info(f"Directory '{base}' exists, using '{candidate}' instead")
            return candidate
        counter += 1
