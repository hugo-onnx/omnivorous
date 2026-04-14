"""CLI integration tests using Typer's CliRunner."""

from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

from omnivorous.cli import app

runner = CliRunner()


def invoke(args: list[str]):
    return runner.invoke(app, args, prog_name="omni")


def rendered_output(result) -> str:
    return click.unstyle(result.output)


def test_help_shows_single_command_interface():
    result = invoke(["--help"])
    output = rendered_output(result)

    assert result.exit_code == 0
    assert "Usage: omni [OPTIONS] FOLDER" in output
    assert "Commands" not in output
    assert "--agent" in output
    assert "--chunk-size" in output
    assert "--encoding" not in output


def test_bare_invocation_shows_help():
    result = invoke([])
    output = rendered_output(result)

    assert result.exit_code == 0
    assert "Usage: omni [OPTIONS] FOLDER" in output


def test_pack(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-ctx"
    result = invoke([str(fixtures_dir), "-o", str(out)])

    assert result.exit_code == 0
    assert (out / "CLAUDE.md").exists()
    assert (out / "PROJECT_CONTEXT.md").exists()
    assert (out / "manifest.json").exists()
    assert (out / "docs" / "chunks").is_dir()
    assert (out / "docs" / "full").is_dir()


def test_pack_with_agent_flag(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-ctx"
    result = invoke([str(fixtures_dir), "-o", str(out), "--agent", "codex"])

    assert result.exit_code == 0
    assert (out / "AGENTS.md").exists()
    assert not (out / "CLAUDE.md").exists()


def test_pack_with_multiple_agents(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-ctx"
    result = invoke([str(fixtures_dir), "-o", str(out), "--agent", "claude", "--agent", "codex"])

    assert result.exit_code == 0
    assert (out / "CLAUDE.md").exists()
    assert (out / "AGENTS.md").exists()


def test_pack_with_all_agents(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-ctx"
    result = invoke([str(fixtures_dir), "-o", str(out), "--agent", "all"])

    assert result.exit_code == 0
    assert (out / "CLAUDE.md").exists()
    assert (out / "AGENTS.md").exists()
    assert (out / ".cursor" / "rules" / "omnivorous.md").exists()


def test_pack_with_invalid_agent(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-ctx"
    result = invoke([str(fixtures_dir), "-o", str(out), "--agent", "invalid"])

    assert result.exit_code == 1


def test_non_directory_input(fixtures_dir: Path):
    result = invoke([str(fixtures_dir / "notes.txt")])

    assert result.exit_code == 1
    assert "Not a directory" in rendered_output(result)


@pytest.mark.parametrize(
    "args",
    [
        ["pack", "docs"],
        ["convert", "document.pdf"],
        ["ingest", "docs"],
        ["inspect", "document.pdf"],
        ["warm-embeddings"],
    ],
)
def test_removed_legacy_invocations_fail(args: list[str]):
    result = invoke(args)

    assert result.exit_code != 0


def test_pack_auto_increments_existing_output(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-context"
    out.mkdir()

    result = invoke([str(fixtures_dir), "-o", str(out)])

    assert result.exit_code == 0
    incremented = tmp_path / "agent-context-1"
    assert incremented.is_dir()
    assert (incremented / "CLAUDE.md").exists()
    assert (incremented / "docs" / "chunks").is_dir()


def test_pack_with_mode_fast(fixtures_dir: Path):
    result = invoke([str(fixtures_dir), "--mode", "fast"])

    assert result.exit_code == 0


def test_pack_with_invalid_mode(fixtures_dir: Path):
    result = invoke([str(fixtures_dir), "--mode", "invalid"])

    assert result.exit_code == 1


def test_pack_with_scientific_mode_no_marker(fixtures_dir: Path):
    """Scientific mode should fail gracefully when marker-pdf is not installed."""
    from unittest.mock import patch

    with patch.dict("sys.modules", {"marker": None}):
        result = invoke([str(fixtures_dir), "--mode", "scientific"])
    output = rendered_output(result)

    assert result.exit_code == 1
    assert "Scientific mode is unavailable because the omnivorous installation is incomplete." in output


def test_pack_with_chunk_options(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-ctx"
    result = invoke(
        [
            str(fixtures_dir),
            "-o",
            str(out),
            "--chunk-size",
            "40",
            "--chunk-by",
            "tokens",
        ]
    )

    assert result.exit_code == 0
    manifest = (out / "manifest.json").read_text()
    assert '"chunk_strategy": "tokens"' in manifest
    assert '"chunk_size": 40' in manifest


def test_pack_with_invalid_chunk_strategy(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-ctx"
    result = invoke([str(fixtures_dir), "-o", str(out), "--chunk-by", "invalid"])

    assert result.exit_code == 1


def test_pack_fails_when_semantic_model_cannot_load(fixtures_dir: Path, tmp_path: Path):
    from unittest.mock import patch

    out = tmp_path / "agent-ctx"
    with patch(
        "omnivorous.embeddings.LocalEmbeddingService._resolve_backend",
        side_effect=ImportError(
            "Omnivorous could not load its local semantic model. "
            "The first successful `omni <folder>` run needs network access to download the model once; "
            "later runs reuse the local cache."
        ),
    ):
        result = invoke([str(fixtures_dir), "-o", str(out)])
    output = rendered_output(result)

    assert result.exit_code == 1
    assert "Omnivorous could not load its local semantic model." in output
    assert "first successful" in output
