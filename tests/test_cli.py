"""CLI integration tests using Typer's CliRunner."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from omnivorous.cli import app

runner = CliRunner()


def invoke(args: list[str]):
    return runner.invoke(app, args, prog_name="omni")


def test_help_shows_single_command_interface():
    result = invoke(["--help"])

    assert result.exit_code == 0
    assert "Usage: omni [OPTIONS] FOLDER" in result.output
    assert "Commands" not in result.output
    assert "--agent" in result.output
    assert "--chunk-size" in result.output
    assert "--semantic" in result.output


def test_bare_invocation_shows_help():
    result = invoke([])

    assert result.exit_code == 0
    assert "Usage: omni [OPTIONS] FOLDER" in result.output


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
    assert "Not a directory" in result.output


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

    assert result.exit_code == 1


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


def test_pack_with_semantic_mode_missing_backend(fixtures_dir: Path, tmp_path: Path):
    from unittest.mock import patch

    out = tmp_path / "agent-ctx"
    with patch(
        "omnivorous.embeddings.LocalEmbeddingService._resolve_backend",
        side_effect=ImportError(
            "Semantic mode requires local embedding support. "
            "Use `uv sync --extra semantic` or run once with "
            "`uv run --extra semantic omni <folder> ...`. "
            "With pip, install `omnivorous[semantic]`."
        ),
    ):
        result = invoke([str(fixtures_dir), "-o", str(out), "--semantic"])

    assert result.exit_code == 1
    assert "uv sync --extra semantic" in result.output
    assert "uv run --extra semantic omni <folder>" in result.output
    assert "omnivorous[semantic]" in result.output
