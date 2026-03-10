"""CLI integration tests using Typer's CliRunner."""

from pathlib import Path

from typer.testing import CliRunner

from omnivorous.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "convert" in result.output
    assert "ingest" in result.output
    assert "inspect" in result.output
    assert "pack" in result.output


def test_convert_txt(fixtures_dir: Path):
    result = runner.invoke(app, ["convert", str(fixtures_dir / "notes.txt")])
    assert result.exit_code == 0
    assert "notes" in result.output


def test_convert_markdown(fixtures_dir: Path):
    result = runner.invoke(app, ["convert", str(fixtures_dir / "readme.md")])
    assert result.exit_code == 0
    assert "Sample Document" in result.output


def test_convert_html(fixtures_dir: Path):
    result = runner.invoke(app, ["convert", str(fixtures_dir / "web.html")])
    assert result.exit_code == 0


def test_convert_with_output(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "result.md"
    result = runner.invoke(
        app, ["convert", str(fixtures_dir / "readme.md"), "-o", str(out)]
    )
    assert result.exit_code == 0
    assert out.exists()
    content = out.read_text()
    assert "---" in content  # frontmatter
    assert "Sample Document" in content


def test_convert_missing_file():
    result = runner.invoke(app, ["convert", "nonexistent.pdf"])
    assert result.exit_code == 1


def test_convert_unsupported_format(tmp_path: Path):
    f = tmp_path / "test.xyz"
    f.write_text("content")
    result = runner.invoke(app, ["convert", str(f)])
    assert result.exit_code == 1


def test_ingest(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "output"
    result = runner.invoke(app, ["ingest", str(fixtures_dir), "-o", str(out)])
    assert result.exit_code == 0
    assert out.is_dir()
    assert any(out.iterdir())


def test_ingest_not_a_directory(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    result = runner.invoke(app, ["ingest", str(f)])
    assert result.exit_code == 1


def test_inspect_txt(fixtures_dir: Path):
    result = runner.invoke(app, ["inspect", str(fixtures_dir / "notes.txt")])
    assert result.exit_code == 0


def test_inspect_missing():
    result = runner.invoke(app, ["inspect", "nonexistent.txt"])
    assert result.exit_code == 1


def test_pack(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-ctx"
    result = runner.invoke(app, ["pack", str(fixtures_dir), "-o", str(out)])
    assert result.exit_code == 0
    assert (out / "CLAUDE.md").exists()
    assert (out / "PROJECT_CONTEXT.md").exists()
    assert (out / "manifest.json").exists()


def test_pack_with_agent_flag(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-ctx"
    result = runner.invoke(app, ["pack", str(fixtures_dir), "-o", str(out), "--agent", "codex"])
    assert result.exit_code == 0
    assert (out / "AGENTS.md").exists()
    assert not (out / "CLAUDE.md").exists()


def test_pack_with_multiple_agents(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-ctx"
    result = runner.invoke(
        app, ["pack", str(fixtures_dir), "-o", str(out), "--agent", "claude", "--agent", "codex"]
    )
    assert result.exit_code == 0
    assert (out / "CLAUDE.md").exists()
    assert (out / "AGENTS.md").exists()


def test_pack_with_all_agents(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-ctx"
    result = runner.invoke(app, ["pack", str(fixtures_dir), "-o", str(out), "--agent", "all"])
    assert result.exit_code == 0
    assert (out / "CLAUDE.md").exists()
    assert (out / "AGENTS.md").exists()
    assert (out / ".cursor" / "rules" / "omnivorous.md").exists()


def test_pack_with_invalid_agent(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "agent-ctx"
    result = runner.invoke(app, ["pack", str(fixtures_dir), "-o", str(out), "--agent", "invalid"])
    assert result.exit_code == 1
