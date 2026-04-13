"""Install a built artifact into a fresh virtualenv and exercise the CLI."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import venv
from pathlib import Path


def _run(command: list[str | Path]) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"+ {printable}", flush=True)
    subprocess.run([str(part) for part in command], check=True)


def _resolve_artifact(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No build artifact matched {pattern!r} under {root}")
    return matches[-1]


def _venv_python(venv_dir: Path) -> Path:
    if (venv_dir / "Scripts" / "python.exe").exists():
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_omni(venv_dir: Path) -> Path:
    scripts_dir = venv_dir / ("Scripts" if (venv_dir / "Scripts").exists() else "bin")
    for candidate in ("omni.exe", "omni"):
        path = scripts_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find omni entrypoint in {scripts_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-glob", required=True, help="Glob for the built artifact to install.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    artifact = _resolve_artifact(args.artifact_glob, repo_root)
    fixture_dir = repo_root / "tests" / "fixtures"

    if not fixture_dir.exists():
        raise FileNotFoundError(f"Fixture directory not found: {fixture_dir}")

    with tempfile.TemporaryDirectory(prefix="omnivorous-package-smoke-") as tmp_dir:
        temp_root = Path(tmp_dir)
        venv_dir = temp_root / "venv"
        output_dir = temp_root / "agent-context"

        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python = _venv_python(venv_dir)

        _run([python, "-m", "pip", "install", "--upgrade", "pip"])
        _run([python, "-m", "pip", "install", artifact])
        omni = _venv_omni(venv_dir)
        _run([omni, "--help"])
        _run([omni, fixture_dir, "-o", output_dir])

        expected_paths = [
            output_dir / "CLAUDE.md",
            output_dir / "PROJECT_CONTEXT.md",
            output_dir / "manifest.json",
            output_dir / "docs" / "chunks",
            output_dir / "docs" / "full",
        ]
        missing = [path for path in expected_paths if not path.exists()]
        if missing:
            raise RuntimeError(f"Pack smoke output was missing expected files: {missing}")

    print(f"Package smoke passed for {artifact.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
