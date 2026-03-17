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
    fixture = repo_root / "tests" / "fixtures" / "notes.txt"

    if not fixture.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture}")

    with tempfile.TemporaryDirectory(prefix="omnivorous-package-smoke-") as tmp_dir:
        temp_root = Path(tmp_dir)
        venv_dir = temp_root / "venv"
        output_path = temp_root / "converted.md"

        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python = _venv_python(venv_dir)

        _run([python, "-m", "pip", "install", "--upgrade", "pip"])
        _run([python, "-m", "pip", "install", artifact])
        omni = _venv_omni(venv_dir)
        _run([omni, "--help"])
        _run([omni, "convert", fixture, "-o", output_path])

        content = output_path.read_text(encoding="utf-8")
        if "---" not in content or "This is a plain text document." not in content:
            raise RuntimeError("Converted smoke output did not contain the expected content")

    print(f"Package smoke passed for {artifact.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
