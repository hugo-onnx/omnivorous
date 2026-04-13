"""Install a built artifact and exercise the CLI through supported installers."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def _run(command: list[str | Path], *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"+ {printable}", flush=True)
    subprocess.run([str(part) for part in command], check=True, env=env)


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
    return _omni_in_bin_dir(scripts_dir)


def _omni_in_bin_dir(bin_dir: Path) -> Path:
    for candidate in ("omni.exe", "omni"):
        path = bin_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find omni entrypoint in {bin_dir}")


def _verify_imports(python: Path) -> None:
    _run(
        [
            python,
            "-c",
            "import fastembed, marker, onnxruntime; "
            "print('bundled scientific and semantic dependencies are importable')",
        ]
    )


def _verify_pack_output(output_dir: Path) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-glob", required=True, help="Glob for the built artifact to install.")
    parser.add_argument(
        "--installer",
        choices=("pip", "pipx"),
        default="pip",
        help="Installer to validate against the built artifact.",
    )
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
        if args.installer == "pip":
            _run([python, "-m", "pip", "install", artifact])
            _verify_imports(python)
            omni = _venv_omni(venv_dir)
        else:
            pipx_home = temp_root / "pipx-home"
            pipx_bin_dir = temp_root / "pipx-bin"
            pipx_env = os.environ.copy()
            pipx_env["PIPX_HOME"] = str(pipx_home)
            pipx_env["PIPX_BIN_DIR"] = str(pipx_bin_dir)

            _run([python, "-m", "pip", "install", "pipx"], env=pipx_env)
            _run(
                [python, "-m", "pipx", "install", "--python", python, artifact],
                env=pipx_env,
            )
            pipx_python = _venv_python(pipx_home / "venvs" / "omnivorous")
            _verify_imports(pipx_python)
            omni = _omni_in_bin_dir(pipx_bin_dir)

        _run([omni, "--help"])
        _run([omni, fixture_dir, "-o", output_dir])
        _verify_pack_output(output_dir)

    print(f"Package smoke passed for {artifact.name} via {args.installer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
