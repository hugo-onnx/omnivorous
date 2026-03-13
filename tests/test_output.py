"""Unit tests for omnivorous.output helpers."""

from __future__ import annotations

from pathlib import Path

from omnivorous.output import unique_output_dir


def test_unique_output_dir_nonexistent(tmp_path: Path):
    base = tmp_path / "output"
    assert unique_output_dir(base) == base


def test_unique_output_dir_exists_once(tmp_path: Path):
    base = tmp_path / "output"
    base.mkdir()
    assert unique_output_dir(base) == tmp_path / "output-1"


def test_unique_output_dir_exists_multiple(tmp_path: Path):
    base = tmp_path / "output"
    base.mkdir()
    (tmp_path / "output-1").mkdir()
    (tmp_path / "output-2").mkdir()
    assert unique_output_dir(base) == tmp_path / "output-3"


def test_unique_output_dir_file_collision(tmp_path: Path):
    base = tmp_path / "output"
    base.write_text("I am a file")
    assert unique_output_dir(base) == tmp_path / "output-1"


def test_unique_output_dir_nested(tmp_path: Path):
    base = tmp_path / "foo" / "bar" / "output"
    base.mkdir(parents=True)
    assert unique_output_dir(base) == tmp_path / "foo" / "bar" / "output-1"
