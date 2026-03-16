"""Tests for pipeline worker selection helpers."""

from pathlib import Path
from unittest.mock import patch

from omnivorous.pipeline import resolve_worker_count


def test_resolve_worker_count_uses_cpu_parallelism():
    files = [Path("a.txt"), Path("b.txt"), Path("c.txt")]
    with patch("omnivorous.pipeline.os.cpu_count", return_value=8):
        assert resolve_worker_count(files) == 3


def test_resolve_worker_count_scientific_uses_multiple_processes():
    files = [Path("paper-a.pdf"), Path("paper-b.pdf")]
    with patch("omnivorous.pipeline.get_pdf_engine", return_value="marker"), \
         patch("omnivorous.pipeline.os.cpu_count", return_value=8):
        assert resolve_worker_count(files) == 2


def test_resolve_worker_count_scientific_caps_parallelism():
    files = [Path("paper-a.pdf"), Path("paper-b.pdf"), Path("paper-c.pdf"), Path("paper-d.pdf")]
    with patch("omnivorous.pipeline.get_pdf_engine", return_value="marker"), \
         patch("omnivorous.pipeline.os.cpu_count", return_value=16):
        assert resolve_worker_count(files) == 4


def test_resolve_worker_count_scientific_uses_pdf_count_not_total_file_count():
    files = [
        Path("paper-a.pdf"),
        Path("notes.md"),
        Path("appendix.txt"),
    ]
    with patch("omnivorous.pipeline.get_pdf_engine", return_value="marker"), \
         patch("omnivorous.pipeline.os.cpu_count", return_value=16):
        assert resolve_worker_count(files) == 1
