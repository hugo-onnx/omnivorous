"""Tests for pipeline worker selection helpers."""

from pathlib import Path
from unittest.mock import patch

from omnivorous.models import ConvertResult, DocumentMetadata
from omnivorous.pipeline import iter_converted_documents, resolve_worker_count


def _result(source: str) -> ConvertResult:
    return ConvertResult(
        content=source,
        metadata=DocumentMetadata(source=source, format="txt"),
    )


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


def test_iter_converted_documents_uses_mixed_mode_for_pdf_inputs():
    files = [Path("paper.pdf"), Path("notes.md")]
    expected = [(0, files[0], _result("paper.pdf"))]

    with patch("omnivorous.pipeline.resolve_worker_count", return_value=2), \
         patch("omnivorous.pipeline._iter_mixed_pdf_documents", return_value=iter(expected)) as mixed, \
         patch("omnivorous.pipeline._iter_threaded_documents") as threaded:
        assert list(iter_converted_documents(files)) == expected

    mixed.assert_called_once_with(files)
    threaded.assert_not_called()


def test_iter_converted_documents_uses_threads_for_non_pdf_inputs():
    files = [Path("readme.md"), Path("notes.txt")]
    expected = [(0, files[0], _result("readme.md"))]

    with patch("omnivorous.pipeline.resolve_worker_count", return_value=2), \
         patch("omnivorous.pipeline._iter_mixed_pdf_documents") as mixed, \
         patch("omnivorous.pipeline._iter_threaded_documents", return_value=iter(expected)) as threaded:
        assert list(iter_converted_documents(files)) == expected

    mixed.assert_not_called()
    threaded.assert_called_once_with(files, 2)
