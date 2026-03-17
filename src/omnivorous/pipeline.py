"""Document conversion helpers with automatic parallelism."""

from __future__ import annotations

import os
from contextlib import ExitStack
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from omnivorous.converters.pdf import get_pdf_engine, set_pdf_engine
from omnivorous.frontmatter import add_frontmatter
from omnivorous.models import ConvertResult
from omnivorous.registry import ensure_registry_loaded, get_converter, supported_extensions
from omnivorous.tokens import get_encoding_name, set_encoding

IngestCallback = Callable[[Path, Path, ConvertResult], None]


def discover_source_files(source_dir: Path) -> list[Path]:
    """Return supported source files under `source_dir` in deterministic order."""
    ensure_registry_loaded()
    exts = set(supported_extensions())
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in exts
    )


def resolve_output_paths(
    source_files: Sequence[Path],
    source_dir: Path,
) -> dict[Path, Path]:
    """Map each source file to a unique relative output path under the output directory."""
    from collections import defaultdict

    initial: dict[Path, Path] = {}
    for file_path in source_files:
        initial[file_path] = file_path.relative_to(source_dir).with_suffix(".md")

    groups: dict[Path, list[Path]] = defaultdict(list)
    for src, out in initial.items():
        groups[out].append(src)

    resolved: dict[Path, Path] = {}
    for out_path, sources in groups.items():
        if len(sources) == 1:
            resolved[sources[0]] = out_path
            continue

        for src in sources:
            ext_tag = src.suffix.lstrip(".")
            new_name = f"{out_path.stem}_{ext_tag}.md"
            resolved[src] = out_path.parent / new_name

    return resolved


def _convert_path(path: Path) -> ConvertResult:
    """Convert a single file using the active in-process configuration."""
    ensure_registry_loaded()
    converter = get_converter(path.suffix.lower())
    return converter.convert(path)


def _convert_path_in_subprocess(
    path: Path,
    encoding_name: str,
    pdf_engine: str,
) -> ConvertResult:
    """Convert a single file after restoring process-local configuration."""
    set_encoding(encoding_name)
    set_pdf_engine(pdf_engine)
    return _convert_path(path)


def _has_scientific_pdfs(
    source_files: Sequence[Path],
    pdf_engine: str | None = None,
) -> bool:
    engine = pdf_engine or get_pdf_engine()
    return engine == "marker" and any(path.suffix.lower() == ".pdf" for path in source_files)


def _has_pdfs(source_files: Sequence[Path]) -> bool:
    return any(path.suffix.lower() == ".pdf" for path in source_files)


def _resolve_pdf_worker_count(source_files: Sequence[Path]) -> int:
    pdf_count = sum(1 for path in source_files if path.suffix.lower() == ".pdf")
    if pdf_count == 0:
        return 0

    cpu_count = os.cpu_count() or 1
    if _has_scientific_pdfs(source_files):
        # Scientific extraction uses separate processes and large model state.
        # Cap the automatic fan-out to keep memory usage bounded.
        return min(pdf_count, cpu_count, 4)

    return min(pdf_count, cpu_count)


def resolve_worker_count(source_files: Sequence[Path]) -> int:
    """Resolve the automatic worker count for the given file set."""
    file_count = len(source_files)
    if file_count == 0:
        return 0

    cpu_count = os.cpu_count() or 1

    if _has_scientific_pdfs(source_files):
        return _resolve_pdf_worker_count(source_files)

    return min(file_count, cpu_count)


def _iter_threaded_documents(
    source_files: Sequence[Path],
    worker_count: int,
) -> Iterator[tuple[int, Path, ConvertResult]]:
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(_convert_path, path): index
            for index, path in enumerate(source_files)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            yield index, source_files[index], future.result()


def _iter_process_documents(
    source_files: Sequence[Path],
    worker_count: int,
) -> Iterator[tuple[int, Path, ConvertResult]]:
    encoding_name = get_encoding_name()
    pdf_engine = get_pdf_engine()

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(
                _convert_path_in_subprocess,
                path,
                encoding_name,
                pdf_engine,
            ): index
            for index, path in enumerate(source_files)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            yield index, source_files[index], future.result()


def _iter_mixed_pdf_documents(
    source_files: Sequence[Path],
) -> Iterator[tuple[int, Path, ConvertResult]]:
    cpu_count = os.cpu_count() or 1
    pdf_worker_count = _resolve_pdf_worker_count(source_files)
    pdf_items = [
        (index, path) for index, path in enumerate(source_files)
        if path.suffix.lower() == ".pdf"
    ]
    other_items = [
        (index, path) for index, path in enumerate(source_files)
        if path.suffix.lower() != ".pdf"
    ]

    future_to_item: dict = {}
    with ExitStack() as stack:
        if _has_scientific_pdfs(source_files) and len(pdf_items) > 1:
            pdf_executor = stack.enter_context(
                ProcessPoolExecutor(max_workers=pdf_worker_count)
            )
            encoding_name = get_encoding_name()
            pdf_engine = get_pdf_engine()
            for index, path in pdf_items:
                future = pdf_executor.submit(
                    _convert_path_in_subprocess,
                    path,
                    encoding_name,
                    pdf_engine,
                )
                future_to_item[future] = (index, path)
        elif pdf_items:
            pdf_executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
            for index, path in pdf_items:
                future_to_item[pdf_executor.submit(_convert_path, path)] = (index, path)

        if other_items:
            other_executor = stack.enter_context(
                ThreadPoolExecutor(max_workers=min(len(other_items), cpu_count))
            )
            for index, path in other_items:
                future_to_item[other_executor.submit(_convert_path, path)] = (index, path)

        for future in as_completed(future_to_item):
            index, path = future_to_item[future]
            yield index, path, future.result()


def iter_converted_documents(
    source_files: Sequence[Path],
) -> Iterator[tuple[int, Path, ConvertResult]]:
    """Yield converted files as they complete."""
    worker_count = resolve_worker_count(source_files)
    if worker_count == 0:
        return

    if worker_count == 1:
        for index, path in enumerate(source_files):
            yield index, path, _convert_path(path)
        return

    # PDF conversion is isolated from the generic thread pool because PyMuPDF-based
    # extraction is not reliably deterministic when multiple PDFs are converted in
    # parallel threads. Fast mode serializes PDF work; scientific mode keeps its
    # separate-process fan-out because model-backed extraction is heavier.
    if _has_pdfs(source_files):
        yield from _iter_mixed_pdf_documents(source_files)
        return

    yield from _iter_threaded_documents(source_files, worker_count)


def ingest_documents(
    source_dir: Path,
    output_dir: Path,
    source_files: Sequence[Path] | None = None,
    on_document: IngestCallback | None = None,
) -> list[Path]:
    """Convert a source tree to markdown files under `output_dir`."""
    files = list(source_files) if source_files is not None else discover_source_files(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_map = resolve_output_paths(files, source_dir)

    for _, source_file, result in iter_converted_documents(files):
        md = add_frontmatter(result.content, result.metadata.to_dict())
        out_rel = output_map[source_file]
        out_path = output_dir / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        if on_document is not None:
            on_document(source_file, out_path, result)

    return [output_dir / output_map[file_path] for file_path in files]
