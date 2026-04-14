"""Run the heavy release validation suite against a corpus."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

from omnivorous.converters.pdf import set_pdf_engine
from omnivorous.packer import pack_context
from omnivorous.pipeline import iter_converted_documents
from omnivorous.registry import ensure_registry_loaded, get_converter
from omnivorous.release_checks import (
    collect_manifest_metrics,
    evaluate_retrieval,
    load_manifest,
    load_retrieval_cases,
    validate_release_corpus,
    validate_manifest,
)
from omnivorous.tokens import set_encoding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Corpus directory to validate.")
    parser.add_argument(
        "--retrieval-cases",
        type=Path,
        help="Optional retrieval evaluation cases JSON.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Preserve temporary output directories for inspection.",
    )
    args = parser.parse_args()

    set_encoding("o200k_base")
    set_pdf_engine("pymupdf")

    retrieval_cases = load_retrieval_cases(args.retrieval_cases) if args.retrieval_cases else []
    source_errors = validate_release_corpus(args.source, retrieval_cases)
    if source_errors:
        print("Release gates failed before packing:")
        for error in source_errors:
            print(f"- {error}")
        return 1

    output_root = Path(tempfile.mkdtemp(prefix="omnivorous-release-gates-", dir="/tmp"))
    failures: list[str] = []

    try:
        heading_pack = output_root / "heading-pack"
        tokens_pack = output_root / "tokens-pack"

        heading_manifest = _run_pack(
            source_dir=args.source,
            output_dir=heading_pack,
            chunk_by="heading",
        )
        tokens_manifest = _run_pack(
            source_dir=args.source,
            output_dir=tokens_pack,
            chunk_by="tokens",
        )

        _report_manifest("heading", heading_pack, heading_manifest, failures)
        _report_manifest("tokens", tokens_pack, tokens_manifest, failures)

        determinism_failures = _check_pdf_determinism(args.source)
        if determinism_failures:
            failures.extend(determinism_failures)
            for failure in determinism_failures:
                print(f"[determinism] FAIL {failure}")
        else:
            print("[determinism] PASS sampled PDFs are stable between standalone and packed conversion")

        if retrieval_cases:
            results = evaluate_retrieval(heading_manifest, retrieval_cases, output_dir=heading_pack)
            for result in results:
                status = "PASS" if result.passed else "FAIL"
                print(
                    f"[retrieval] {status} {result.case_id} "
                    f"doc_rank={result.document_rank} chunk_rank={result.chunk_rank} "
                    f"top_docs={result.top_documents}"
                )
            failures.extend(
                f"retrieval:{result.case_id}"
                for result in results
                if not result.passed
            )

        if failures:
            print("\nRelease gates failed:")
            for failure in failures:
                print(f"- {failure}")
            if args.keep_output:
                print(f"Temporary outputs kept at {output_root}")
            return 1

        print("\nRelease gates passed.")
        if args.keep_output:
            print(f"Temporary outputs kept at {output_root}")
        return 0
    finally:
        if not args.keep_output and output_root.exists():
            shutil.rmtree(output_root)


def _run_pack(
    *,
    source_dir: Path,
    output_dir: Path,
    chunk_by: str,
    chunk_size: int = 500,
) -> dict[str, object]:
    start = time.perf_counter()
    pack_context(
        source_dir,
        output_dir,
        agents=["codex"],
        chunk_by=chunk_by,
        chunk_size=chunk_size,
    )
    elapsed = time.perf_counter() - start
    _, manifest = load_manifest(output_dir)
    manifest["_elapsed_seconds"] = round(elapsed, 3)
    return manifest


def _report_manifest(
    name: str,
    output_dir: Path,
    manifest: dict[str, object],
    failures: list[str],
) -> None:
    metrics = collect_manifest_metrics(manifest)
    errors = validate_manifest(output_dir, manifest)
    if manifest["relationship_strategy"] != "hybrid_reference_tfidf_embedding":
        errors.append("semantic_relationship_strategy")
    if errors:
        failures.extend(f"{name}:{item}" for item in errors)
    print(
        f"[{name}] seconds={manifest['_elapsed_seconds']} "
        f"docs={metrics['documents']} chunks={metrics['chunks']} "
        f"tokens={metrics['total_tokens']} p95_chunk_tokens={metrics['p95_chunk_tokens']} "
        f"errors={len(errors)}"
    )


def _check_pdf_determinism(source_dir: Path) -> list[str]:
    ensure_registry_loaded()
    converter = get_converter(".pdf")
    sample_files = [
        source_dir / name
        for name in (
            "attention-is-all-you-need.pdf",
            "eu-ai-act.pdf",
            "RGPD.pdf",
        )
        if (source_dir / name).exists()
    ]
    if not sample_files:
        sample_files = sorted(source_dir.rglob("*.pdf"))[:3]
    if not sample_files:
        return []

    standalone = {path.name: converter.convert(path) for path in sample_files}
    parallel = {path.name: result for _, path, result in iter_converted_documents(sample_files)}

    failures: list[str] = []
    for name in standalone:
        left = standalone[name]
        right = parallel[name]
        if left.content != right.content:
            failures.append(f"pdf_content_mismatch:{name}")
        if left.metadata.headings != right.metadata.headings:
            failures.append(f"pdf_heading_mismatch:{name}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
