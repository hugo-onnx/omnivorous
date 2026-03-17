"""Tests for release-gate helpers."""

from __future__ import annotations

import json
from pathlib import Path

from omnivorous.packer import pack_context
from omnivorous.release_checks import (
    RetrievalCase,
    evaluate_retrieval,
    validate_manifest,
    validate_release_corpus,
)


def test_validate_manifest_accepts_valid_pack(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "pack"
    pack_context(fixtures_dir, out, agents=["codex"])

    manifest = json.loads((out / "manifest.json").read_text())
    assert validate_manifest(out, manifest) == []


def test_validate_release_corpus_reports_missing_source(tmp_path: Path):
    missing = tmp_path / "missing"
    assert validate_release_corpus(missing) == [f"missing_source_dir:{missing}"]


def test_validate_release_corpus_reports_missing_expected_documents(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "attention-transformer.md").write_text("# Attention\n\nTransformer architecture.")

    errors = validate_release_corpus(
        source,
        [
            RetrievalCase(
                case_id="attention",
                query="Where is the transformer architecture described?",
                expected_document="attention-transformer.md",
            ),
            RetrievalCase(
                case_id="missing",
                query="Where is the service standard?",
                expected_document="govuk-service-standard.html",
            ),
        ],
    )

    assert errors == ["missing_expected_document:govuk-service-standard.html"]


def test_retrieval_eval_finds_expected_documents_and_chunks(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "payments.md").write_text(
        "# Payments API\n\n"
        "## Refunds\n\n"
        "Refund invoice payments when retries fail.\n\n"
        "## Retries\n\n"
        "Retry failed invoice charges with backoff.\n"
    )
    (source / "billing.txt").write_text(
        "Billing runbook for invoice refunds and payment retries after failed charges."
    )
    (source / "hr.txt").write_text("Employee onboarding handbook and vacation policy.")

    out = tmp_path / "pack"
    pack_context(source, out, chunk_size=20, chunk_by="tokens")
    manifest = json.loads((out / "manifest.json").read_text())

    cases = [
        RetrievalCase(
            case_id="refunds",
            query="How do I refund invoice payments after retry failures?",
            expected_document="payments.md",
            expected_chunk_path="docs/chunks/payments_001.md",
        ),
        RetrievalCase(
            case_id="vacation-policy",
            query="Where is the vacation policy and onboarding guide?",
            expected_document="hr.txt",
        ),
    ]

    results = evaluate_retrieval(manifest, cases, output_dir=out)

    assert all(result.passed for result in results)


def test_retrieval_eval_ranks_hyphenated_identifiers_first(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "eu-ai-procurement-clauses.md").write_text(
        "# Standard contractual clauses\n\n"
        "Procurement clauses for artificial intelligence systems used by public organisations.\n"
    )
    (source / "overview.md").write_text(
        "# Overview\n\n"
        "General risk management system overview for enterprise delivery teams.\n"
    )

    out = tmp_path / "pack"
    pack_context(source, out, chunk_size=40, chunk_by="tokens")
    manifest = json.loads((out / "manifest.json").read_text())

    result = evaluate_retrieval(
        manifest,
        [
            RetrievalCase(
                case_id="procurement",
                query="Which AI procurement clauses document applies to public organisations?",
                expected_document="eu-ai-procurement-clauses.md",
                top_k=1,
            )
        ],
        output_dir=out,
    )[0]

    assert result.document_rank == 1
    assert result.passed is True


def test_retrieval_eval_uses_full_pack_text_for_late_matches(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    late_match = "\n\n".join(f"Paragraph {index}: background material." for index in range(18))
    (source / "caching-guide.txt").write_text(
        late_match
        + "\n\nValidation uses ETag headers and Cache-Control directives to reuse stored responses."
    )
    (source / "rules.md").write_text("# Rules\n\nGeneral rules for contributors and release management.")

    out = tmp_path / "pack"
    pack_context(source, out, chunk_size=15, chunk_by="tokens")
    manifest = json.loads((out / "manifest.json").read_text())

    result = evaluate_retrieval(
        manifest,
        [
            RetrievalCase(
                case_id="http-cache",
                query="Where are Cache-Control and ETag validation rules described?",
                expected_document="caching-guide.txt",
                top_k=1,
            )
        ],
        output_dir=out,
    )[0]

    assert result.document_rank == 1
    assert result.passed is True
