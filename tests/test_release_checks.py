"""Tests for release-gate helpers."""

from __future__ import annotations

import json
from pathlib import Path

from omnivorous.packer import pack_context
from omnivorous.release_checks import RetrievalCase, evaluate_retrieval, validate_manifest


def test_validate_manifest_accepts_valid_pack(fixtures_dir: Path, tmp_path: Path):
    out = tmp_path / "pack"
    pack_context(fixtures_dir, out, agents=["codex"])

    manifest = json.loads((out / "manifest.json").read_text())
    assert validate_manifest(out, manifest) == []


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

    results = evaluate_retrieval(manifest, cases)

    assert all(result.passed for result in results)
