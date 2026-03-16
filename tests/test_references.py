"""Tests for explicit reference extraction and resolution."""

from omnivorous.references import (
    ReferenceTarget,
    build_reference_index,
    extract_reference_candidates,
    resolve_references,
)


def test_extract_reference_candidates():
    candidates = extract_reference_candidates(
        "See `billing.txt`, RFC-101, section 2.1, and `PaymentClient` for details."
    )
    assert "billing.txt" in candidates["path"]
    assert "RFC-101" in candidates["identifier"]
    assert "2.1" in candidates["section"]
    assert "PaymentClient" in candidates["symbol"]


def test_resolve_references_matches_explicit_targets():
    index = build_reference_index(
        [
            ReferenceTarget(
                key="billing-doc",
                path="docs/full/billing.md",
                kind="document",
                label="Billing",
                group="docs/full/billing.md",
                path_aliases=("billing.txt", "docs/full/billing.md"),
                identifiers=("RFC-101",),
                section_numbers=("2.1",),
                headings=("Billing refunds",),
                symbols=("PaymentClient",),
            )
        ]
    )

    resolved = resolve_references(
        "See `billing.txt`, RFC-101, section 2.1, and `PaymentClient` for details.",
        index,
        source_key="overview-doc",
        source_group="docs/full/overview.md",
        different_group_only=True,
    )

    assert "billing-doc" in resolved
    signal_types = {match.signal_type for match in resolved["billing-doc"]}
    assert {"path", "identifier", "section", "symbol"} <= signal_types
