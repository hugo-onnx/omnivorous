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


def test_extract_reference_candidates_ignores_weak_identifiers():
    candidates = extract_reference_candidates(
        "Map DE.AE-03 and AE-04, but keep RFC-101."
    )

    assert "RFC-101" in candidates["identifier"]
    assert "AE-03" not in candidates["identifier"]
    assert "AE-04" not in candidates["identifier"]


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


def test_extract_reference_candidates_ignores_generic_symbols():
    candidates = extract_reference_candidates(
        "Read `THE`, `III`, `e.g`, `i.e`, and `PaymentClient` before section 4."
    )

    assert "PaymentClient" in candidates["symbol"]
    assert "THE" not in candidates["symbol"]
    assert "III" not in candidates["symbol"]
    assert "e.g" not in candidates["symbol"]
    assert "i.e" not in candidates["symbol"]


def test_resolve_references_skips_ambiguous_or_low_confidence_sections():
    index = build_reference_index(
        [
            ReferenceTarget(
                key="doc-a",
                path="docs/full/a.md",
                kind="document",
                label="Doc A",
                group="docs/full/a.md",
                section_numbers=("2.1", "4"),
            ),
            ReferenceTarget(
                key="doc-b",
                path="docs/full/b.md",
                kind="document",
                label="Doc B",
                group="docs/full/b.md",
                section_numbers=("2.1", "4"),
            ),
        ]
    )

    resolved = resolve_references(
        "Compare section 2.1 and section 4 before shipping.",
        index,
        source_key="overview-doc",
        source_group="docs/full/overview.md",
        different_group_only=True,
    )

    assert resolved == {}


def test_resolve_references_skips_weak_identifiers():
    index = build_reference_index(
        [
            ReferenceTarget(
                key="weak-doc",
                path="docs/full/weak.md",
                kind="document",
                label="Weak",
                group="docs/full/weak.md",
                identifiers=("AE-03",),
            ),
            ReferenceTarget(
                key="strong-doc",
                path="docs/full/strong.md",
                kind="document",
                label="Strong",
                group="docs/full/strong.md",
                identifiers=("RFC-101",),
            ),
        ]
    )

    resolved = resolve_references(
        "See AE-03 and RFC-101.",
        index,
        source_key="overview-doc",
        source_group="docs/full/overview.md",
        different_group_only=True,
    )

    assert "weak-doc" not in resolved
    assert "strong-doc" in resolved
