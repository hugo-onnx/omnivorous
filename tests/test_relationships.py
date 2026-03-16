"""Tests for deterministic relationship scoring."""

from omnivorous.relationships import RelationshipNode, build_relationships, extract_keywords


def test_extract_keywords_deduplicates_terms():
    keywords = extract_keywords("Payments API", "Refund payments", "Refund retries")
    assert keywords[:3] == ["payments", "api", "refund"]


def test_build_relationships_links_similar_nodes():
    nodes = [
        RelationshipNode(
            key="payments",
            label="Payments",
            path="payments",
            body="invoice payments refunds retries charges",
            group="payments",
        ),
        RelationshipNode(
            key="billing",
            label="Billing",
            path="billing",
            body="billing retries failed invoice refunds",
            group="billing",
        ),
        RelationshipNode(
            key="hr",
            label="HR",
            path="hr",
            body="employee vacation handbook onboarding",
            group="hr",
        ),
    ]

    relationships = build_relationships(nodes, min_score=0.01)
    assert relationships["payments"][0].target_key == "billing"
    assert "invoice" in relationships["payments"][0].shared_terms
    assert relationships["hr"] == []


def test_build_relationships_excludes_same_group_when_requested():
    nodes = [
        RelationshipNode(
            key="doc-a-1",
            label="Doc A 1",
            path="doc-a-1",
            body="invoice refunds retries",
            group="doc-a",
        ),
        RelationshipNode(
            key="doc-a-2",
            label="Doc A 2",
            path="doc-a-2",
            body="invoice refunds charges",
            group="doc-a",
        ),
        RelationshipNode(
            key="doc-b-1",
            label="Doc B 1",
            path="doc-b-1",
            body="invoice refunds billing",
            group="doc-b",
        ),
    ]

    relationships = build_relationships(
        nodes,
        min_score=0.01,
        require_different_group=True,
    )
    assert relationships["doc-a-1"][0].target_key == "doc-b-1"
