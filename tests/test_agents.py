"""Tests for agent target registry."""

import pytest

from agentmd.agents import AGENT_TARGETS, resolve_agents, supported_agents


def test_agent_target_fields():
    agent = AGENT_TARGETS["claude"]
    assert agent.name == "claude"
    assert agent.display_name == "Claude Code"
    assert agent.file_path == "CLAUDE.md"


def test_supported_agents_returns_sorted():
    result = supported_agents()
    assert result == sorted(result)
    assert len(result) == len(AGENT_TARGETS)


def test_supported_agents_contains_all_keys():
    result = supported_agents()
    for key in AGENT_TARGETS:
        assert key in result


def test_resolve_single_agent():
    result = resolve_agents(["claude"])
    assert len(result) == 1
    assert result[0].name == "claude"


def test_resolve_multiple_agents():
    result = resolve_agents(["claude", "codex"])
    assert len(result) == 2
    names = [a.name for a in result]
    assert "claude" in names
    assert "codex" in names


def test_resolve_all():
    result = resolve_agents(["all"])
    assert len(result) == len(AGENT_TARGETS)


def test_resolve_deduplicates():
    result = resolve_agents(["claude", "claude"])
    assert len(result) == 1


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown agent"):
        resolve_agents(["nonexistent"])


def test_resolve_unknown_lists_valid():
    with pytest.raises(ValueError, match="claude"):
        resolve_agents(["bad_agent"])
