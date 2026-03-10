"""Agent target definitions for multi-agent output support."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTarget:
    """Describes where to write instruction files for a specific AI coding agent."""

    name: str
    display_name: str
    file_path: str


# To add a new agent, just add an entry here.
AGENT_TARGETS: dict[str, AgentTarget] = {
    "claude": AgentTarget("claude", "Claude Code", "CLAUDE.md"),
    "codex": AgentTarget("codex", "Codex CLI", "AGENTS.md"),
    "copilot": AgentTarget("copilot", "GitHub Copilot", ".github/copilot-instructions.md"),
    "cursor": AgentTarget("cursor", "Cursor", ".cursor/rules/agentmd.md"),
    "antigravity": AgentTarget("antigravity", "Google Antigravity", ".agent/skills/agentmd.md"),
}

DEFAULT_AGENT = "claude"


def supported_agents() -> list[str]:
    """Return sorted list of supported agent keys."""
    return sorted(AGENT_TARGETS)


def resolve_agents(names: list[str]) -> list[AgentTarget]:
    """Resolve agent names to AgentTarget instances.

    Handles "all" to return every known agent. Raises ValueError for unknown names.
    """
    if "all" in names:
        return list(AGENT_TARGETS.values())

    unknown = [n for n in names if n not in AGENT_TARGETS]
    if unknown:
        valid = ", ".join(supported_agents())
        raise ValueError(f"Unknown agent(s): {', '.join(unknown)}. Valid agents: {valid}")

    seen: set[str] = set()
    result: list[AgentTarget] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            result.append(AGENT_TARGETS[n])
    return result
