#!/usr/bin/env python3
"""Unit tests for binary persona easing behavior."""

from massgen.agent_config import AgentConfig
from massgen.orchestrator import Orchestrator
from massgen.persona_generator import GeneratedPersona


def test_has_peer_answers_excludes_own_answer():
    """Peer detection should not count the agent's own prior answer."""
    assert Orchestrator._has_peer_answers("agent_a", None) is False
    assert Orchestrator._has_peer_answers("agent_a", {}) is False
    assert Orchestrator._has_peer_answers("agent_a", {"agent_a": "my answer"}) is False
    assert Orchestrator._has_peer_answers("agent_a", {"agent_b": "peer answer"}) is True
    assert (
        Orchestrator._has_peer_answers(
            "agent_a",
            {"agent_a": "my answer", "agent_b": "peer answer"},
        )
        is True
    )


def test_persona_is_strong_before_peer_answers():
    """Strong persona text should be used until peer answers are visible."""
    orchestrator = Orchestrator(agents={}, config=AgentConfig())
    persona = GeneratedPersona(
        agent_id="agent_a",
        persona_text="Prioritize maintainability and simplicity.",
        attributes={},
    )
    orchestrator._generated_personas = {"agent_a": persona}

    assert (
        orchestrator._get_persona_for_agent(
            "agent_a",
            has_peer_answers=False,
        )
        == "Prioritize maintainability and simplicity."
    )


def test_persona_is_eased_after_peer_answers():
    """Eased persona text should emphasize synthesis over persona purity."""
    orchestrator = Orchestrator(agents={}, config=AgentConfig())
    persona = GeneratedPersona(
        agent_id="agent_a",
        persona_text="Prioritize maintainability and simplicity.",
        attributes={},
    )
    orchestrator._generated_personas = {"agent_a": persona}

    eased_text = orchestrator._get_persona_for_agent(
        "agent_a",
        has_peer_answers=True,
    )
    assert eased_text is not None
    # Template wraps across a line break; normalize whitespace for assertion
    normalized = " ".join(eased_text.split())
    assert "preference, not a position to defend" in normalized
    assert "synthesize the strongest ideas" in normalized
    assert "Prioritize maintainability and simplicity." in eased_text
