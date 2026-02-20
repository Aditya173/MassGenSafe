"""Unit tests for core Orchestrator coordination behavior."""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import AsyncMock

import pytest

from massgen.backend.base import StreamChunk


@pytest.mark.asyncio
async def test_phase_transitions_initial_to_enforcement(mock_orchestrator, monkeypatch):
    """Agents should move from answer submission to voting in the next iteration."""
    orchestrator = mock_orchestrator(num_agents=2)
    orchestrator.current_task = "Test task"

    # Keep this unit test purely in-memory.
    monkeypatch.setattr("massgen.orchestrator.get_log_session_dir", lambda: None)
    orchestrator._save_agent_snapshot = AsyncMock(return_value="snapshot-ts")

    call_counts = defaultdict(int)
    agent_ids = list(orchestrator.agents.keys())
    winner_id = agent_ids[0]

    async def fake_stream_agent_execution(
        agent_id,
        task,
        answers,
        conversation_context=None,
        paraphrase=None,
    ):
        _ = (task, answers, conversation_context, paraphrase)
        call_counts[agent_id] += 1

        # First round: each agent submits an answer.
        if call_counts[agent_id] == 1:
            yield ("result", ("answer", f"{agent_id} answer"))
            yield ("done", None)
            return

        # Second round: each agent votes, completing coordination.
        yield ("result", ("vote", {"agent_id": winner_id, "reason": "Best answer"}))
        yield ("done", None)

    monkeypatch.setattr(orchestrator, "_stream_agent_execution", fake_stream_agent_execution)

    votes = {}
    observed_chunks = []
    async for chunk in orchestrator._stream_coordination_with_agents(votes, {}):
        observed_chunks.append(chunk)

    assert call_counts[agent_ids[0]] == 2
    assert call_counts[agent_ids[1]] == 2
    assert orchestrator.agent_states[agent_ids[0]].answer == f"{agent_ids[0]} answer"
    assert orchestrator.agent_states[agent_ids[1]].answer == f"{agent_ids[1]} answer"
    assert all(state.has_voted for state in orchestrator.agent_states.values())
    assert set(votes.keys()) == set(agent_ids)
    assert all(vote["agent_id"] == winner_id for vote in votes.values())
    assert observed_chunks  # coordination streamed at least one chunk


@pytest.mark.asyncio
async def test_presentation_fallback_uses_stored_answer(mock_orchestrator, monkeypatch):
    """If final presentation yields no content, fallback should use stored answer."""
    orchestrator = mock_orchestrator(num_agents=1)
    selected_agent_id = next(iter(orchestrator.agents.keys()))
    orchestrator._selected_agent = selected_agent_id
    stored_answer = "Stored answer from coordination phase."
    orchestrator.agent_states[selected_agent_id].answer = stored_answer

    monkeypatch.setattr("massgen.orchestrator.get_log_session_dir", lambda: None)
    orchestrator._copy_all_snapshots_to_temp_workspace = AsyncMock(return_value=None)
    orchestrator._save_agent_snapshot = AsyncMock(return_value="final")

    async def empty_presentation_chat(*args, **kwargs):
        _ = (args, kwargs)
        yield StreamChunk(type="done")

    orchestrator.agents[selected_agent_id].chat = empty_presentation_chat

    chunks = []
    async for chunk in orchestrator.get_final_presentation(
        selected_agent_id,
        {
            "vote_counts": {selected_agent_id: 1},
            "voter_details": {},
            "is_tie": False,
        },
    ):
        chunks.append(chunk)

    fallback_chunks = [c for c in chunks if getattr(c, "type", None) == "content" and "Using stored answer as final presentation" in (getattr(c, "content", "") or "")]

    assert fallback_chunks
    assert stored_answer in fallback_chunks[0].content
    assert orchestrator._final_presentation_content == stored_answer


def test_get_coordination_result_includes_timeout_metadata(mock_orchestrator):
    """Execution mode needs explicit timeout signals for chunk retry handling."""
    orchestrator = mock_orchestrator(num_agents=1)
    orchestrator.is_orchestrator_timeout = True
    orchestrator.timeout_reason = "Time limit exceeded (120.0s/120s)"

    result = orchestrator.get_coordination_result()

    assert result["is_orchestrator_timeout"] is True
    assert result["timeout_reason"] == "Time limit exceeded (120.0s/120s)"
