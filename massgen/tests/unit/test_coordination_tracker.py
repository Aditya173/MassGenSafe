"""Unit tests for core CoordinationTracker behaviors."""

from massgen.coordination_tracker import CoordinationTracker


def _init_tracker(agent_ids=None) -> CoordinationTracker:
    tracker = CoordinationTracker()
    tracker.initialize_session(agent_ids or ["agent_a", "agent_b"], user_prompt="test")
    return tracker


def test_add_agent_answer_assigns_incrementing_labels():
    tracker = _init_tracker(["agent_a", "agent_b"])

    tracker.add_agent_answer("agent_a", "first")
    tracker.add_agent_answer("agent_a", "second")

    labels = [a.label for a in tracker.answers_by_agent["agent_a"]]
    assert labels == ["agent1.1", "agent1.2"]
    assert tracker.get_latest_answer_label("agent_a") == "agent1.2"


def test_vote_uses_label_from_voter_context():
    tracker = _init_tracker(["agent_a", "agent_b"])
    tracker.add_agent_answer("agent_a", "answer from a")
    tracker.add_agent_answer("agent_b", "answer from b")
    tracker.start_new_iteration()

    # Agent A is shown only agent B's answer in context.
    tracker.track_agent_context("agent_a", {"agent_b": "answer from b"})
    tracker.add_agent_vote("agent_a", {"agent_id": "agent_b", "reason": "best fit"})

    vote = tracker.votes[-1]
    assert vote.voter_id == "agent_a"
    assert vote.voter_anon_id == "agent1"
    assert vote.voted_for == "agent_b"
    assert vote.voted_for_label == "agent2.1"


def test_complete_agent_restart_increments_round_only_when_pending():
    tracker = _init_tracker(["agent_a", "agent_b"])

    # No pending restart yet; should be a no-op.
    tracker.complete_agent_restart("agent_a")
    assert tracker.get_agent_round("agent_a") == 0

    tracker.track_restart_signal("agent_b", ["agent_a"])
    tracker.complete_agent_restart("agent_a")
    assert tracker.get_agent_round("agent_a") == 1

    # Completing again without a new pending restart remains unchanged.
    tracker.complete_agent_restart("agent_a")
    assert tracker.get_agent_round("agent_a") == 1


def test_start_final_round_sets_winner_and_advances_round_from_max():
    tracker = _init_tracker(["agent_a", "agent_b"])
    tracker.agent_rounds["agent_a"] = 1
    tracker.agent_rounds["agent_b"] = 2

    tracker.start_final_round("agent_a")

    assert tracker.is_final_round is True
    assert tracker.final_winner == "agent_a"
    assert tracker.get_agent_round("agent_a") == 3


def test_anonymous_mapping_uses_sorted_agent_ids():
    tracker = _init_tracker(["agent_c", "agent_a", "agent_b"])

    anon_to_real = tracker.get_anonymous_agent_mapping()
    real_to_anon = tracker.get_reverse_agent_mapping()

    assert anon_to_real == {
        "agent1": "agent_a",
        "agent2": "agent_b",
        "agent3": "agent_c",
    }
    assert real_to_anon["agent_a"] == "agent1"
    assert real_to_anon["agent_b"] == "agent2"
    assert real_to_anon["agent_c"] == "agent3"
