"""Tests for the MassGen viewer module (TDD - written before implementation)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from massgen.events import MassGenEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def log_dir(tmp_path: Path) -> Path:
    """Create a minimal log directory with status.json and events.jsonl."""
    attempt_dir = tmp_path / "log_20260309_120000" / "turn_1" / "attempt_1"
    attempt_dir.mkdir(parents=True)

    # Write status.json
    status = {
        "is_complete": True,
        "meta": {"question": "Write a poem about the sea"},
        "agents": {
            "agent_alpha": {"status": "completed", "model": "openai/gpt-4o"},
            "agent_beta": {"status": "completed", "model": "anthropic/claude-sonnet-4-20250514"},
        },
    }
    (attempt_dir / "status.json").write_text(json.dumps(status))

    # Write events.jsonl
    events = [
        MassGenEvent.create("round_start", agent_id=None, round_number=1),
        MassGenEvent.create("text", agent_id="agent_alpha", round_number=1, content="The sea is vast"),
        MassGenEvent.create("text", agent_id="agent_beta", round_number=1, content="Waves crash on shore"),
        MassGenEvent.create("answer_submitted", agent_id="agent_alpha", round_number=1),
        MassGenEvent.create("answer_submitted", agent_id="agent_beta", round_number=1),
        MassGenEvent.create("vote", agent_id="agent_beta", round_number=1, voted_for="agent_alpha"),
        MassGenEvent.create("winner_selected", agent_id=None, round_number=1, winner="agent_alpha"),
    ]
    with open(attempt_dir / "events.jsonl", "w") as f:
        for event in events:
            f.write(event.to_json() + "\n")

    return attempt_dir


@pytest.fixture()
def log_base_dir(tmp_path: Path) -> Path:
    """Create a base log dir with multiple sessions."""
    base = tmp_path / ".massgen" / "massgen_logs"
    base.mkdir(parents=True)

    # Older session
    old = base / "log_20260308_100000" / "turn_1" / "attempt_1"
    old.mkdir(parents=True)
    old_status = {"is_complete": True, "meta": {"question": "Old question"}, "agents": {"agent_a": {}}}
    (old / "status.json").write_text(json.dumps(old_status))
    (old / "events.jsonl").write_text("")

    # Newer session
    new = base / "log_20260309_120000" / "turn_1" / "attempt_1"
    new.mkdir(parents=True)
    new_status = {"is_complete": False, "meta": {"question": "New question"}, "agents": {"agent_x": {}, "agent_y": {}}}
    (new / "status.json").write_text(json.dumps(new_status))
    (new / "events.jsonl").write_text("")

    return base


@pytest.fixture()
def live_log_dir(tmp_path: Path) -> Path:
    """Create a log directory for a session still in progress."""
    attempt_dir = tmp_path / "log_live" / "turn_1" / "attempt_1"
    attempt_dir.mkdir(parents=True)

    status = {
        "is_complete": False,
        "meta": {"question": "Live question"},
        "agents": {"agent_a": {"status": "streaming"}, "agent_b": {"status": "waiting"}},
    }
    (attempt_dir / "status.json").write_text(json.dumps(status))
    (attempt_dir / "events.jsonl").write_text("")

    return attempt_dir


# ---------------------------------------------------------------------------
# Tests: extract_session_info
# ---------------------------------------------------------------------------


class TestExtractSessionInfo:
    def test_from_status_json(self, log_dir: Path):
        from massgen.viewer import extract_session_info

        info = extract_session_info(log_dir)

        assert set(info.agent_ids) == {"agent_alpha", "agent_beta"}
        assert info.question == "Write a poem about the sea"
        assert info.is_complete is True
        assert info.log_dir == log_dir

    def test_fallback_to_events_when_no_status(self, log_dir: Path):
        from massgen.viewer import extract_session_info

        # Remove status.json to force fallback
        (log_dir / "status.json").unlink()

        info = extract_session_info(log_dir)

        # Should still find agent_ids from events.jsonl
        assert "agent_alpha" in info.agent_ids
        assert "agent_beta" in info.agent_ids
        # Question unknown without status.json
        assert info.question == ""
        # Assume not complete when we can't determine
        assert info.is_complete is False

    def test_empty_events_and_no_status_raises(self, tmp_path: Path):
        from massgen.viewer import extract_session_info

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        (empty_dir / "events.jsonl").write_text("")

        with pytest.raises(ValueError, match="No agents found"):
            extract_session_info(empty_dir)

    def test_agent_models_extracted(self, log_dir: Path):
        from massgen.viewer import extract_session_info

        info = extract_session_info(log_dir)

        assert info.agent_models.get("agent_alpha") == "openai/gpt-4o"
        assert info.agent_models.get("agent_beta") == "anthropic/claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Tests: resolve_log_dir
# ---------------------------------------------------------------------------


class TestResolveLogDir:
    def test_explicit_path(self, log_dir: Path):
        from massgen.viewer import resolve_log_dir

        # Passing the attempt dir directly
        resolved = resolve_log_dir(str(log_dir))

        assert resolved == log_dir
        assert (resolved / "events.jsonl").exists()

    def test_explicit_session_root_finds_latest_turn_attempt(self, log_dir: Path):
        from massgen.viewer import resolve_log_dir

        # Pass the session root (parent of turn_1/)
        session_root = log_dir.parent.parent
        resolved = resolve_log_dir(str(session_root))

        assert resolved == log_dir
        assert (resolved / "events.jsonl").exists()

    def test_finds_latest_session(self, log_base_dir: Path):
        from massgen.viewer import resolve_log_dir

        # viewer.resolve_log_dir(None) delegates to session_exporter.resolve_log_dir(None)
        # which calls find_latest_log() -> get_logs_dir(). Mock session_exporter to use
        # our test fixture directory.
        latest_attempt = log_base_dir / "log_20260309_120000" / "turn_1" / "attempt_1"
        with patch("massgen.session_exporter.resolve_log_dir", return_value=latest_attempt):
            resolved = resolve_log_dir(None)

        # Should pick the newer session (log_20260309_120000)
        assert "log_20260309_120000" in str(resolved)

    def test_turn_and_attempt_selection(self, tmp_path: Path):
        from massgen.viewer import resolve_log_dir

        session_root = tmp_path / "log_test"
        # Create two turns, two attempts each
        for t in (1, 2):
            for a in (1, 2):
                d = session_root / f"turn_{t}" / f"attempt_{a}"
                d.mkdir(parents=True)
                (d / "events.jsonl").write_text("")

        resolved = resolve_log_dir(str(session_root), turn=1, attempt=2)

        assert resolved == session_root / "turn_1" / "attempt_2"

    def test_nonexistent_path_raises(self):
        from massgen.viewer import resolve_log_dir

        with pytest.raises(FileNotFoundError):
            resolve_log_dir("/nonexistent/path/that/does/not/exist")


# ---------------------------------------------------------------------------
# Tests: EventFeeder
# ---------------------------------------------------------------------------


class TestEventFeeder:
    def test_replay_all_events(self, log_dir: Path):
        from massgen.viewer import EventFeeder

        received: list[MassGenEvent] = []

        feeder = EventFeeder(
            events_path=log_dir / "events.jsonl",
            event_callback=received.append,
            is_live=False,
        )
        feeder.start()
        feeder.wait(timeout=5.0)

        assert len(received) == 7
        assert received[0].event_type == "round_start"
        assert received[-1].event_type == "winner_selected"

    def test_live_streaming(self, live_log_dir: Path):
        from massgen.viewer import EventFeeder

        events_path = live_log_dir / "events.jsonl"
        received: list[MassGenEvent] = []

        feeder = EventFeeder(
            events_path=events_path,
            event_callback=received.append,
            is_live=True,
            status_path=live_log_dir / "status.json",
        )
        feeder.start()

        # Simulate events being written by another process
        time.sleep(0.2)
        with open(events_path, "a") as f:
            f.write(MassGenEvent.create("text", agent_id="agent_a", round_number=1, content="Hello").to_json() + "\n")

        time.sleep(0.8)

        # Should have picked up the event
        assert any(e.event_type == "text" for e in received)

        feeder.stop()

    def test_detects_completion(self, live_log_dir: Path):
        from massgen.viewer import EventFeeder

        events_path = live_log_dir / "events.jsonl"
        status_path = live_log_dir / "status.json"
        received: list[MassGenEvent] = []

        feeder = EventFeeder(
            events_path=events_path,
            event_callback=received.append,
            is_live=True,
            status_path=status_path,
        )
        feeder.start()

        # Write an event, then mark session complete
        time.sleep(0.2)
        with open(events_path, "a") as f:
            f.write(MassGenEvent.create("text", agent_id="agent_a", round_number=1, content="Final").to_json() + "\n")

        # Update status.json to mark complete
        status = json.loads(status_path.read_text())
        status["is_complete"] = True
        status_path.write_text(json.dumps(status))

        # Feeder should stop on its own
        feeder.wait(timeout=5.0)

        assert feeder.is_done
        assert any(e.event_type == "text" for e in received)


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestViewerCLI:
    def test_parse_basic_args(self):
        from massgen.viewer import build_viewer_parser

        parser = build_viewer_parser()
        args = parser.parse_args(["/some/log/dir"])

        assert args.log_dir == "/some/log/dir"
        assert args.turn is None
        assert args.attempt is None
        assert args.replay_speed == 0
        assert args.pick is False
        assert args.web is False

    def test_parse_all_flags(self):
        from massgen.viewer import build_viewer_parser

        parser = build_viewer_parser()
        args = parser.parse_args(
            [
                "/some/log",
                "--turn",
                "2",
                "--attempt",
                "1",
                "--replay-speed",
                "1.5",
                "--pick",
                "--web",
                "--port",
                "9000",
            ],
        )

        assert args.log_dir == "/some/log"
        assert args.turn == 2
        assert args.attempt == 1
        assert args.replay_speed == 1.5
        assert args.pick is True
        assert args.web is True
        assert args.port == 9000

    def test_no_args_defaults(self):
        from massgen.viewer import build_viewer_parser

        parser = build_viewer_parser()
        args = parser.parse_args([])

        assert args.log_dir is None
        assert args.replay_speed == 0
