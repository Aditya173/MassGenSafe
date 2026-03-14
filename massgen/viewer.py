"""MassGen Viewer — observe automation sessions in the TUI.

Usage:
    massgen viewer [log_dir]           # View latest or specified session
    massgen viewer --pick              # Interactive session picker
    massgen viewer <log_dir> --web     # View in browser via textual-serve
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from massgen.events import EventReader

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class SessionInfo:
    """Metadata extracted from a log directory."""

    log_dir: Path
    agent_ids: list[str]
    agent_models: dict[str, str] = field(default_factory=dict)
    question: str = ""
    is_complete: bool = False
    session_root: Path | None = None


# ---------------------------------------------------------------------------
# Log directory resolution
#
# Delegates to session_exporter.resolve_log_dir / find_latest_log for core
# resolution logic (skips incomplete logs, handles name lookup, log_ prefix).
# Adds viewer-specific features: --turn, --attempt, --pick.
# ---------------------------------------------------------------------------


def resolve_log_dir(
    path: str | None = None,
    turn: int | None = None,
    attempt: int | None = None,
    pick: bool = False,
) -> Path:
    """Resolve a log directory to the attempt-level directory with events.jsonl.

    Uses session_exporter.resolve_log_dir() for core resolution (handles
    name lookup, skips incomplete logs, etc.), then applies --turn/--attempt
    overrides if specified.

    Args:
        path: Explicit path or log directory name.
               If None, finds the latest session with data.
        turn: Specific turn number (default: latest).
        attempt: Specific attempt number (default: latest).
        pick: If True and path is None, show interactive session picker.

    Returns:
        Path to the attempt directory containing events.jsonl.

    Raises:
        FileNotFoundError: If no session can be found.
    """
    if pick and path is None:
        return _interactive_pick()

    from massgen.session_exporter import resolve_log_dir as _exporter_resolve

    resolved = _exporter_resolve(path)

    # If --turn or --attempt specified, re-resolve from the session root
    if turn is not None or attempt is not None:
        session_root = _find_session_root(resolved)
        if session_root:
            resolved = _resolve_to_attempt(session_root, turn=turn, attempt=attempt)

    return resolved


def _resolve_to_attempt(root: Path, turn: int | None = None, attempt: int | None = None) -> Path:
    """Given a session root, descend to a specific turn/attempt directory.

    Only used when --turn or --attempt is explicitly specified.
    """
    # Already at attempt level?
    if (root / "events.jsonl").exists() or (root / "status.json").exists():
        return root

    # Find turn directories
    turn_dirs = sorted(root.glob("turn_*"), key=lambda p: p.name)
    if not turn_dirs:
        raise FileNotFoundError(f"No turn directories found in {root}")

    if turn is not None:
        turn_dir = root / f"turn_{turn}"
        if not turn_dir.exists():
            raise FileNotFoundError(f"Turn {turn} not found in {root}")
    else:
        turn_dir = turn_dirs[-1]

    # Find attempt directories
    attempt_dirs = sorted(turn_dir.glob("attempt_*"), key=lambda p: p.name)
    if not attempt_dirs:
        if (turn_dir / "events.jsonl").exists():
            return turn_dir
        raise FileNotFoundError(f"No attempt directories found in {turn_dir}")

    if attempt is not None:
        attempt_dir = turn_dir / f"attempt_{attempt}"
        if not attempt_dir.exists():
            raise FileNotFoundError(f"Attempt {attempt} not found in {turn_dir}")
    else:
        attempt_dir = attempt_dirs[-1]

    return attempt_dir


def _interactive_pick() -> Path:
    """Show an interactive session picker in the terminal."""
    from massgen.logs_analyzer import get_logs_dir
    from massgen.session_exporter import resolve_log_dir as _exporter_resolve

    logs_dir = get_logs_dir()
    if not logs_dir.exists():
        raise FileNotFoundError(f"No log directory found at {logs_dir}")

    sessions = sorted(logs_dir.glob("log_*"), key=lambda p: p.name, reverse=True)
    if not sessions:
        raise FileNotFoundError(f"No sessions found in {logs_dir}")

    # Show sessions with basic info
    print("\nAvailable sessions:\n")
    choices: list[tuple[Path, Path | None]] = []  # (session_dir, resolved_attempt)
    for i, session_dir in enumerate(sessions[:20]):
        try:
            resolved = _exporter_resolve(session_dir.name)
            status_path = resolved / "status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text())
                question = status.get("meta", {}).get("question", "")[:60]
                is_complete = status.get("is_complete", False)
                agents = list(status.get("agents", {}).keys())
                state = "DONE" if is_complete else "LIVE"
                print(f'  [{i + 1}] {session_dir.name}  [{state}]  {len(agents)} agents  "{question}"')
            else:
                print(f"  [{i + 1}] {session_dir.name}")
            choices.append((session_dir, resolved))
        except Exception:
            print(f"  [{i + 1}] {session_dir.name}  (no data)")
            choices.append((session_dir, None))

    print()
    while True:
        try:
            raw = input("Select session number (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if raw.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                _, resolved = choices[idx]
                if resolved:
                    return resolved
                # Fallback: try to resolve from name
                return _exporter_resolve(choices[idx][0].name)
            print(f"  Invalid selection. Enter 1-{len(choices)}.")
        except ValueError:
            print("  Enter a number.")


# ---------------------------------------------------------------------------
# Session info extraction
# ---------------------------------------------------------------------------


def extract_session_info(log_dir: Path) -> SessionInfo:
    """Extract session metadata from a log directory.

    Primary source: status.json (has agents, question, completion state).
    Fallback: scan events.jsonl for unique agent_ids.

    Raises:
        ValueError: If no agents can be found from any source.
    """
    log_dir = Path(log_dir)
    status_path = log_dir / "status.json"

    agent_ids: list[str] = []
    agent_models: dict[str, str] = {}
    question = ""
    is_complete = False
    session_root = _find_session_root(log_dir)

    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
            agents_dict = status.get("agents", {})
            agent_ids = list(agents_dict.keys())
            for aid, info in agents_dict.items():
                if isinstance(info, dict) and info.get("model"):
                    agent_models[aid] = info["model"]
            question = status.get("meta", {}).get("question", "")
            is_complete = bool(status.get("is_complete", False))
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: scan events.jsonl for agent_ids
    if not agent_ids:
        events_path = log_dir / "events.jsonl"
        if events_path.exists():
            reader = EventReader(events_path)
            events = reader.read_all()
            seen: dict[str, None] = {}
            for event in events:
                if event.agent_id and event.agent_id not in seen:
                    seen[event.agent_id] = None
            agent_ids = list(seen.keys())

    if not agent_ids:
        raise ValueError(f"No agents found in {log_dir} (checked status.json and events.jsonl)")

    return SessionInfo(
        log_dir=log_dir,
        agent_ids=agent_ids,
        agent_models=agent_models,
        question=question,
        is_complete=is_complete,
        session_root=session_root,
    )


def _find_session_root(log_dir: Path) -> Path | None:
    """Walk up from an attempt/turn dir to find the session root (log_YYYYMMDD_...)."""
    current = log_dir
    for _ in range(5):
        if current.name.startswith("log_"):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


# ---------------------------------------------------------------------------
# EventFeeder — streams events from disk into a callback
# ---------------------------------------------------------------------------


class EventFeeder:
    """Reads events from events.jsonl and feeds them to a callback.

    In replay mode (is_live=False): reads all events and feeds at once.
    In live mode (is_live=True): tails the file and feeds events as they appear,
    polling status.json for completion.
    """

    def __init__(
        self,
        events_path: Path,
        event_callback: Any,
        is_live: bool = False,
        replay_speed: float = 0,
        status_path: Path | None = None,
    ) -> None:
        self._events_path = Path(events_path)
        self._event_callback = event_callback
        self._is_live = is_live
        self._replay_speed = replay_speed
        self._status_path = Path(status_path) if status_path else None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._done = False

    @property
    def is_done(self) -> bool:
        return self._done

    def start(self) -> threading.Thread:
        self._thread = threading.Thread(target=self._run, daemon=True, name="viewer-event-feeder")
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        try:
            if self._is_live:
                self._run_live()
            else:
                self._run_replay()
        finally:
            self._done = True

    def _run_replay(self) -> None:
        """Feed all events from a completed session."""
        reader = EventReader(self._events_path)
        events = reader.read_all()

        if self._replay_speed > 0 and len(events) > 1:
            # Pace events based on timestamps
            prev_time: float | None = None
            for event in events:
                if self._stop_event.is_set():
                    break
                try:
                    from datetime import datetime

                    current_time = datetime.fromisoformat(event.timestamp).timestamp()
                    if prev_time is not None:
                        delay = (current_time - prev_time) / self._replay_speed
                        if delay > 0:
                            self._stop_event.wait(min(delay, 2.0))
                    prev_time = current_time
                except (ValueError, TypeError):
                    pass
                self._event_callback(event)
        else:
            # Instant replay
            for event in events:
                if self._stop_event.is_set():
                    break
                self._event_callback(event)

    def _run_live(self) -> None:
        """Tail events.jsonl and poll status.json for completion."""
        reader = EventReader(self._events_path)
        poll_interval = 0.3
        status_check_interval = 2.0
        last_status_check = 0.0

        while not self._stop_event.is_set():
            # Read new events
            events = reader.get_new_events()
            for event in events:
                self._event_callback(event)

            # Check if session completed
            now = time.monotonic()
            if self._status_path and (now - last_status_check) >= status_check_interval:
                last_status_check = now
                if self._check_completion():
                    # Drain any remaining events
                    final_events = reader.get_new_events()
                    for event in final_events:
                        self._event_callback(event)
                    return

            if not events:
                self._stop_event.wait(poll_interval)

    def _check_completion(self) -> bool:
        """Check status.json for is_complete."""
        if not self._status_path or not self._status_path.exists():
            return False
        try:
            status = json.loads(self._status_path.read_text())
            return bool(status.get("is_complete", False))
        except (json.JSONDecodeError, OSError):
            return False


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def build_viewer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="massgen viewer",
        description="View a MassGen session in the TUI (read-only)",
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=None,
        help="Log directory to view (default: latest session)",
    )
    parser.add_argument(
        "--turn",
        type=int,
        default=None,
        help="Turn number to view (default: latest)",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        default=None,
        help="Attempt number to view (default: latest)",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=0,
        help="Replay speed for completed sessions (0=instant, 1=realtime, 2=2x, etc.)",
    )
    parser.add_argument(
        "--pick",
        action="store_true",
        help="Interactively pick from recent sessions",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Serve TUI in browser via textual-serve",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web serving (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open browser when using --web",
    )
    return parser


# ---------------------------------------------------------------------------
# Main viewer command
# ---------------------------------------------------------------------------


BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_YELLOW = "\033[93m"
RESET = "\033[0m"


def viewer_command(args: argparse.Namespace) -> int:
    """Main entry point for the viewer subcommand.

    Returns:
        Exit code (0 = success).
    """
    try:
        # 1. Resolve log directory
        log_dir = resolve_log_dir(
            path=args.log_dir,
            turn=args.turn,
            attempt=args.attempt,
            pick=args.pick,
        )
        print(f"{BRIGHT_CYAN}Viewing session: {log_dir}{RESET}")

        # 2. Extract session metadata
        session_info = extract_session_info(log_dir)
        mode = "REPLAY" if session_info.is_complete else "LIVE"
        print(f"{BRIGHT_GREEN}Mode: {mode} | Agents: {', '.join(session_info.agent_ids)} | Q: \"{session_info.question[:60]}\"{RESET}")

        # 3. Handle web mode
        if args.web:
            return _run_web_viewer(args, log_dir)

        # 4. Launch TUI viewer
        return _run_tui_viewer(session_info, args.replay_speed)

    except FileNotFoundError as e:
        print(f"{BRIGHT_RED}Error: {e}{RESET}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"{BRIGHT_RED}Error: {e}{RESET}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


def _run_tui_viewer(session_info: SessionInfo, replay_speed: float) -> int:
    """Launch the Textual TUI in viewer mode."""
    import asyncio

    from massgen.frontend.displays.textual_terminal_display import (
        TextualTerminalDisplay,
    )

    # Create display in viewer mode
    display = TextualTerminalDisplay(
        session_info.agent_ids,
        agent_models=session_info.agent_models,
        viewer_mode=True,
    )
    display.initialize(session_info.question)

    # Create event feeder
    feeder = EventFeeder(
        events_path=session_info.log_dir / "events.jsonl",
        event_callback=display._app._handle_event_from_emitter,
        is_live=not session_info.is_complete,
        replay_speed=replay_speed,
        status_path=session_info.log_dir / "status.json" if not session_info.is_complete else None,
    )

    # Store feeder on display so the app can start it on mount
    display._viewer_event_feeder = feeder

    # Run the app
    asyncio.run(display.run_async())

    # Cleanup
    feeder.stop()
    return 0


def _run_web_viewer(args: argparse.Namespace, log_dir: Path) -> int:
    """Launch the viewer TUI in browser via textual-serve."""
    try:
        from textual_serve.server import Server
    except ImportError:
        print(f"{BRIGHT_RED}textual-serve not installed.{RESET}")
        print(f"{BRIGHT_CYAN}   Run: uv pip install textual-serve{RESET}")
        return 1

    cmd = f"massgen viewer {log_dir.resolve()}"
    if args.replay_speed:
        cmd += f" --replay-speed {args.replay_speed}"

    port = args.port
    print(f"{BRIGHT_CYAN}Starting MassGen Viewer in browser...{RESET}")
    print(f"{BRIGHT_GREEN}   URL: http://localhost:{port}{RESET}")
    print(f"{BRIGHT_YELLOW}   Press Ctrl+C to stop{RESET}\n")

    if not args.no_browser:
        import webbrowser

        threading.Timer(1.5, webbrowser.open, args=(f"http://localhost:{port}",)).start()

    server = Server(cmd, host="0.0.0.0", port=port)
    server.serve()
    return 0
