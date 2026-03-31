"""Tests for auto_trace_analysis config wiring and runtime behaviour.

Covers:
- CLI wiring of ``auto_trace_analysis`` from YAML coordination config
- Disabled by default
- _should_spawn_trace_analyzer guard logic
- _build_trace_analyzer_task content
- _read_execution_trace_for_agent from snapshot storage
- _write_trace_analysis_to_memory file creation
- Cancellation of in-flight trace tasks
- Result enqueuing via _on_background_subagent_complete
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Config wiring (existing tests)
# ---------------------------------------------------------------------------


def test_parse_coordination_config_wires_auto_trace_analysis():
    """_parse_coordination_config should forward auto_trace_analysis."""
    from massgen.cli import _parse_coordination_config

    coord_cfg: dict[str, Any] = {"auto_trace_analysis": True}
    result = _parse_coordination_config(coord_cfg)
    assert result.auto_trace_analysis is True


def test_auto_trace_analysis_defaults_to_false():
    """auto_trace_analysis should be False by default."""
    from massgen.agent_config import CoordinationConfig

    config = CoordinationConfig()
    assert config.auto_trace_analysis is False


def test_parse_coordination_config_auto_trace_analysis_absent():
    """When auto_trace_analysis is not in YAML, it defaults to False."""
    from massgen.cli import _parse_coordination_config

    coord_cfg: dict[str, Any] = {}
    result = _parse_coordination_config(coord_cfg)
    assert result.auto_trace_analysis is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeCoordinationConfig:
    auto_trace_analysis: bool = False
    orchestrator_managed_round_evaluator: bool = False


@dataclass
class _FakeConfig:
    coordination_config: _FakeCoordinationConfig = field(
        default_factory=_FakeCoordinationConfig,
    )


@dataclass
class _FakeAgentState:
    restart_count: int = 0
    answer: str | None = None


class _FakeFilesystemManager:
    def __init__(self, tmp: Path) -> None:
        self.snapshot_storage = tmp / "snapshots"
        self.snapshot_storage.mkdir(parents=True, exist_ok=True)
        self.cwd = tmp / "workspace"
        self.cwd.mkdir(parents=True, exist_ok=True)


class _FakeBackend:
    def __init__(self, fs_mgr: _FakeFilesystemManager) -> None:
        self.filesystem_manager = fs_mgr


class _FakeAgent:
    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend


def _make_orchestrator(
    tmp_path: Path,
    *,
    auto_trace: bool = True,
    restart_count: int = 1,
):
    """Build a minimal orchestrator-like object for unit testing."""
    from massgen.orchestrator import Orchestrator

    fs_mgr = _FakeFilesystemManager(tmp_path)
    agent = _FakeAgent(_FakeBackend(fs_mgr))
    config = _FakeConfig(
        coordination_config=_FakeCoordinationConfig(
            auto_trace_analysis=auto_trace,
        ),
    )

    # Orchestrator.__init__ requires agents dict and does a lot of setup.
    # We patch __init__ and manually set only the attributes we need.
    with patch.object(Orchestrator, "__init__", lambda self, **kw: None):
        orch = Orchestrator()

    orch.config = config
    orch.agents = {"agent_a": agent}
    orch.agent_states = {
        "agent_a": _FakeAgentState(restart_count=restart_count),
    }
    orch._background_trace_tasks = {}  # type: ignore[attr-defined]
    orch._pending_subagent_results = {}  # type: ignore[attr-defined]
    orch._original_task = "Write a poem"
    return orch


# ---------------------------------------------------------------------------
# _should_spawn_trace_analyzer
# ---------------------------------------------------------------------------


def test_should_spawn_round_1_returns_false(tmp_path: Path):
    """Should NOT spawn at round 1 (restart_count=0)."""
    orch = _make_orchestrator(tmp_path, restart_count=0)
    assert orch._should_spawn_trace_analyzer("agent_a") is False


def test_should_spawn_round_2_returns_true(tmp_path: Path):
    """Should spawn when restart_count >= 1 and auto_trace_analysis=True."""
    orch = _make_orchestrator(tmp_path, restart_count=1)
    assert orch._should_spawn_trace_analyzer("agent_a") is True


def test_should_spawn_explicit_false(tmp_path: Path):
    """Should NOT spawn when auto_trace_analysis=False even at round 2+."""
    orch = _make_orchestrator(tmp_path, auto_trace=False, restart_count=2)
    assert orch._should_spawn_trace_analyzer("agent_a") is False


@pytest.mark.asyncio
async def test_no_double_spawn(tmp_path: Path):
    """Should skip if an in-flight trace task already exists."""
    orch = _make_orchestrator(tmp_path, restart_count=2)

    # Simulate a running task
    async def _hang() -> None:
        await asyncio.sleep(999)

    task = asyncio.create_task(_hang())
    orch._background_trace_tasks["agent_a"] = task
    assert orch._should_spawn_trace_analyzer("agent_a") is False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_should_spawn_unknown_agent(tmp_path: Path):
    """Should return False for an agent not in agent_states."""
    orch = _make_orchestrator(tmp_path, restart_count=1)
    assert orch._should_spawn_trace_analyzer("unknown_agent") is False


# ---------------------------------------------------------------------------
# _get_execution_trace_path_for_agent
# ---------------------------------------------------------------------------


def test_get_trace_path_from_snapshot(tmp_path: Path):
    """Should return path to execution_trace.md from snapshot_storage."""
    orch = _make_orchestrator(tmp_path)
    fs_mgr = orch.agents["agent_a"].backend.filesystem_manager
    trace_path = fs_mgr.snapshot_storage / "execution_trace.md"
    trace_path.write_text("# Trace\nTool call 1...", encoding="utf-8")

    result = orch._get_execution_trace_path_for_agent("agent_a")
    assert result == trace_path


def test_get_trace_path_missing_file(tmp_path: Path):
    """Should return None when no trace file exists."""
    orch = _make_orchestrator(tmp_path)
    result = orch._get_execution_trace_path_for_agent("agent_a")
    assert result is None


def test_get_trace_path_unknown_agent(tmp_path: Path):
    """Should return None for an agent not in agents dict."""
    orch = _make_orchestrator(tmp_path)
    result = orch._get_execution_trace_path_for_agent("no_such_agent")
    assert result is None


# ---------------------------------------------------------------------------
# _build_trace_analyzer_task
# ---------------------------------------------------------------------------


def test_build_task_includes_trace_path(tmp_path: Path):
    """Task string should include original task and trace file path."""
    orch = _make_orchestrator(tmp_path)
    task = orch._build_trace_analyzer_task(
        "agent_a",
        round_number=2,
        trace_path="/snapshots/execution_trace.md",
    )
    assert "Write a poem" in task  # original_task
    assert "/snapshots/execution_trace.md" in task  # trace path
    assert "round 1" in task  # round_number - 1
    assert "DO / DON'T / CRITICAL ERRORS" in task


# ---------------------------------------------------------------------------
# _write_trace_analysis_to_memory
# ---------------------------------------------------------------------------


def test_write_trace_analysis_to_memory(tmp_path: Path):
    """Memory file should be written to memory/short_term/."""
    orch = _make_orchestrator(tmp_path)
    memory_block = "---\nname: test\n---\nDO: something"
    orch._write_trace_analysis_to_memory("agent_a", 2, memory_block)

    fs_mgr = orch.agents["agent_a"].backend.filesystem_manager
    target = fs_mgr.cwd / "memory" / "short_term" / "trace_analysis_round_2.md"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == memory_block


def test_write_trace_analysis_creates_dirs(tmp_path: Path):
    """Should create memory/short_term/ if it doesn't exist."""
    orch = _make_orchestrator(tmp_path)
    # Remove the workspace dir to test directory creation
    fs_mgr = orch.agents["agent_a"].backend.filesystem_manager
    memory_dir = fs_mgr.cwd / "memory" / "short_term"
    assert not memory_dir.exists()

    orch._write_trace_analysis_to_memory("agent_a", 3, "test content")
    assert memory_dir.exists()


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_trace_analyzer_on_new_answer(tmp_path: Path):
    """_cancel_running_background_work_for_agent should cancel trace task."""
    orch = _make_orchestrator(tmp_path)

    # Mock the subagent cancellation methods
    async def _noop_cancel(_aid: str) -> int:
        return 0

    orch._cancel_running_subagents_for_agent = _noop_cancel  # type: ignore[assignment]

    # Create a long-running background trace task
    async def _hang() -> None:
        await asyncio.sleep(999)

    task = asyncio.create_task(_hang())
    orch._background_trace_tasks["agent_a"] = task

    await orch._cancel_running_background_work_for_agent("agent_a")

    # Give event loop a tick to finalize the cancellation
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert task.cancelled()
    assert "agent_a" not in orch._background_trace_tasks


# ---------------------------------------------------------------------------
# Result enqueuing
# ---------------------------------------------------------------------------


def test_format_trace_analyzer_for_memory():
    """_format_trace_analyzer_for_memory_static produces YAML frontmatter."""
    from massgen.orchestrator import Orchestrator
    from massgen.subagent.models import SubagentResult

    result = SubagentResult(
        subagent_id="trace_analyzer_r2",
        status="completed",
        success=True,
        answer="### DO\n- Read files first",
    )
    block = Orchestrator._format_trace_analyzer_for_memory_static(result, 2)
    assert block is not None
    assert "name: execution_trace_round_2" in block
    assert "tier: short_term" in block
    assert "### DO" in block


def test_format_trace_analyzer_empty_answer():
    """Empty answer should return None."""
    from massgen.orchestrator import Orchestrator
    from massgen.subagent.models import SubagentResult

    result = SubagentResult(
        subagent_id="trace_analyzer_r2",
        status="completed",
        success=True,
        answer="   ",
    )
    block = Orchestrator._format_trace_analyzer_for_memory_static(result, 2)
    assert block is None


# ---------------------------------------------------------------------------
# Direct spawn lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_spawn_lock_exists():
    """_get_direct_spawn_lock should return an asyncio.Lock."""
    from massgen.mcp_tools.subagent._subagent_mcp_server import (
        _get_direct_spawn_lock,
    )

    lock = _get_direct_spawn_lock()
    assert isinstance(lock, asyncio.Lock)
    # Same instance on second call
    assert _get_direct_spawn_lock() is lock


# ---------------------------------------------------------------------------
# Registry visibility for direct spawns
# ---------------------------------------------------------------------------


def test_direct_spawn_saves_registry_called():
    """spawn_subagents_direct should call _save_subagents_to_filesystem."""
    # Verify the source code of spawn_subagents_direct references the save
    import inspect

    from massgen.mcp_tools.subagent import _subagent_mcp_server as mod

    src = inspect.getsource(mod.spawn_subagents_direct)
    assert "_save_subagents_to_filesystem" in src
