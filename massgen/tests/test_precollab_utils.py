"""Tests for the shared pre-collab artifact discovery utility."""

from __future__ import annotations

import time
from pathlib import Path

from massgen.precollab_utils import find_precollab_artifact


def _make_subagent_dir(tmp_path: Path, subagent_id: str) -> Path:
    """Create the standard subagent directory structure."""
    d = tmp_path / "subagents" / subagent_id
    d.mkdir(parents=True)
    return d


# --- Pattern 1: full_logs/final/agent_*/workspace/ ---


def test_find_precollab_artifact_in_final_workspace(tmp_path: Path):
    base = _make_subagent_dir(tmp_path, "persona_generation")
    target = base / "full_logs" / "final" / "agent_a" / "workspace"
    target.mkdir(parents=True)
    (target / "personas.json").write_text('{"personas": {}}')

    result = find_precollab_artifact(str(tmp_path), "persona_generation", "personas.json")

    assert result is not None
    assert result.name == "personas.json"
    assert "full_logs/final/agent_a/workspace" in str(result)


# --- Pattern 4: workspace/agent_*/ ---


def test_find_precollab_artifact_in_direct_workspace(tmp_path: Path):
    base = _make_subagent_dir(tmp_path, "criteria_generation")
    target = base / "workspace" / "agent_b"
    target.mkdir(parents=True)
    (target / "criteria.json").write_text('{"criteria": []}')

    result = find_precollab_artifact(str(tmp_path), "criteria_generation", "criteria.json")

    assert result is not None
    assert result.name == "criteria.json"


# --- No match ---


def test_find_precollab_artifact_returns_none_when_missing(tmp_path: Path):
    _make_subagent_dir(tmp_path, "prompt_improvement")

    result = find_precollab_artifact(str(tmp_path), "prompt_improvement", "improved_prompt.json")

    assert result is None


# --- Multiple matches: most recent wins ---


def test_find_precollab_artifact_prefers_most_recent(tmp_path: Path):
    base = _make_subagent_dir(tmp_path, "persona_generation")

    # Older file in direct workspace
    older = base / "workspace" / "agent_a"
    older.mkdir(parents=True)
    older_file = older / "personas.json"
    older_file.write_text('{"personas": {"agent_a": {}}}')

    # Ensure mtime difference
    time.sleep(0.05)

    # Newer file in final workspace
    newer = base / "full_logs" / "final" / "agent_b" / "workspace"
    newer.mkdir(parents=True)
    newer_file = newer / "personas.json"
    newer_file.write_text('{"personas": {"agent_b": {}}}')

    result = find_precollab_artifact(str(tmp_path), "persona_generation", "personas.json")

    assert result is not None
    assert result == newer_file


# --- Pattern 2: full_logs/agent_*/<ts>/<ts2>/ ---


def test_find_precollab_artifact_in_timestamped_logs(tmp_path: Path):
    base = _make_subagent_dir(tmp_path, "prompt_improvement")
    target = base / "full_logs" / "agent_x" / "20260321" / "run1"
    target.mkdir(parents=True)
    (target / "improved_prompt.json").write_text('{"prompt": "better"}')

    result = find_precollab_artifact(str(tmp_path), "prompt_improvement", "improved_prompt.json")

    assert result is not None
    assert result.name == "improved_prompt.json"


# --- Pattern 3: workspace/snapshots/agent_*/ ---


def test_find_precollab_artifact_in_snapshots(tmp_path: Path):
    base = _make_subagent_dir(tmp_path, "criteria_generation")
    target = base / "workspace" / "snapshots" / "agent_c"
    target.mkdir(parents=True)
    (target / "criteria.json").write_text('{"criteria": []}')

    result = find_precollab_artifact(str(tmp_path), "criteria_generation", "criteria.json")

    assert result is not None
    assert "snapshots" in str(result)


# --- Pattern 5: workspace/temp/agent_*/agent*/ ---


def test_find_precollab_artifact_in_temp_nested(tmp_path: Path):
    base = _make_subagent_dir(tmp_path, "persona_generation")
    target = base / "workspace" / "temp" / "agent_d" / "agent_inner"
    target.mkdir(parents=True)
    (target / "personas.json").write_text('{"personas": {}}')

    result = find_precollab_artifact(str(tmp_path), "persona_generation", "personas.json")

    assert result is not None
    assert "temp" in str(result)


# --- Nonexistent subagent dir ---


def test_find_precollab_artifact_nonexistent_subagent_dir(tmp_path: Path):
    """Returns None when the subagent directory doesn't exist at all."""
    result = find_precollab_artifact(str(tmp_path), "nonexistent", "file.json")

    assert result is None


# --- Scoping: files outside the subagent dir are NOT found ---


def test_find_precollab_artifact_ignores_files_outside_subagent_dir(tmp_path: Path):
    """Files in the log_directory root should NOT be found (unlike rglob)."""
    _make_subagent_dir(tmp_path, "prompt_improvement")
    # Place file at log root — should NOT be discovered
    (tmp_path / "improved_prompt.json").write_text('{"prompt": "stray"}')

    result = find_precollab_artifact(str(tmp_path), "prompt_improvement", "improved_prompt.json")

    assert result is None
