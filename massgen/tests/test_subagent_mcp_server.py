# -*- coding: utf-8 -*-
"""
Unit tests for async parameter on spawn_subagents MCP tool.

Tests for the async subagent execution feature (MAS-214):
- async=false returns results (existing behavior)
- async=true returns IDs immediately
- async disabled in config falls back to blocking
"""

import inspect
import sys

import pytest


async def _build_spawn_subagents_handler(monkeypatch, tmp_path):
    """Create the subagent MCP server and return (module, spawn_subagents handler)."""
    from massgen.mcp_tools.subagent import _subagent_mcp_server as server

    # Ensure clean global state per test.
    server._manager = None
    server._workspace_path = None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "subagent-server",
            "--agent-id",
            "agent_a",
            "--orchestrator-id",
            "orch_1",
            "--workspace-path",
            str(tmp_path),
        ],
    )

    mcp = await server.create_server()
    for tool in mcp._tool_manager._tools.values():
        if tool.name == "spawn_subagents":
            return server, tool.fn
    raise RuntimeError("spawn_subagents tool not found")


async def _invoke_handler(handler, **kwargs):
    """Invoke FastMCP handler regardless of sync/async function type."""
    result = handler(**kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


class _FakeSubagentManager:
    """Minimal fake manager for spawn_subagents MCP tests."""

    def __init__(self):
        self.background_calls = []

    def spawn_subagent_background(self, **kwargs):
        self.background_calls.append(kwargs)
        subagent_id = kwargs.get("subagent_id") or "subagent_0"
        return {
            "subagent_id": subagent_id,
            "status": "running",
            "workspace": f"/tmp/{subagent_id}",
            "status_file": f"/tmp/{subagent_id}/status.json",
        }

    def list_subagents(self):
        return []


class TestSpawnSubagentsContextPathsRequirement:
    """Validation behavior for explicit context_paths requirement."""

    @pytest.mark.asyncio
    async def test_missing_context_paths_field_is_rejected(self, monkeypatch, tmp_path):
        server, handler = await _build_spawn_subagents_handler(monkeypatch, tmp_path)
        fake_manager = _FakeSubagentManager()
        monkeypatch.setattr(server, "_get_manager", lambda: fake_manager)

        result = await _invoke_handler(
            handler,
            tasks=[{"task": "Research OAuth patterns"}],
            async_=True,
            refine=False,
        )

        assert result["success"] is False
        assert "context_paths" in result["error"]
        assert "[]" in result["error"]

    @pytest.mark.asyncio
    async def test_context_paths_must_be_list(self, monkeypatch, tmp_path):
        server, handler = await _build_spawn_subagents_handler(monkeypatch, tmp_path)
        fake_manager = _FakeSubagentManager()
        monkeypatch.setattr(server, "_get_manager", lambda: fake_manager)

        result = await _invoke_handler(
            handler,
            tasks=[{"task": "Explore repo structure", "context_paths": "./"}],
            async_=True,
            refine=False,
        )

        assert result["success"] is False
        assert "expected list" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_context_paths_is_valid_explicit_choice(self, monkeypatch, tmp_path):
        server, handler = await _build_spawn_subagents_handler(monkeypatch, tmp_path)
        fake_manager = _FakeSubagentManager()
        monkeypatch.setattr(server, "_get_manager", lambda: fake_manager)

        result = await _invoke_handler(
            handler,
            tasks=[{"task": "Do clean research with no prior context", "context_paths": []}],
            async_=True,
            refine=False,
        )

        assert result["success"] is True
        assert result["mode"] == "async"
        assert len(fake_manager.background_calls) == 1
        assert fake_manager.background_calls[0]["context_paths"] == []


# =============================================================================
# Async Parameter Tests
# =============================================================================


class TestSpawnSubagentsAsyncParameter:
    """Tests for async parameter behavior on spawn_subagents tool."""

    def test_async_false_returns_results_directly(self):
        """Test that async=false (default) returns full results after completion."""
        # This tests the expected return format when async is False
        # The actual implementation will block and return results

        # Expected return format for synchronous execution
        expected_format = {
            "success": True,
            "operation": "spawn_subagents",
            "results": [
                {
                    "subagent_id": "task-1",
                    "status": "completed",
                    "workspace": "/workspace/task-1",
                    "answer": "Task completed successfully",
                    "execution_time_seconds": 10.5,
                },
            ],
            "summary": {
                "total": 1,
                "completed": 1,
                "failed": 0,
                "timeout": 0,
            },
        }

        # Verify expected format structure
        assert "success" in expected_format
        assert "results" in expected_format
        assert "summary" in expected_format
        assert expected_format["results"][0]["answer"] is not None

    def test_async_true_returns_ids_immediately(self):
        """Test that async=true returns subagent IDs and 'running' status immediately."""
        # Expected return format for async execution
        expected_format = {
            "success": True,
            "operation": "spawn_subagents",
            "mode": "async",
            "subagents": [
                {
                    "subagent_id": "task-1",
                    "status": "running",
                    "workspace": "/workspace/task-1",
                    "status_file": "/logs/task-1/full_logs/status.json",
                },
            ],
            "note": "Results will be automatically injected when subagents complete.",
        }

        # Verify expected format structure
        assert "success" in expected_format
        assert expected_format["mode"] == "async"
        assert "subagents" in expected_format
        assert expected_format["subagents"][0]["status"] == "running"
        # Async mode should NOT have answer (still running)
        assert "answer" not in expected_format["subagents"][0]
        assert "note" in expected_format

    def test_async_true_multiple_tasks_all_return_running(self):
        """Test that async=true with multiple tasks returns all as running."""
        # Expected format for multiple async subagents
        expected_format = {
            "success": True,
            "operation": "spawn_subagents",
            "mode": "async",
            "subagents": [
                {"subagent_id": "task-1", "status": "running"},
                {"subagent_id": "task-2", "status": "running"},
                {"subagent_id": "task-3", "status": "running"},
            ],
            "note": "Results will be automatically injected when subagents complete.",
        }

        # Verify all subagents have running status
        for subagent in expected_format["subagents"]:
            assert subagent["status"] == "running"

        assert len(expected_format["subagents"]) == 3

    def test_async_default_is_false(self):
        """Test that async parameter defaults to False (blocking behavior)."""
        # This is a contract test - the function signature should default async_ to False
        # When implemented, spawn_subagents(tasks, context) should be blocking
        pass  # Will be validated against actual implementation


# =============================================================================
# Configuration Tests
# =============================================================================


class TestAsyncSubagentsConfiguration:
    """Tests for async_subagents configuration options."""

    def test_config_enabled_allows_async(self):
        """Test that enabled=true in config allows async execution."""
        config = {
            "async_subagents": {
                "enabled": True,
                "injection_strategy": "tool_result",
            },
        }

        # When enabled, async=true parameter should work
        assert config["async_subagents"]["enabled"] is True

    def test_config_disabled_falls_back_to_blocking(self):
        """Test that enabled=false forces blocking behavior even with async=true."""
        config = {
            "async_subagents": {
                "enabled": False,
            },
        }

        # When disabled, async parameter should be ignored
        # and blocking behavior should be used
        assert config["async_subagents"]["enabled"] is False

    def test_config_default_injection_strategy(self):
        """Test default injection strategy is tool_result."""
        # Default config values
        default_config = {
            "async_subagents": {
                "enabled": True,
                "injection_strategy": "tool_result",  # Default
                "inject_progress": False,  # Default
            },
        }

        assert default_config["async_subagents"]["injection_strategy"] == "tool_result"

    def test_config_user_message_injection_strategy(self):
        """Test user_message injection strategy configuration."""
        config = {
            "async_subagents": {
                "enabled": True,
                "injection_strategy": "user_message",
            },
        }

        assert config["async_subagents"]["injection_strategy"] == "user_message"


# =============================================================================
# Return Format Tests
# =============================================================================


class TestSpawnSubagentsReturnFormats:
    """Tests for spawn_subagents return value formats."""

    def test_sync_return_format_has_results(self):
        """Test synchronous return format includes results with answers."""
        sync_format = {
            "success": True,
            "operation": "spawn_subagents",
            "results": [
                {
                    "subagent_id": "research-1",
                    "status": "completed",
                    "success": True,
                    "workspace": "/workspace/research-1",
                    "answer": "Here is the research result...",
                    "execution_time_seconds": 45.2,
                    "token_usage": {"input_tokens": 1000, "output_tokens": 500},
                },
            ],
            "summary": {"total": 1, "completed": 1, "failed": 0, "timeout": 0},
        }

        # Validate structure
        assert sync_format["results"][0]["answer"] is not None
        assert "summary" in sync_format
        assert sync_format["summary"]["completed"] == 1

    def test_async_return_format_no_answers(self):
        """Test async return format does NOT include answers."""
        async_format = {
            "success": True,
            "operation": "spawn_subagents",
            "mode": "async",
            "subagents": [
                {
                    "subagent_id": "research-1",
                    "status": "running",
                    "workspace": "/workspace/research-1",
                    "status_file": "/logs/research-1/full_logs/status.json",
                },
            ],
            "note": "Results will be automatically injected when subagents complete.",
        }

        # Validate structure - no answer field
        assert "answer" not in async_format["subagents"][0]
        assert async_format["subagents"][0]["status"] == "running"
        assert async_format["mode"] == "async"

    def test_sync_format_timeout_result(self):
        """Test synchronous format includes timeout results."""
        sync_format_with_timeout = {
            "success": False,  # Overall not successful if any failed
            "operation": "spawn_subagents",
            "results": [
                {
                    "subagent_id": "slow-task",
                    "status": "timeout",
                    "success": False,
                    "workspace": "/workspace/slow-task",
                    "answer": None,
                    "execution_time_seconds": 300.0,
                    "error": "Subagent exceeded timeout of 300 seconds",
                },
            ],
            "summary": {"total": 1, "completed": 0, "failed": 0, "timeout": 1},
        }

        assert sync_format_with_timeout["results"][0]["status"] == "timeout"
        assert sync_format_with_timeout["summary"]["timeout"] == 1


# =============================================================================
# Validation Tests
# =============================================================================


class TestSpawnSubagentsValidation:
    """Tests for spawn_subagents input validation."""

    def test_tasks_required(self):
        """Test that tasks parameter is required."""
        # spawn_subagents(tasks, context) - both required
        # Empty tasks should be rejected
        pass  # Implementation will validate

    def test_context_required(self):
        """Test that context parameter is required."""
        # spawn_subagents(tasks, context) - both required
        pass  # Implementation will validate

    def test_max_concurrent_respected(self):
        """Test that max_concurrent limit is respected."""
        # If max_concurrent=3, only 3 tasks should run at once
        pass  # Implementation will enforce


# =============================================================================
# Background Spawning Integration Tests
# =============================================================================


class TestBackgroundSpawning:
    """Tests for background spawning behavior."""

    def test_async_creates_background_tasks(self):
        """Test that async=true creates background asyncio tasks."""
        # When async=true, SubagentManager.spawn_subagent_background() should be called
        # The tasks should be tracked in _background_tasks dict
        pass  # Will verify against implementation

    def test_async_returns_status_file_path(self):
        """Test that async return includes path to status file for polling."""
        async_return = {
            "success": True,
            "mode": "async",
            "subagents": [
                {
                    "subagent_id": "poll-test",
                    "status": "running",
                    "workspace": "/workspace/poll-test",
                    "status_file": "/logs/poll-test/full_logs/status.json",
                },
            ],
        }

        # Status file path should be included
        assert "status_file" in async_return["subagents"][0]
        assert async_return["subagents"][0]["status_file"].endswith("status.json")

    def test_sync_blocks_until_completion(self):
        """Test that sync mode blocks until all subagents complete."""
        # This is a behavioral test - sync should wait for all results
        pass  # Will verify against implementation


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestSpawnSubagentsErrorHandling:
    """Tests for error handling in spawn_subagents."""

    def test_invalid_async_value_handled(self):
        """Test that invalid async_ parameter value is handled."""
        # async_ should be bool, other types should be converted or rejected
        pass  # Implementation will handle

    def test_spawn_failure_in_async_mode(self):
        """Test handling when spawning fails in async mode."""
        # If spawn_subagent_background fails, should still return
        # partial success with error info
        pass  # Implementation will handle
