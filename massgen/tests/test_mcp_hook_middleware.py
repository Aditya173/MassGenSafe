"""Tests for MassGenHookMiddleware — FastMCP server-level PostToolUse injection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_hook_file(hook_dir: Path, payload: dict) -> Path:
    """Write a hook_post_tool_use.json file."""
    hook_dir.mkdir(parents=True, exist_ok=True)
    path = hook_dir / "hook_post_tool_use.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_payload(
    content: str = "injected content",
    tool_matcher: str = "*",
    expires_at: float | None = None,
    sequence: int = 1,
) -> dict:
    return {
        "inject": {"content": content, "strategy": "tool_result"},
        "tool_matcher": tool_matcher,
        "expires_at": expires_at or (time.time() + 60),
        "sequence": sequence,
    }


# ---------------------------------------------------------------------------
# Unit tests for _read_post_tool_use_injection
# ---------------------------------------------------------------------------


class TestReadPostToolUseInjection:
    """Tests for the file-based IPC read logic."""

    def test_returns_none_when_no_file(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        mw = MassGenHookMiddleware(hook_dir=tmp_path)
        assert mw._read_post_tool_use_injection("some_tool") is None

    def test_reads_and_returns_injection_content(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        payload = _make_payload(content="hello from peer")
        _write_hook_file(tmp_path, payload)

        mw = MassGenHookMiddleware(hook_dir=tmp_path)
        result = mw._read_post_tool_use_injection("any_tool")
        assert result == "hello from peer"

    def test_consumes_file_after_read(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        _write_hook_file(tmp_path, _make_payload())

        mw = MassGenHookMiddleware(hook_dir=tmp_path)
        mw._read_post_tool_use_injection("any_tool")

        # File should be deleted
        assert not (tmp_path / "hook_post_tool_use.json").exists()

    def test_skips_expired_payload(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        payload = _make_payload(expires_at=time.time() - 10)
        _write_hook_file(tmp_path, payload)

        mw = MassGenHookMiddleware(hook_dir=tmp_path)
        result = mw._read_post_tool_use_injection("any_tool")
        assert result is None

    def test_skips_duplicate_sequence(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        mw = MassGenHookMiddleware(hook_dir=tmp_path)

        # First read with sequence=5
        _write_hook_file(tmp_path, _make_payload(content="first", sequence=5))
        result1 = mw._read_post_tool_use_injection("any_tool")
        assert result1 == "first"

        # Second read with same sequence=5 — should skip
        _write_hook_file(tmp_path, _make_payload(content="dup", sequence=5))
        result2 = mw._read_post_tool_use_injection("any_tool")
        assert result2 is None

    def test_accepts_higher_sequence(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        mw = MassGenHookMiddleware(hook_dir=tmp_path)

        _write_hook_file(tmp_path, _make_payload(content="first", sequence=1))
        mw._read_post_tool_use_injection("any_tool")

        _write_hook_file(tmp_path, _make_payload(content="second", sequence=2))
        result = mw._read_post_tool_use_injection("any_tool")
        assert result == "second"

    def test_handles_malformed_json(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        hook_dir = tmp_path
        hook_dir.mkdir(parents=True, exist_ok=True)
        (hook_dir / "hook_post_tool_use.json").write_text("not json {{{", encoding="utf-8")

        mw = MassGenHookMiddleware(hook_dir=tmp_path)
        result = mw._read_post_tool_use_injection("any_tool")
        assert result is None

    def test_glob_matching_star(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        _write_hook_file(tmp_path, _make_payload(tool_matcher="*"))
        mw = MassGenHookMiddleware(hook_dir=tmp_path)
        assert mw._read_post_tool_use_injection("anything") is not None

    def test_glob_matching_specific(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        _write_hook_file(tmp_path, _make_payload(tool_matcher="submit_*"))
        mw = MassGenHookMiddleware(hook_dir=tmp_path)

        result_match = mw._read_post_tool_use_injection("submit_checklist")
        assert result_match is not None

    def test_glob_matching_no_match(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        _write_hook_file(tmp_path, _make_payload(tool_matcher="submit_*"))
        mw = MassGenHookMiddleware(hook_dir=tmp_path)

        result = mw._read_post_tool_use_injection("read_file")
        assert result is None
        # File should NOT be consumed when tool doesn't match
        assert (tmp_path / "hook_post_tool_use.json").exists()

    def test_missing_inject_key(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        payload = {
            "tool_matcher": "*",
            "expires_at": time.time() + 60,
            "sequence": 1,
        }
        _write_hook_file(tmp_path, payload)

        mw = MassGenHookMiddleware(hook_dir=tmp_path)
        result = mw._read_post_tool_use_injection("any_tool")
        assert result is None


# ---------------------------------------------------------------------------
# Unit tests for on_call_tool
# ---------------------------------------------------------------------------


class TestOnCallTool:
    """Tests for the FastMCP middleware on_call_tool method."""

    @pytest.mark.asyncio
    async def test_appends_injection_to_result(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        _write_hook_file(tmp_path, _make_payload(content="[Peer update]: new answer"))

        mw = MassGenHookMiddleware(hook_dir=tmp_path)

        # Mock the FastMCP middleware context
        mock_context = MagicMock()
        mock_context.message.name = "some_tool"

        # The original tool result
        original_result = ["Original tool output"]

        async def mock_call_next(ctx):
            return original_result

        result = await mw.on_call_tool(mock_context, mock_call_next)

        # Result should be a list with original + injected content
        assert len(result) == 2
        assert result[0] == "Original tool output"
        # The injected part should contain the injection content
        assert hasattr(result[1], "text") or isinstance(result[1], str)

    @pytest.mark.asyncio
    async def test_passes_through_when_no_injection(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        mw = MassGenHookMiddleware(hook_dir=tmp_path)

        mock_context = MagicMock()
        mock_context.message.name = "some_tool"

        original_result = ["Original output"]

        async def mock_call_next(ctx):
            return original_result

        result = await mw.on_call_tool(mock_context, mock_call_next)
        assert result == original_result

    @pytest.mark.asyncio
    async def test_handles_string_result(self, tmp_path: Path) -> None:
        from massgen.mcp_tools.hook_middleware import MassGenHookMiddleware

        _write_hook_file(tmp_path, _make_payload(content="injected"))

        mw = MassGenHookMiddleware(hook_dir=tmp_path)

        mock_context = MagicMock()
        mock_context.message.name = "tool"

        async def mock_call_next(ctx):
            return "plain string result"

        result = await mw.on_call_tool(mock_context, mock_call_next)
        # Should handle string results by converting to list with TextContent
        assert isinstance(result, list)
        assert len(result) == 2
