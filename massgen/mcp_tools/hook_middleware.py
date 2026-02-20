"""FastMCP middleware for PostToolUse hook injection into MCP tool results.

This middleware reads injection payloads from a file-based IPC channel
and appends them to tool results before returning to the caller (e.g., Codex).

The orchestrator writes hook_post_tool_use.json; the middleware reads and
consumes it. This allows mid-stream injection of peer answers, human input,
and subagent completions without touching individual tool handler code.

File format for hook_post_tool_use.json:
{
  "inject": {"content": "...", "strategy": "tool_result"},
  "tool_matcher": "*",
  "expires_at": 1740000000.0,
  "sequence": 42
}
"""

from __future__ import annotations

import fnmatch
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastmcp.server.middleware import Middleware

logger = logging.getLogger(__name__)

# Import types for ToolResult construction
try:
    import mcp.types as mcp_types

    _HAS_MCP_TYPES = True
except ImportError:
    _HAS_MCP_TYPES = False


class MassGenHookMiddleware(Middleware):
    """FastMCP middleware that injects hook content into MCP tool results.

    Reads hook_post_tool_use.json from a hook directory. When present,
    appends injection content to the tool result before returning to the caller.
    Uses file-based IPC: orchestrator writes, middleware reads and consumes.
    """

    def __init__(self, hook_dir: Path) -> None:
        self._hook_dir = hook_dir
        self._last_post_sequence: int = -1

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        """Intercept tool calls to append injection content to results."""
        tool_name = context.message.name

        # Execute the actual tool
        result = await call_next(context)

        # Check for pending injection — wrapped so injection bugs never
        # break the underlying tool call that already succeeded.
        try:
            injection = self._read_post_tool_use_injection(tool_name)
            if injection:
                result = self._append_to_result(result, injection)
        except Exception as e:
            logger.error(
                "Hook middleware injection failed for tool %s: %s. " "Returning original result.",
                tool_name,
                e,
                exc_info=True,
            )

        return result

    def _read_post_tool_use_injection(self, tool_name: str) -> str | None:
        """Read and conditionally consume hook_post_tool_use.json.

        Returns injection content string if valid and matching, None otherwise.
        The file is consumed (deleted) only when matched and valid.
        """
        hook_file = self._hook_dir / "hook_post_tool_use.json"

        try:
            raw = hook_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning(
                "Failed to read hook file %s: %s",
                hook_file,
                e,
            )
            return None

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Malformed JSON in hook file %s", hook_file)
            # Consume malformed file to prevent re-reading
            hook_file.unlink(missing_ok=True)
            return None

        if not isinstance(payload, dict):
            logger.warning(
                "Hook file %s contains non-dict payload (type=%s); discarding",
                hook_file,
                type(payload).__name__,
            )
            hook_file.unlink(missing_ok=True)
            return None

        # Validate tool_matcher glob against tool_name
        tool_matcher = payload.get("tool_matcher", "*")
        if not fnmatch.fnmatch(tool_name, tool_matcher):
            # Don't consume — a different tool may match later
            return None

        # Validate expiry
        expires_at = payload.get("expires_at")
        if expires_at is not None:
            try:
                if time.time() > float(expires_at):
                    logger.debug("Hook payload expired (expires_at=%s)", expires_at)
                    hook_file.unlink(missing_ok=True)
                    return None
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid expires_at value %r in hook file %s; treating as non-expiring",
                    expires_at,
                    hook_file,
                )

        # Validate sequence (monotonically increasing)
        sequence = payload.get("sequence", 0)
        try:
            sequence = int(sequence)
        except (TypeError, ValueError):
            sequence = 0

        if sequence <= self._last_post_sequence:
            logger.debug(
                "Skipping duplicate sequence %d (last seen: %d)",
                sequence,
                self._last_post_sequence,
            )
            hook_file.unlink(missing_ok=True)
            return None

        # Extract injection content
        inject = payload.get("inject")
        if not isinstance(inject, dict) or not inject.get("content"):
            logger.warning(
                "Hook file %s has missing or empty inject.content; discarding",
                hook_file,
            )
            hook_file.unlink(missing_ok=True)
            return None

        # All checks passed — consume the file and update sequence
        self._last_post_sequence = sequence
        hook_file.unlink(missing_ok=True)

        content = inject["content"]
        logger.info(
            "Hook middleware injecting %d chars into %s result (seq=%d)",
            len(content),
            tool_name,
            sequence,
        )
        return content

    @staticmethod
    def _append_to_result(result: Any, injection: str) -> Any:
        """Append injection text to the tool result.

        ToolResult in FastMCP is typically a list of content objects or a string.
        We normalize to a list and append a TextContent with the injection.
        """
        # Build the injection content item
        if _HAS_MCP_TYPES:
            injection_item = mcp_types.TextContent(
                type="text",
                text=f"\n{injection}",
            )
        else:
            # Fallback: use a simple string
            injection_item = f"\n{injection}"

        # Handle different result types
        if isinstance(result, list):
            return [*result, injection_item]
        elif isinstance(result, str):
            # Convert string result to list format
            if _HAS_MCP_TYPES:
                original_item = mcp_types.TextContent(type="text", text=result)
                return [original_item, injection_item]
            else:
                return [result, injection_item]
        else:
            # Unknown type — try to wrap
            return [result, injection_item]
