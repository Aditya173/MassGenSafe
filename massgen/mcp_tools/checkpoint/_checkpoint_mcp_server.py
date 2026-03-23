"""
Checkpoint MCP Server for MassGen.

Provides the `checkpoint` tool that allows the main agent to delegate
tasks to the multi-agent team for collaborative execution.

The checkpoint tool produces a signal file that the orchestrator detects
to switch from solo mode to checkpoint mode.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level globals (set during server creation)
_workspace_path: Path | None = None
_agent_id: str | None = None
_gated_patterns: list[str] | None = None

CHECKPOINT_SIGNAL_FILE = ".massgen_checkpoint_signal.json"


def validate_checkpoint_params(
    task: str,
    context: str = "",
    expected_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate checkpoint tool parameters.

    Args:
        task: What agents should accomplish (required, non-empty).
        context: Background info, prior work, constraints.
        expected_actions: Hints about gated actions agents should propose.

    Returns:
        Validated parameter dict.

    Raises:
        ValueError: If parameters are invalid.
    """
    if not task or not task.strip():
        raise ValueError("task is required and must be non-empty")

    if expected_actions is not None:
        for i, action in enumerate(expected_actions):
            if "tool" not in action:
                raise ValueError(
                    f"expected_actions[{i}] must have a 'tool' field",
                )

    return {
        "task": task.strip(),
        "context": context or "",
        "expected_actions": expected_actions or [],
    }


def build_checkpoint_signal(
    task: str,
    context: str = "",
    expected_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a checkpoint signal dict for orchestrator detection.

    Args:
        task: What agents should accomplish.
        context: Background info.
        expected_actions: Hints about expected gated actions.

    Returns:
        Signal dict with type, task, context, expected_actions.
    """
    return {
        "type": "checkpoint",
        "task": task,
        "context": context or "",
        "expected_actions": expected_actions or [],
    }


def write_checkpoint_signal(
    signal: dict[str, Any],
    workspace: Path,
) -> Path:
    """Write checkpoint signal to workspace for orchestrator detection.

    Args:
        signal: The checkpoint signal dict.
        workspace: Workspace directory path.

    Returns:
        Path to the written signal file.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    signal_file = workspace / CHECKPOINT_SIGNAL_FILE
    signal_file.write_text(json.dumps(signal, indent=2))
    logger.info(f"[Checkpoint] Wrote signal to {signal_file}")
    return signal_file


def format_checkpoint_result(
    consensus: str,
    workspace_changes: list[dict[str, str]],
    action_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Format checkpoint result for return to the main agent.

    Args:
        consensus: The winning answer text from checkpoint coordination.
        workspace_changes: List of file changes (file, change type).
        action_results: Results of executed proposed_actions.

    Returns:
        Formatted result dict.
    """
    return {
        "consensus": consensus,
        "workspace_changes": workspace_changes,
        "action_results": action_results,
    }


async def create_server():
    """Factory function to create the checkpoint MCP server."""
    import argparse

    import fastmcp

    global _workspace_path, _agent_id, _gated_patterns

    parser = argparse.ArgumentParser(description="Checkpoint MCP Server")
    parser.add_argument("--workspace-path", type=str, required=True)
    parser.add_argument("--agent-id", type=str, required=True)
    parser.add_argument("--gated-patterns", type=str, default="[]")
    parser.add_argument("--hook-dir", type=str, default=None)
    args = parser.parse_args()

    _workspace_path = Path(args.workspace_path)
    _agent_id = args.agent_id
    _gated_patterns = json.loads(args.gated_patterns)

    mcp = fastmcp.FastMCP("massgen_checkpoint")

    @mcp.tool()
    def checkpoint(
        task: str,
        context: str = "",
        expected_actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Delegate a task to the multi-agent team for collaborative execution.

        All configured agents activate and work on the task using standard
        coordination (iterate, refine, vote). The consensus result and any
        workspace changes sync back to you.

        Use expected_actions to describe tools agents should include in their
        proposed_actions (especially tools they may not have access to).

        Args:
            task: What agents should accomplish (required).
            context: Background info, prior work, constraints.
            expected_actions: Hints about gated actions agents should propose.
                Each entry: {"tool": "tool_name", "description": "what it does"}

        Returns:
            Dict with consensus, workspace_changes, and action_results.
        """
        try:
            params = validate_checkpoint_params(task, context, expected_actions)
        except ValueError as e:
            return {
                "success": False,
                "operation": "checkpoint",
                "error": str(e),
            }

        signal = build_checkpoint_signal(
            task=params["task"],
            context=params["context"],
            expected_actions=params["expected_actions"],
        )

        write_checkpoint_signal(signal, _workspace_path)

        return {
            "success": True,
            "operation": "checkpoint",
            "message": (f"Checkpoint delegated: {params['task'][:100]}. " "All agents are now working on this task. " "Results will be returned when consensus is reached."),
            "signal": signal,
        }

    return mcp


if __name__ == "__main__":
    import asyncio

    import fastmcp

    asyncio.run(fastmcp.run(create_server))
