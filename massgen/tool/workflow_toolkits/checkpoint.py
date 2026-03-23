"""
Checkpoint toolkit for MassGen checkpoint coordination mode.

The checkpoint tool allows the main agent to delegate tasks to
the multi-agent team for collaborative execution.
"""

from typing import Any

from .base import BaseToolkit, ToolType

_EXPECTED_ACTIONS_SCHEMA = {
    "type": "array",
    "description": (
        "Hints about tools agents should propose in their answers. " "Each entry: {tool: 'tool_name', description: 'what it does'}. " "Useful for tools that checkpoint agents may not have access to."
    ),
    "items": {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "description": "Tool name (e.g., 'mcp__vercel__deploy')",
            },
            "description": {
                "type": "string",
                "description": "What the tool does",
            },
        },
        "required": ["tool", "description"],
    },
}


class CheckpointToolkit(BaseToolkit):
    """Checkpoint toolkit for main agent task delegation."""

    @property
    def toolkit_id(self) -> str:
        return "checkpoint"

    @property
    def toolkit_type(self) -> ToolType:
        return ToolType.WORKFLOW

    def is_enabled(self, config: dict[str, Any]) -> bool:
        return config.get("checkpoint_mode", False)

    def get_tools(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        api_format = config.get("api_format", "chat_completions")

        if api_format == "claude":
            return [self._build_claude_format()]
        elif api_format == "response":
            return [self._build_response_format()]
        else:
            return [self._build_chat_completions_format()]

    def _build_claude_format(self) -> dict[str, Any]:
        return {
            "name": "checkpoint",
            "description": (
                "Delegate a task to the multi-agent team for collaborative "
                "execution. All configured agents activate and work on the "
                "task using standard coordination (iterate, refine, vote). "
                "The consensus result and workspace changes sync back to you."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What agents should accomplish",
                    },
                    "context": {
                        "type": "string",
                        "description": ("Background info, prior work, constraints"),
                    },
                    "expected_actions": _EXPECTED_ACTIONS_SCHEMA,
                },
                "required": ["task"],
            },
        }

    def _build_response_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "checkpoint",
                "description": ("Delegate a task to the multi-agent team for " "collaborative execution. All configured agents " "activate and work on the task together."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": ("What agents should accomplish"),
                        },
                        "context": {
                            "type": "string",
                            "description": ("Background info, prior work, constraints"),
                        },
                        "expected_actions": _EXPECTED_ACTIONS_SCHEMA,
                    },
                    "required": ["task"],
                },
            },
        }

    def _build_chat_completions_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "checkpoint",
                "description": ("Delegate a task to the multi-agent team for " "collaborative execution. All configured agents " "activate and work on the task together."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": ("What agents should accomplish"),
                        },
                        "context": {
                            "type": "string",
                            "description": ("Background info, prior work, constraints"),
                        },
                        "expected_actions": _EXPECTED_ACTIONS_SCHEMA,
                    },
                    "required": ["task"],
                },
            },
        }
