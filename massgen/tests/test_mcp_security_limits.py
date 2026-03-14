"""Regression tests for MCP tool argument size limits."""

from massgen.mcp_tools.security import MAX_STRING_LENGTH, validate_tool_arguments


def test_validate_tool_arguments_allows_large_spawn_subagents_task() -> None:
    """spawn_subagents should accept prompt-sized task payloads beyond the default string cap."""
    large_task = "A" * (MAX_STRING_LENGTH + 5000)

    validated = validate_tool_arguments(
        {
            "tasks": [
                {
                    "subagent_id": "round_eval_r2",
                    "task": large_task,
                },
            ],
        },
        tool_name="spawn_subagents",
    )

    assert validated["tasks"][0]["task"] == large_task
