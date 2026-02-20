"""Unit tests for TaskDecomposer parsing helpers."""

import importlib.util
from pathlib import Path


def _load_task_decomposer_module():
    """Load task_decomposer module directly to avoid package import side effects."""
    module_path = Path(__file__).resolve().parents[1] / "task_decomposer.py"
    spec = importlib.util.spec_from_file_location("task_decomposer_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_subtasks_from_plain_json_text() -> None:
    module = _load_task_decomposer_module()
    decomposer = module.TaskDecomposer(module.TaskDecomposerConfig())

    text = '{"subtasks": {"agent_a": "Research data", "agent_b": "Build implementation"}}'
    parsed = decomposer._parse_subtasks_from_text(text, ["agent_a", "agent_b"])

    assert parsed == {
        "agent_a": "Research data",
        "agent_b": "Build implementation",
    }


def test_parse_subtasks_from_markdown_json_block() -> None:
    module = _load_task_decomposer_module()
    decomposer = module.TaskDecomposer(module.TaskDecomposerConfig())

    text = "Here is the plan:\n" "```json\n" '{"subtasks": {"agent_a": "Design architecture", "agent_b": "Implement UI"}}\n' "```"
    parsed = decomposer._parse_subtasks_from_text(text, ["agent_a", "agent_b"])

    assert parsed == {
        "agent_a": "Design architecture",
        "agent_b": "Implement UI",
    }


def test_normalize_subtasks_fills_missing_agents() -> None:
    module = _load_task_decomposer_module()
    decomposer = module.TaskDecomposer(module.TaskDecomposerConfig())

    parsed = decomposer._normalize_subtasks(
        {
            "agent_a": "Write tests",
        },
        ["agent_a", "agent_b", "agent_c"],
    )

    assert parsed["agent_a"] == "Write tests"
    assert "agent_b" in parsed
    assert "agent_c" in parsed


def test_build_decomposition_prompt_includes_balance_and_bleed_guidance() -> None:
    module = _load_task_decomposer_module()
    decomposer = module.TaskDecomposer(module.TaskDecomposerConfig())

    prompt = decomposer._build_decomposition_prompt(
        task="Create a website about Bob Dylan",
        agent_descriptions=[
            "- agent_a: General-purpose agent",
            "- agent_b: General-purpose agent",
        ],
        agent_ids=["agent_a", "agent_b"],
    )

    assert "Balance depth: each subtask should include implementation work plus quality validation." in prompt
    assert "Keep ownership-first boundaries" in prompt
    assert "Keep cross-subtask bleed limited to integration needs" in prompt
    assert "Write each subtask as 2-3 sentences" in prompt
    assert "start at kickoff in parallel" in prompt
    assert "use anonymous aliases (`agent1`, `agent2`, ...)" in prompt


def test_fallback_subtasks_do_not_echo_system_message_snippets() -> None:
    module = _load_task_decomposer_module()
    decomposer = module.TaskDecomposer(module.TaskDecomposerConfig())

    marker = "DO_NOT_LEAK_THIS_PROMPT_SNIPPET"
    subtasks = decomposer._generate_fallback_subtasks(
        task="Build a website",
        agent_ids=["agent_a"],
        system_messages={"agent_a": f"custom prompt with {marker} and extra instructions"},
    )

    assert "agent_a" in subtasks
    assert marker not in subtasks["agent_a"]
    assert "aligned with your specialization" in subtasks["agent_a"]
