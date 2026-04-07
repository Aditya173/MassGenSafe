"""Standalone MCP server for objective-based safety checkpointing.

Any agent (Claude Code, MassGen, third-party) can connect via MCP and use
objective-based checkpointing to get structured safety plans for sequences
of irreversible actions.

Two tools:
  init   — store session context (workspace, trajectory, tools)
  checkpoint — generate a structured safety plan for an objective

Usage:
    massgen-checkpoint-mcp --config /path/to/config.yaml
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from massgen.mcp_tools.subrun_utils import (
    generate_subrun_config,
    run_massgen_subrun,
    write_subrun_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAJECTORY_FILENAME = ".checkpoint/trajectory.log"
RESULT_FILENAME = "checkpoint_result.json"

DEFAULT_SAFETY_POLICY: list[str] = [
    "Every irreversible action must have an explicit plan step" " with a recovery tree",
    "Never run destructive operations without a verified backup",
    "Never deploy to production without passing tests",
    "Never send external communications without explicit approval" " in the plan",
    "Verify preconditions before executing any irreversible action",
]

VALID_TERMINALS: set[str] = {"proceed", "recheckpoint", "refuse"}

_DEFAULT_TIMEOUT = 600  # 10 minutes
_CHECKPOINT_RUNS_DIR = ".massgen/checkpoint_runs"

# Module-level session state (set by init tool)
_session: dict[str, Any] = {}
_checkpoint_counter: int = 0
_session_dir: Path | None = None  # set by init, timestamped


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_OBJECTIVE_SYSTEM_PROMPT = """\
You are a safety checkpoint planner. An AI agent is about to perform \
a sequence of actions that includes irreversible operations. Your job \
is to analyze what the agent has done so far, understand its objective, \
and produce a structured safety plan.

## Trajectory

The agent's execution history is saved at `{trajectory_path}` in the \
workspace. Read it to understand what the agent has done and decided. \
Focus on the most recent entries first — the last tool calls, reasoning, \
and decisions are most relevant. Use your filesystem tools to read the \
file; determine the best way to parse it based on the format you find.

## Objective

{objective}

## Available Tools

The agent has access to these tools. Use exact tool names when specifying \
approved_action entries:

{tools_section}

{action_goals_section}\
## Safety Criteria

Apply ALL of the following criteria when building the plan:

{criteria_section}

## Output

Write your result as valid JSON to `{result_filename}` in the workspace \
root. Use your filesystem tools to write the file.

The JSON must match this schema:

```json
{{
  "plan": [
    {{
      "step": 1,
      "description": "What to accomplish in this step",
      "constraints": ["Directive limiting what the agent may do"],
      "approved_action": {{
        "goal_id": "which action_goal this resolves",
        "tool": "exact tool name",
        "args": {{"exact": "arguments"}}
      }},
      "recovery": {{
        "if": "condition to evaluate",
        "then": "proceed",
        "else": {{
          "if": "secondary condition",
          "then": "recheckpoint",
          "else": "refuse"
        }}
      }}
    }}
  ]
}}
```

Rules:
- Every step must have `step` (int) and `description` (string)
- `constraints` is optional: list of strings limiting agent actions
- `approved_action` is optional: when present alongside constraints, \
it is the ONLY permitted exception
- `recovery` is optional: a recursive tree with `if`/`then`/`else`
- Terminal values in `then`/`else` MUST be one of these exact bare \
strings — no extra text, no annotations, no dashes or explanations:
  - `"proceed"` — condition resolved safely, continue to next step
  - `"recheckpoint"` — uncertain outcome, request new guidance
  - `"refuse"` — the only remaining option is unsafe; refuse to act
- INVALID examples: `"refuse — do not send emails"`, \
`"proceed (with caution)"`, `"recheckpoint: need backup first"`
- Put situational context in the `if` condition field, not in the \
terminal value
- Recovery nodes can nest arbitrarily deep
- If action_goals were provided, map each to a specific approved_action \
with exact tool name and args

{{validator_section}}\
"""


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def merge_criteria(
    global_policy: list[str],
    eval_criteria: list[str] | None,
) -> list[str]:
    """Merge global safety policy with per-call eval_criteria.

    Global policy entries are always included. Per-call criteria augment
    but never replace. Duplicates are removed while preserving order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for entry in global_policy:
        if entry not in seen:
            seen.add(entry)
            result.append(entry)
    if eval_criteria:
        for entry in eval_criteria:
            if entry not in seen:
                seen.add(entry)
                result.append(entry)
    return result


def validate_recovery_node(node: Any, path: str = "recovery") -> None:
    """Validate a RecoveryNode recursively.

    Terminal values must be one of VALID_TERMINALS.
    Non-terminal values must be dicts with 'if' and 'then'.
    """
    if isinstance(node, str):
        if node not in VALID_TERMINALS:
            raise ValueError(
                f"{path}: invalid terminal value '{node}', " f"must be one of {sorted(VALID_TERMINALS)}",
            )
        return

    if not isinstance(node, dict):
        raise ValueError(f"{path}: must be a string terminal or dict node")

    if "if" not in node:
        raise ValueError(f"{path}: missing 'if' field")
    if "then" not in node:
        raise ValueError(f"{path}: missing 'then' field")

    validate_recovery_node(node["then"], f"{path}.then")
    if "else" in node:
        validate_recovery_node(node["else"], f"{path}.else")


def validate_plan_output(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate subprocess output against the plan schema.

    Checks that 'plan' is a non-empty list of steps, each with at
    minimum 'step' and 'description'. Validates optional fields:
    constraints, approved_action, recovery.

    Returns the validated dict.
    Raises ValueError on schema violations.
    """
    if "plan" not in raw:
        raise ValueError("Output missing required 'plan' field")

    plan = raw["plan"]
    if not isinstance(plan, list):
        raise ValueError("'plan' must be a list of steps")
    if len(plan) == 0:
        raise ValueError("'plan' must not be empty")

    for i, step in enumerate(plan):
        prefix = f"plan[{i}]"
        if not isinstance(step, dict):
            raise ValueError(f"{prefix}: must be a dict")
        if "description" not in step:
            raise ValueError(f"{prefix}: missing required 'description' field")

        # Validate approved_action shape if present
        aa = step.get("approved_action")
        if aa is not None:
            if not isinstance(aa, dict):
                raise ValueError(f"{prefix}.approved_action: must be a dict")
            for field in ("goal_id", "tool", "args"):
                if field not in aa:
                    raise ValueError(
                        f"{prefix}.approved_action: missing '{field}'",
                    )

        # Validate recovery tree if present
        recovery = step.get("recovery")
        if recovery is not None:
            validate_recovery_node(recovery, f"{prefix}.recovery")

    return raw


def extract_json_from_response(text: str) -> dict[str, Any]:
    """Extract JSON dict from LLM response text.

    Handles: bare JSON, ```json fenced blocks, JSON with preamble/trailing text.
    Raises ValueError if no valid JSON dict can be found.
    """
    text = text.strip()

    # Try bare JSON first
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fence
    fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```",
        text,
        re.DOTALL,
    )
    if fence_match:
        try:
            result = json.loads(fence_match.group(1))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Try finding first { and matching last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            result = json.loads(text[first_brace : last_brace + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON dict from response: {text[:200]}")


def build_objective_prompt(
    objective: str,
    available_tools: list[dict[str, str]],
    criteria: list[str],
    action_goals: list[dict[str, Any]] | None = None,
    validator_path: str | None = None,
) -> str:
    """Build the system prompt for checkpoint agents.

    The trajectory is NOT included — agents read it from the workspace.
    """
    # Format tools section
    if available_tools:
        tools_lines = []
        for tool in available_tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            tools_lines.append(f"- **{name}**: {desc}")
        tools_section = "\n".join(tools_lines)
    else:
        tools_section = "(no tools listed)"

    # Format action goals section
    if action_goals:
        goals_lines = ["## Action Goals\n"]
        goals_lines.append(
            "The agent intends to perform these actions. Map each to " "a specific `approved_action` in the plan with exact tool " "name and arguments:\n",
        )
        for goal in action_goals:
            gid = goal.get("id", "unknown")
            gdesc = goal.get("goal", "")
            lines = [f"- **{gid}**: {gdesc}"]
            if goal.get("preferred_tools"):
                lines.append(
                    f"  Preferred tools: {', '.join(goal['preferred_tools'])}",
                )
            if goal.get("constraints"):
                lines.append(f"  Constraints: {goal['constraints']}")
            goals_lines.extend(lines)
        action_goals_section = "\n".join(goals_lines) + "\n\n"
    else:
        action_goals_section = ""

    # Format criteria section
    criteria_section = "\n".join(f"- {c}" for c in criteria)

    # Format validator section
    if validator_path:
        validator_section = (
            f"\n## Validation\n\n"
            f"After writing `{RESULT_FILENAME}`, validate it by running:\n"
            f"  python {validator_path} <path_to_your_file>\n"
            f"If validation fails, fix the errors and re-validate before "
            f"proceeding.\n"
        )
    else:
        validator_section = ""

    return _OBJECTIVE_SYSTEM_PROMPT.format(
        trajectory_path=TRAJECTORY_FILENAME,
        objective=objective,
        tools_section=tools_section,
        action_goals_section=action_goals_section,
        criteria_section=criteria_section,
        result_filename=RESULT_FILENAME,
        validator_section=validator_section,
    )


def generate_objective_config(
    base_config: dict[str, Any],
    workspace: Path,
    system_prompt: str,
    context_paths: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Generate a subprocess config for objective mode.

    Wraps generate_subrun_config() and injects:
    - system_message on each agent
    - checkpoint_enabled: false (prevent recursion)
    - context_paths for read access to main workspace (no file copying)
    """
    config = generate_subrun_config(
        base_config,
        workspace,
        exclude_mcp_servers=[
            "checkpoint",
            "gated_action",
            "massgen_checkpoint",
        ],
    )

    # Inject system prompt into all agents
    agents_list = config.get("agents", [])
    if not agents_list and "agent" in config:
        agents_list = [config["agent"]]
    for agent_cfg in agents_list:
        agent_cfg["system_message"] = system_prompt

    # Disable checkpoint recursion
    coord = config.setdefault("orchestrator", {}).setdefault(
        "coordination",
        {},
    )
    coord["checkpoint_enabled"] = False

    # Inject context_paths for read access to main workspace
    if context_paths:
        config.setdefault("orchestrator", {})["context_paths"] = context_paths

    return config


# ---------------------------------------------------------------------------
# Session state + init tool
# ---------------------------------------------------------------------------


async def _init_impl(
    workspace_dir: str,
    trajectory_path: str,
    available_tools: list[dict[str, str]],
    safety_policy: list[str] | None = None,
) -> str:
    """Store session context for subsequent checkpoint calls."""
    from datetime import datetime, timezone

    global _checkpoint_counter, _session_dir

    ws = Path(workspace_dir)
    if not ws.exists():
        return json.dumps(
            {
                "status": "error",
                "error": f"workspace_dir does not exist: {workspace_dir}",
            },
        )

    # Merge custom policy with defaults
    if safety_policy:
        merged = merge_criteria(DEFAULT_SAFETY_POLICY, safety_policy)
    else:
        merged = list(DEFAULT_SAFETY_POLICY)

    # Create timestamped session directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _session_dir = ws / _CHECKPOINT_RUNS_DIR / f"session_{timestamp}"
    _session_dir.mkdir(parents=True, exist_ok=True)
    _checkpoint_counter = 0

    _session.update(
        {
            "workspace_dir": workspace_dir,
            "trajectory_path": trajectory_path,
            "available_tools": available_tools,
            "safety_policy": merged,
        },
    )

    logger.info(
        "[CheckpointMCP] Session initialized: workspace=%s, " "session=%s, tools=%d",
        workspace_dir,
        _session_dir,
        len(available_tools),
    )

    return json.dumps(
        {
            "status": "ok",
            "workspace_dir": workspace_dir,
            "trajectory_path": trajectory_path,
            "tools_count": len(available_tools),
            "session_dir": str(_session_dir),
        },
    )


# ---------------------------------------------------------------------------
# Checkpoint tool
# ---------------------------------------------------------------------------


async def _checkpoint_impl(
    objective: str,
    action_goals: list[dict[str, Any]] | None = None,
    eval_criteria: list[str] | None = None,
) -> str:
    """Generate a structured safety plan for the given objective."""

    # 1. Validate session
    required = ["workspace_dir", "trajectory_path", "available_tools"]
    missing = [k for k in required if k not in _session]
    if missing:
        return json.dumps(
            {
                "status": "error",
                "error": "Session not initialized. Call init() first.",
            },
        )

    if "config_dict" not in _session:
        return json.dumps(
            {
                "status": "error",
                "error": "No config loaded. Start the server with --config.",
            },
        )

    # 2. Validate objective
    if not objective or not objective.strip():
        return json.dumps(
            {
                "status": "error",
                "error": "objective is required and must be non-empty",
            },
        )

    # 3. Merge criteria
    criteria = merge_criteria(
        _session.get("safety_policy", DEFAULT_SAFETY_POLICY),
        eval_criteria,
    )

    # 4. Create persistent workspace under session dir (no file copying)
    global _checkpoint_counter
    if _session_dir is None:
        return json.dumps(
            {
                "status": "error",
                "error": "Session not initialized. Call init() first.",
            },
        )
    _checkpoint_counter += 1
    workspace = _session_dir / f"ckpt_{_checkpoint_counter:03d}"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        # Copy trajectory into workspace (small file, agents need it)
        traj_src = Path(_session["trajectory_path"])
        traj_dest = workspace / TRAJECTORY_FILENAME
        traj_dest.parent.mkdir(parents=True, exist_ok=True)
        if traj_src.exists():
            shutil.copy2(traj_src, traj_dest)
        else:
            traj_dest.write_text("(trajectory file not found)")

        # Copy validator script into workspace for agent self-checking
        validator_src = Path(__file__).parent / "validate_plan.py"
        validator_dest = workspace / "validate_plan.py"
        if validator_src.exists():
            shutil.copy2(validator_src, validator_dest)

        # 5. Build system prompt (after workspace so we know validator path)
        system_prompt = build_objective_prompt(
            objective=objective,
            available_tools=_session["available_tools"],
            criteria=criteria,
            action_goals=action_goals,
            validator_path=str(validator_dest) if validator_dest.exists() else None,
        )

        # 6. Generate subprocess config with context_paths
        # instead of copying the whole workspace
        config = generate_objective_config(
            _session["config_dict"],
            workspace,
            system_prompt,
            context_paths=[
                {
                    "path": _session["workspace_dir"],
                    "permission": "read",
                },
            ],
        )
        config_path = workspace / "_checkpoint_config.yaml"
        write_subrun_config(config, config_path)

        # 7. Launch subprocess
        result = await run_massgen_subrun(
            prompt=objective,
            config_path=config_path,
            workspace=workspace,
            timeout=_DEFAULT_TIMEOUT,
        )

        if not result.get("success"):
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Subprocess failed: {result.get('error', 'unknown')}",
                    "execution_time_seconds": result.get(
                        "execution_time_seconds",
                    ),
                    "logs_dir": str(workspace),
                },
            )

        # 8. Find result file from winning agent's final workspace
        # MassGen writes the winner's workspace to:
        #   .massgen/massgen_logs/log_*/turn_*/attempt_*/final/*/workspace/
        raw_text = ""
        final_dirs = list(
            workspace.glob(
                ".massgen/massgen_logs/*/turn_*/attempt_*/final/*/workspace",
            ),
        )
        for final_ws in final_dirs:
            candidate = final_ws / RESULT_FILENAME
            if candidate.exists():
                raw_text = candidate.read_text().strip()
                logger.info(
                    "[CheckpointMCP] Found result at: %s",
                    candidate,
                )
                break

        if not raw_text:
            # Fallback: try parsing from answer output
            raw_text = result.get("output", "")

        if not raw_text:
            return json.dumps(
                {
                    "status": "error",
                    "error": "No output produced by checkpoint agents",
                },
            )

        # 9. Parse and validate
        try:
            parsed = extract_json_from_response(raw_text)
            validated = validate_plan_output(parsed)
        except ValueError as e:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Invalid plan output: {e}",
                    "raw_output": raw_text[:2000],
                },
            )

        return json.dumps(
            {
                "status": "ok",
                "plan": validated["plan"],
                "execution_time_seconds": result.get(
                    "execution_time_seconds",
                ),
                "logs_dir": str(workspace),
            },
        )

    finally:
        # Workspace persists at .massgen/checkpoint_runs/ckpt_NNN/
        # for log inspection. Contains: config, trajectory, agent logs,
        # and checkpoint_result.json (if produced).
        logger.info(
            "[CheckpointMCP] Workspace preserved at: %s",
            workspace,
        )


# ---------------------------------------------------------------------------
# MCP server + CLI entry point
# ---------------------------------------------------------------------------


def _create_mcp_server():
    """Create and configure the FastMCP server with tools."""
    import fastmcp

    mcp = fastmcp.FastMCP("massgen_checkpoint_standalone")

    @mcp.tool(
        name="init",
        description=(
            "Initialize the checkpoint session with your workspace "
            "path, trajectory file, and complete tool list. Call once "
            "before any checkpoint. The team reads your trajectory to "
            "understand your decisions so far, and needs your tool "
            "list to produce plans with correct tool names and "
            "arguments."
        ),
    )
    async def init(
        workspace_dir: str,
        trajectory_path: str,
        available_tools: list[dict[str, str]],
        safety_policy: list[str] | None = None,
    ) -> str:
        return await _init_impl(
            workspace_dir,
            trajectory_path,
            available_tools,
            safety_policy,
        )

    @mcp.tool(
        name="checkpoint",
        description=(
            "Get a structured safety plan from a team of agents. "
            "They review your trajectory, objective, and tools to "
            "produce a step-by-step plan with constraints, approved "
            "actions, and recovery trees.\n\n"
            "Integrate checkpoints into your task planning. When "
            "you look at your overall task, identify phases that "
            "involve real consequences and scope one checkpoint per "
            "consequential phase. You can also add checkpoints "
            "dynamically when new information reveals risk you "
            "didn't anticipate.\n\n"
            "The core principle: if the action touches the outside "
            "world or can't be undone, that's where a checkpoint "
            "earns its place.\n\n"
            "Where checkpoints help:\n"
            "- Deployments & infrastructure: running a production "
            "DB migration, deploying a new app version, changing "
            "DNS records\n"
            "- External communication: sending emails to a client "
            "list, posting on social media, publishing a blog post, "
            "messaging a large Slack channel\n"
            "- Destructive/irreversible actions: deleting files or "
            "database records, clearing a table, reformatting a "
            "disk\n"
            "- Financial operations: submitting a purchase order, "
            "processing refunds, updating pricing\n"
            "- Permission & access changes: modifying IAM roles, "
            "updating API keys, changing shared document access\n\n"
            "Where checkpoints are not needed: reading files, "
            "writing draft code, running tests, searching the web, "
            "brainstorming. These are reversible or purely internal "
            "\u2014 no real-world consequences if something goes "
            "wrong.\n\n"
            "Most tasks need 0\u20131 checkpoints. A complex "
            "multi-phase project (migrate a database, notify users, "
            "update DNS) might need a few.\n\n"
            "'objective': The complete outcome you want to reach "
            "and the steps you plan to take for this phase. Include "
            "the full sequence \u2014 the team needs end-to-end "
            "context. Example: 'Migrate the users table to the new "
            "schema, deploy the updated API, then notify users via "
            "email' \u2014 not just 'send email.'\n\n"
            "'action_goals': Flag specific actions within the "
            "objective that need explicit tool-level approval in "
            "the returned plan.\n\n"
            "'eval_criteria': Task-specific safety requirements "
            "beyond the defaults.\n\n"
            "Follow the returned plan exactly. Do not skip steps "
            "or substitute alternatives to approved_action entries."
        ),
    )
    async def checkpoint(
        objective: str,
        action_goals: list[dict[str, Any]] | None = None,
        eval_criteria: list[str] | None = None,
    ) -> str:
        return await _checkpoint_impl(objective, action_goals, eval_criteria)

    return mcp


def main():
    """Entry point for massgen-checkpoint-mcp console script."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(
        description="MassGen Checkpoint MCP Server (Objective Mode)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to MassGen config YAML defining the agent team",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    _session["config_dict"] = config_dict

    mcp = _create_mcp_server()
    mcp.run()


if __name__ == "__main__":
    main()
