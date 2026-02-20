"""
Shared plan execution setup logic for CLI and TUI.

This module provides reusable functions for preparing plan execution,
ensuring both CLI (--execute-plan) and TUI execute mode use the same logic.
"""

import copy
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .plan_storage import PlanMetadata, PlanSession

logger = logging.getLogger(__name__)


COMPLETED_TASK_STATUSES = {"completed", "verified"}
PROGRESS_TASK_STATUSES = {"in_progress", "completed", "verified"}


class PlanValidationError(ValueError):
    """Raised when a plan is invalid for chunked execution."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_dependency_ids(task: dict[str, Any]) -> list[str]:
    """Return normalized dependency IDs from either depends_on or dependencies."""
    raw = task.get("depends_on")
    if raw is None:
        raw = task.get("dependencies", [])

    if not isinstance(raw, list):
        return []

    deps: list[str] = []
    for dep in raw:
        if dep is None:
            continue
        dep_str = str(dep).strip()
        if dep_str:
            deps.append(dep_str)
    return deps


def _get_task_chunk(task: dict[str, Any]) -> str:
    chunk = task.get("chunk", "")
    if chunk is None:
        return ""
    return str(chunk).strip()


def _chunk_archive_filename(chunk_label: str) -> str:
    """Generate stable archive filename for a completed chunk task plan."""
    chunk = chunk_label.strip()
    match = re.match(r"^(c\d+)", chunk, flags=re.IGNORECASE)
    if match:
        return f"tasks_{match.group(1).lower()}.json"

    safe = re.sub(r"[^a-zA-Z0-9]+", "_", chunk).strip("_").lower()
    if not safe:
        safe = "chunk"
    return f"tasks_{safe}.json"


def _archive_existing_operational_plan(
    tasks_dir: Path,
    next_chunk: str | None,
) -> None:
    """
    Archive existing tasks/plan.json before writing the next chunk plan.

    Stores a snapshot as tasks/tasks_cXX.json when possible, so prior chunk
    task state remains inspectable while tasks/plan.json stays canonical.
    """
    plan_file = tasks_dir / "plan.json"
    if not plan_file.exists():
        return

    try:
        existing_payload = json.loads(plan_file.read_text())
    except json.JSONDecodeError:
        logger.warning(
            "[PlanExecution] Skipping plan archive: existing %s is not valid JSON",
            plan_file,
        )
        return

    if not isinstance(existing_payload, dict):
        return

    existing_scope = existing_payload.get("execution_scope", {})
    if not isinstance(existing_scope, dict):
        return

    existing_chunk_raw = existing_scope.get("active_chunk")
    existing_chunk = str(existing_chunk_raw).strip() if existing_chunk_raw is not None else ""
    if not existing_chunk:
        return

    if next_chunk and existing_chunk == next_chunk:
        # Same chunk; avoid self-archiving on repeated setup calls.
        return

    archive_file = tasks_dir / _chunk_archive_filename(existing_chunk)
    if archive_file.exists():
        return

    archive_file.write_text(json.dumps(existing_payload, indent=2))
    logger.info(
        "[PlanExecution] Archived previous operational plan chunk '%s' to %s",
        existing_chunk,
        archive_file,
    )


def load_frozen_plan(plan_session: "PlanSession") -> dict[str, Any]:
    """Load frozen plan data from disk."""
    frozen_plan_file = plan_session.frozen_dir / "plan.json"
    if not frozen_plan_file.exists():
        raise FileNotFoundError(f"Frozen plan not found at {frozen_plan_file}")

    try:
        plan_data = json.loads(frozen_plan_file.read_text())
    except json.JSONDecodeError as e:
        raise PlanValidationError(f"Frozen plan is invalid JSON: {e}") from e

    if not isinstance(plan_data, dict):
        raise PlanValidationError("Frozen plan must be a JSON object")

    return plan_data


def validate_chunked_plan(
    plan_data: dict[str, Any],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """
    Validate plan data for planner-defined chunked execution.

    Returns:
        Tuple of (chunk_order, tasks_by_chunk)
    """
    tasks = plan_data.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise PlanValidationError("Plan must contain a non-empty 'tasks' list")

    errors: list[str] = []
    task_ids: dict[str, int] = {}
    task_chunk_by_id: dict[str, str] = {}
    chunk_order: list[str] = []
    tasks_by_chunk: dict[str, list[dict[str, Any]]] = {}

    # Pass 1: per-task shape + chunk presence + deterministic chunk order.
    for idx, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            errors.append(f"task[{idx}] must be an object")
            continue

        task_id_raw = task.get("id")
        task_id = str(task_id_raw).strip() if task_id_raw is not None else ""
        if not task_id:
            errors.append(f"task[{idx}] is missing non-empty 'id'")
            continue
        if task_id in task_ids:
            errors.append(f"duplicate task id '{task_id}'")
            continue
        task_ids[task_id] = idx

        chunk = _get_task_chunk(task)
        if not chunk:
            errors.append(f"task '{task_id}' is missing non-empty 'chunk'")
            continue

        task_chunk_by_id[task_id] = chunk
        if chunk not in tasks_by_chunk:
            tasks_by_chunk[chunk] = []
            chunk_order.append(chunk)
        tasks_by_chunk[chunk].append(task)

    # Pass 2: dependency checks (existence and chunk ordering).
    chunk_index = {chunk: i for i, chunk in enumerate(chunk_order)}
    for task in tasks:
        if not isinstance(task, dict):
            continue

        task_id_raw = task.get("id")
        task_id = str(task_id_raw).strip() if task_id_raw is not None else ""
        if not task_id or task_id not in task_chunk_by_id:
            continue

        task_chunk = task_chunk_by_id[task_id]
        for dep_id in _normalize_dependency_ids(task):
            if dep_id not in task_ids:
                errors.append(f"task '{task_id}' depends on unknown task '{dep_id}'")
                continue

            dep_chunk = task_chunk_by_id.get(dep_id)
            if dep_chunk is None:
                continue

            # A task may depend on same or earlier chunk, never a future chunk.
            if chunk_index[dep_chunk] > chunk_index[task_chunk]:
                errors.append(
                    f"task '{task_id}' in chunk '{task_chunk}' depends on future " f"chunk '{dep_chunk}' via task '{dep_id}'",
                )

    if errors:
        bullet_list = "\n".join(f"- {err}" for err in errors)
        raise PlanValidationError(
            "Plan validation failed for chunked execution:\n" + bullet_list,
        )

    return chunk_order, tasks_by_chunk


def get_next_pending_chunk(metadata: "PlanMetadata") -> str | None:
    """Return the next pending chunk from metadata."""
    chunk_order = metadata.chunk_order or []
    completed = set(metadata.completed_chunks or [])
    for chunk in chunk_order:
        if chunk not in completed:
            return chunk
    return None


def initialize_chunk_execution_state(plan_session: "PlanSession") -> "PlanMetadata":
    """Validate plan and initialize metadata fields used by chunked execution."""
    metadata = plan_session.load_metadata()
    plan_data = load_frozen_plan(plan_session)
    chunk_order, _ = validate_chunked_plan(plan_data)

    metadata.execution_mode = "chunked_by_planner_v1"
    metadata.chunk_order = chunk_order
    metadata.completed_chunks = [chunk for chunk in (metadata.completed_chunks or []) if chunk in set(chunk_order)]
    metadata.chunk_history = metadata.chunk_history or []
    metadata.planning_feedback_history = metadata.planning_feedback_history or []

    current_chunk = metadata.current_chunk
    if not current_chunk or current_chunk not in chunk_order:
        current_chunk = get_next_pending_chunk(metadata)
    elif current_chunk in set(metadata.completed_chunks or []):
        current_chunk = get_next_pending_chunk(metadata)

    metadata.current_chunk = current_chunk

    # Ensure pre-execution plans are ready unless already completed/failed/resumable.
    if metadata.status in {"planning", "ready", "resumable", "executing", "completed", "failed"}:
        pass
    else:
        metadata.status = "ready"

    if current_chunk is None and chunk_order:
        metadata.status = "completed"

    plan_session.save_metadata(metadata)
    return metadata


def resolve_active_chunk(
    plan_session: "PlanSession",
    requested_chunk: str | None = None,
) -> tuple["PlanMetadata", list[str]]:
    """Resolve and persist the active chunk, optionally overriding with user selection."""
    metadata = initialize_chunk_execution_state(plan_session)
    chunk_order = metadata.chunk_order or []

    if requested_chunk:
        requested = requested_chunk.strip()
        if requested not in chunk_order:
            raise PlanValidationError(
                f"Unknown chunk selection '{requested}'. Available: {', '.join(chunk_order)}",
            )
        metadata.current_chunk = requested
        plan_session.save_metadata(metadata)
        return metadata, chunk_order

    if not metadata.current_chunk:
        metadata.current_chunk = get_next_pending_chunk(metadata)
        plan_session.save_metadata(metadata)

    return metadata, chunk_order


def evaluate_chunk_progress(
    chunk_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute completion/progress statistics for a chunk."""
    total = len(chunk_tasks)
    completed_ids: list[str] = []
    progressed_ids: list[str] = []

    for task in chunk_tasks:
        task_id = str(task.get("id", "")).strip()
        status = str(task.get("status", "pending")).strip().lower()
        if status in COMPLETED_TASK_STATUSES:
            if task_id:
                completed_ids.append(task_id)
        if status in PROGRESS_TASK_STATUSES:
            if task_id:
                progressed_ids.append(task_id)

    completed_count = len(completed_ids)
    progressed_count = len(progressed_ids)

    return {
        "total_tasks": total,
        "completed_count": completed_count,
        "progressed_count": progressed_count,
        "completed_task_ids": completed_ids,
        "progressed_task_ids": progressed_ids,
        "is_complete": total > 0 and completed_count == total,
        "made_progress": progressed_count > 0,
    }


def record_chunk_checkpoint(
    plan_session: "PlanSession",
    *,
    chunk: str,
    status: str,
    attempt: int,
    progress: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> "PlanMetadata":
    """Persist chunk checkpoint metadata and append chunk history entry."""
    metadata = plan_session.load_metadata()
    metadata.chunk_history = metadata.chunk_history or []
    metadata.completed_chunks = metadata.completed_chunks or []

    entry: dict[str, Any] = {
        "chunk": chunk,
        "status": status,
        "attempt": attempt,
        "timestamp": _utc_now_iso(),
    }
    if progress:
        entry["progress"] = progress
    if error_message:
        entry["error"] = error_message
    metadata.chunk_history.append(entry)

    if status == "completed" and chunk not in metadata.completed_chunks:
        metadata.completed_chunks.append(chunk)

    next_chunk = get_next_pending_chunk(metadata)
    metadata.current_chunk = next_chunk

    if status == "failed":
        metadata.status = "failed"
    elif next_chunk is None and metadata.chunk_order:
        metadata.status = "completed"
        metadata.resumable_state = None
    else:
        metadata.status = "executing"

    plan_session.save_metadata(metadata)
    plan_session.log_event(
        "chunk_checkpoint",
        {
            "chunk": chunk,
            "status": status,
            "attempt": attempt,
            "progress": progress or {},
            "error": error_message,
        },
    )
    return metadata


def mark_session_resumable(
    plan_session: "PlanSession",
    *,
    current_chunk: str | None,
    reason: str,
    retry_counts: dict[str, int] | None = None,
) -> "PlanMetadata":
    """Mark the plan session as resumable with the latest checkpoint pointer."""
    metadata = plan_session.load_metadata()
    metadata.status = "resumable"
    metadata.current_chunk = current_chunk or metadata.current_chunk
    metadata.resumable_state = {
        "marked_at": _utc_now_iso(),
        "current_chunk": metadata.current_chunk,
        "reason": reason,
        "retry_counts": retry_counts or {},
    }
    plan_session.save_metadata(metadata)
    plan_session.log_event(
        "execution_resumable",
        {
            "current_chunk": metadata.current_chunk,
            "reason": reason,
            "retry_counts": retry_counts or {},
        },
    )
    return metadata


# Plan execution guidance injected into agent system messages
PLAN_EXECUTION_GUIDANCE = """
## Plan Execution Mode

You are executing a pre-approved task plan. The plan has been AUTO-LOADED into `tasks/plan.json`.

### Getting Started - Plan is Ready

Your task plan is already loaded. Use MCP planning tools to track progress:

1. **See all tasks**: `get_task_plan()` - view full plan with current status
2. **See ready tasks**: `get_ready_tasks()` - tasks with dependencies satisfied
3. **Start a task**: `update_task_status("T001", "in_progress")`
4. **Complete a task**: `update_task_status("T001", "completed", "How you completed it")`

Supporting docs from planning phase are in `planning_docs/` for reference.

### CRITICAL: Verification Workflow

**Do NOT just write code and mark tasks complete. You MUST verify from the user perspective first, then confirm technical correctness.**

#### Task Status Flow
- `pending` → `in_progress` → `completed` → `verified`
- **completed**: Implementation is done (code written)
- **verified**: Task has been tested and confirmed working

#### How to Use Verification
1. Mark task `completed` when implementation is done
2. Verify user-visible behavior first (run it, use it, inspect output quality)
3. At logical checkpoints, verify groups of completed tasks together
4. Mark tasks `verified` after verification passes

#### Verification Checkpoints (when to verify)
Tasks have `verification_group` labels (e.g., "foundation", "frontend_ui", "api"). Verify when:

1. **After completing all tasks in a verification_group** - e.g., after all "foundation" tasks, run `npm run dev`
2. **After major milestones** - e.g., project setup, feature completion
3. **Before declaring work complete** - Run full build (`npm run build`)

Use `get_task_plan()` to see tasks grouped by `verification_group` under `verification_groups`.

#### Verification Commands
Tasks have `verification_method` in metadata - USE IT as guidance:
```
update_task_status("F001", "completed", "Created Next.js project")
# ... complete more foundation tasks ...
# Verify user-visible behavior first:
# npm run dev, open home page, confirm hero/nav/footer render correctly on mobile + desktop
# Then run technical checks:
# npm run build
update_task_status("F001", "verified", "Home page renders correctly in browser and build passes")
```

**A task should NOT be marked `verified` if:**
- User-visible behavior was not actually exercised
- The code doesn't compile/build
- The dev server crashes on startup
- The feature doesn't render or function as described

Fix issues before marking as verified.

### Evaluating CURRENT_ANSWERS

When you see other agents' work, you'll receive **progress stats** showing task completion.
These are INFORMATIONAL only - they help you understand where others are, but task count alone
doesn't determine quality.

**Focus on Deliverable Quality** (the end product matters most):
- Does the deliverable work? (website loads, app runs, API responds)
- Does it meet the original requirements from the planning docs?
- Is the user-facing quality good? (UI looks right, features work as expected)

**Progress stats are context, not judgment**:
- An agent with fewer tasks completed might have better quality work
- An agent with all tasks done might have rushed and produced poor quality
- Use progress info to understand scope, but evaluate the actual deliverable

**Only vote when work is TRULY COMPLETE and HIGH QUALITY**:
- All planned tasks should be done (or have documented reasons for deviation)
- The deliverable must be functional and meet quality expectations
- Don't vote for partial implementations, even if task count looks good

### Adopting Another Agent's Work

If you see a CURRENT_ANSWER that's excellent and you want to build on it:
1. Their plan progress is in their `tasks/plan.json`
2. To adopt: copy their plan.json content into YOUR `tasks/plan.json` via `create_task_plan(tasks=[...])`
3. Then continue from where they left off

If no agent is fully complete with quality work, continue your own implementation rather than voting for incomplete work.
"""


def prepare_plan_execution_config(
    config: dict[str, Any],
    plan_session: "PlanSession",
) -> dict[str, Any]:
    """
    Prepare config for plan execution (used by both CLI and TUI).

    Modifies config to:
    1. Add frozen plan as read-only context path
    2. Enable planning MCP tools for task tracking
    3. Inject plan execution guidance into agents' system messages
    """
    exec_config = copy.deepcopy(config)

    orchestrator_cfg = exec_config.setdefault("orchestrator", {})
    context_paths = orchestrator_cfg.setdefault("context_paths", [])
    if not isinstance(context_paths, list):
        context_paths = []
        orchestrator_cfg["context_paths"] = context_paths

    # Restore context paths from planning phase (if stored in metadata)
    try:
        metadata = plan_session.load_metadata()
        if metadata.context_paths:
            context_paths.extend(metadata.context_paths)
            logger.info(
                f"[PlanExecution] Restored {len(metadata.context_paths)} context paths from planning phase",
            )
    except Exception as e:
        logger.warning(f"[PlanExecution] Could not load context paths from metadata: {e}")

    # Add frozen plan as read-only context if not already present.
    frozen_path = str(plan_session.frozen_dir.resolve())
    normalized_existing = set()
    for ctx in context_paths:
        path = ctx.get("path") if isinstance(ctx, dict) else None
        if not path:
            continue
        try:
            normalized_existing.add(str(Path(path).resolve()))
        except Exception:
            normalized_existing.add(str(path))
    if frozen_path not in normalized_existing:
        context_paths.append({"path": frozen_path, "permission": "read"})

    coordination_cfg = orchestrator_cfg.setdefault("coordination", {})
    coordination_cfg["enable_agent_task_planning"] = True
    coordination_cfg["task_planning_filesystem_mode"] = True

    # Inject plan execution guidance into agents' system messages.
    if "agents" in exec_config:
        for agent_cfg in exec_config["agents"]:
            existing_msg = agent_cfg.get("system_message", "")
            agent_cfg["system_message"] = existing_msg + PLAN_EXECUTION_GUIDANCE
    elif "agent" in exec_config:
        agent_cfg = exec_config["agent"]
        existing_msg = agent_cfg.get("system_message", "")
        agent_cfg["system_message"] = existing_msg + PLAN_EXECUTION_GUIDANCE

    return exec_config


def setup_agent_workspaces_for_execution(
    agents: dict[str, Any],
    plan_session: "PlanSession",
    active_chunk: str | None = None,
) -> int:
    """
    Copy plan and supporting docs to each agent's workspace.

    If active_chunk is provided (or available in session metadata), writes a
    chunk-only operational plan to tasks/plan.json.

    Returns:
        Number of tasks in the operational plan.
    """
    try:
        metadata, chunk_order = resolve_active_chunk(
            plan_session,
            requested_chunk=active_chunk,
        )
        plan_data = load_frozen_plan(plan_session)
        _, tasks_by_chunk = validate_chunked_plan(plan_data)
    except (FileNotFoundError, PlanValidationError) as e:
        logger.error(f"[PlanExecution] Cannot prepare execution workspace: {e}")
        return 0

    plan_tasks = plan_data.get("tasks", [])
    selected_chunk = metadata.current_chunk
    if selected_chunk:
        operational_tasks = tasks_by_chunk.get(selected_chunk, [])
    else:
        operational_tasks = plan_tasks

    operational_plan = dict(plan_data)
    operational_plan["tasks"] = operational_tasks
    operational_plan["execution_scope"] = {
        "mode": "chunked_by_planner_v1",
        "active_chunk": selected_chunk,
        "chunk_order": chunk_order,
        "completed_chunks": metadata.completed_chunks or [],
    }

    task_count = len(operational_tasks)
    if task_count == 0:
        logger.warning(
            f"[PlanExecution] No tasks found for active chunk: {selected_chunk or '(none)'}",
        )

    for agent_id, agent in agents.items():
        if not (hasattr(agent.backend, "filesystem_manager") and agent.backend.filesystem_manager):
            continue

        agent_workspace = Path(agent.backend.filesystem_manager.cwd)

        planning_docs_dest = agent_workspace / "planning_docs"
        planning_docs_dest.mkdir(exist_ok=True)
        for doc in plan_session.frozen_dir.glob("*.md"):
            shutil.copy2(doc, planning_docs_dest / doc.name)
            logger.info(f"[PlanExecution] Copied {doc.name} to {agent_id}'s planning_docs/")

        # Keep full frozen plan available in workspace as read-only reference copy.
        full_plan_reference = planning_docs_dest / "full_plan.json"
        full_plan_reference.write_text(json.dumps(plan_data, indent=2))

        tasks_dir = agent_workspace / "tasks"
        tasks_dir.mkdir(exist_ok=True)
        _archive_existing_operational_plan(tasks_dir, selected_chunk)
        plan_file = tasks_dir / "plan.json"
        plan_file.write_text(json.dumps(operational_plan, indent=2))
        logger.info(
            "[PlanExecution] Wrote operational plan to %s (%d tasks, chunk=%s)",
            plan_file,
            task_count,
            selected_chunk or "all",
        )

    return task_count


def build_execution_prompt(
    question: str,
    active_chunk: str | None = None,
    chunk_order: list[str] | None = None,
) -> str:
    """
    Build the execution prompt that guides agents through plan-based work.

    Args:
        question: The original user question/task.
        active_chunk: Optional active chunk label for this execution turn.
        chunk_order: Optional ordered chunk list for context.
    """
    chunk_scope_lines = []
    if active_chunk:
        chunk_scope_lines.append(f"- Active chunk: `{active_chunk}`")
        if chunk_order:
            chunk_scope_lines.append(f"- Chunk order: {', '.join(chunk_order)}")
        chunk_scope_lines.append("- Execute only tasks in this active chunk for this turn")
    else:
        chunk_scope_lines.append("- Execute the loaded plan scope in `tasks/plan.json`")

    chunk_scope = "\n".join(chunk_scope_lines)

    return f"""# PLAN EXECUTION MODE

Your task plan has been AUTO-LOADED into `tasks/plan.json`. Start executing!

## Your Task
{question}

## Active Scope
{chunk_scope}

## Getting Started

1. **Check ready tasks**: Use `get_ready_tasks()` to see what to work on first
2. **Track progress**: Use `update_task_status(task_id, status, completion_notes)` as you work
3. **Execute the loaded tasks**: Implement everything in the currently loaded scope
4. **Evaluate others**: See system prompt for how to assess CURRENT_ANSWERS

## Reference Materials
- `planning_docs/` - supporting docs from planning phase (user stories, design, etc.)
- `planning_docs/full_plan.json` - frozen full plan for read-only reference
- Frozen plan available via read-only context path for validation

Begin execution now."""
