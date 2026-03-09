---
name: massgen-evaluator
description: Get multi-agent evaluation and critical feedback on your current work by invoking MassGen. Use whenever you've iterated enough and need outside perspective, before submitting PRs, or when you want diverse AI evaluators to critique code, documents, or artifacts in your working directory. Use this skill when you've stalled, need a second opinion, want pre-PR review, or need critical evaluation of any work artifact.
---

# MassGen Evaluator

Invoke MassGen's multi-agent evaluation to get diverse, critical feedback on your work. Multiple AI agents independently evaluate your artifacts and converge on the strongest critique through MassGen's checklist-gated voting system.

## When to Use

- After iterating and stalling — need outside perspective
- Before submitting PRs or delivering artifacts
- When wanting diverse AI perspectives on implementation quality
- Whenever you've self-improved as much as you can alone

## Prerequisites

1. **Check if massgen is installed**:
   ```bash
   uv run massgen --help 2>/dev/null || massgen --help 2>/dev/null
   ```

2. **If not installed**:
   ```bash
   pip install massgen
   # or
   uv pip install massgen
   ```

3. **If no config exists** (`.massgen/config.yaml`), run headless quickstart:
   ```bash
   uv run massgen --quickstart --headless
   ```
   This auto-detects API keys, selects the best backend, generates config,
   and installs Docker/skills. If no API keys are found, it creates a
   `.env` template for the user to fill in.

## Workflow

### Step 0: Create Evaluation Directory

Create a timestamped subdirectory so parallel evaluations don't conflict:

```bash
EVAL_DIR=".massgen/eval/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_DIR"
```

All evaluation artifacts (context, criteria, prompt, output, logs) go in this directory.

### Step 1: Write the Context File

Write `$EVAL_DIR/context.md` with structured context about the work being evaluated.

**What you provide** (evaluators cannot infer this):
- Requirements and acceptance criteria
- File structure overview
- Git info (diff stats, commits, branch)
- Areas of concern
- Verification results (test output, build logs, lint results)

**What evaluators discover on their own** (from reading the cwd):
- Code quality issues, bugs, antipatterns
- Architecture assessment
- Test coverage gaps
- Security vulnerabilities
- Alignment between requirements and implementation

**Context file template:**

```markdown
## Original Task
<what the user asked for — the requirements>

## What Was Done
<summary of implementation work completed>

## File Structure
<relevant directory tree / key files overview>

## Git Info
<git diff --stat, recent commits, branch info>
<for patches: include actual diff or key changed files>

## Requirements / Acceptance Criteria
<what 'done' looks like — explicit success criteria>

## Areas of Concern
<where you're stuck or unsure — optional but valuable>

## Verification Results
<test output, build results, lint output, etc.>

## Key Files to Examine
<most important files for evaluators to focus on>
```

### Step 2: Generate Evaluation Criteria

Read the criteria writing guide at `references/eval_criteria_guide.md`
(relative to this skill).

Based on the task context from Step 1, generate 4-7 evaluation criteria
as a JSON file. Your criteria should be task-specific, concrete, and
scoreable — following the tier system from the guide.

If there's a specific evaluation focus (security, performance, architecture,
test coverage, code quality), weight your criteria toward that focus area.
In Claude Code: use AskUserQuestion to ask the user for focus preference.
In Codex or non-interactive: default to general coverage.

```bash
cat > $EVAL_DIR/criteria.json << 'EOF'
[
  {"text": "...", "category": "must"},
  {"text": "...", "category": "must"},
  {"text": "...", "category": "should"},
  {"text": "...", "category": "should", "verify_by": "..."},
  {"text": "...", "category": "could"}
]
EOF
```

These criteria are passed to MassGen via `--eval-criteria` and used by
the checklist-gated voting system to evaluate agent answers. This is
faster and more precise than MassGen's internal criteria generation
subagent, because you already have the full task context.

### Step 3: Construct the Evaluation Prompt

1. Read the prompt template from `references/evaluation_prompt_template.md` (relative to this skill)
2. Read the context file you wrote in Step 1
3. Replace `{{CONTEXT_FILE_CONTENT}}` with the context file contents
4. Replace `{{CUSTOM_FOCUS}}` with the focus directive (or empty string for General)
5. Write the final prompt to `$EVAL_DIR/prompt.md`

### Step 4: Run MassGen (in background) and Open Viewer

Launch MassGen in the background and open the web viewer so the user
can observe evaluation progress in their browser.

**4a. Start MassGen in background, capturing the log directory:**

```bash
uv run massgen --automation \
  --no-parse-at-references \
  --cwd-context ro \
  --eval-criteria $EVAL_DIR/criteria.json \
  --output-file $EVAL_DIR/result.md \
  "$(cat $EVAL_DIR/prompt.md)" \
  > $EVAL_DIR/output.log 2>&1 &
MASSGEN_PID=$!
```

Use your agent's native background execution mechanism
(e.g., `run_in_background` in Claude Code).

**4b. Extract the log directory and launch the web viewer:**

The automation output's first line is `LOG_DIR: <path>`. Wait briefly
for it to appear, then launch the viewer:

```bash
# Wait for LOG_DIR to be written (usually < 2 seconds)
for i in $(seq 1 10); do
  LOG_DIR=$(grep -m1 '^LOG_DIR:' $EVAL_DIR/output.log 2>/dev/null | cut -d' ' -f2)
  [ -n "$LOG_DIR" ] && break
  sleep 1
done

if [ -n "$LOG_DIR" ]; then
  # Launch web viewer — automatically opens browser for the user
  uv run massgen viewer "$LOG_DIR" --web &
fi
```

The viewer automatically opens `http://localhost:8000` in the user's
browser, showing live agent rounds, voting, and convergence as they
happen. No need to open the browser yourself — it launches automatically.

**Flags explained:**
- `--automation`: clean parseable output, no TUI
- `--no-parse-at-references`: prevents MassGen from interpreting `@path` in the prompt text
- `--cwd-context ro`: gives evaluators read-only access to the current working directory
- `--eval-criteria`: passes your task-specific criteria JSON (overrides any YAML inline criteria)
- `--output-file`: writes the winning evaluator's answer to a parseable file

No `--config` flag — uses the default config from `.massgen/config.yaml`.

**Timing:** expect 2-10 minutes for standard evaluations, 10-30 minutes for complex tasks.

### Step 5: Parse the Output

Read `$EVAL_DIR/result.md` and extract:

1. **Verdict**: ITERATE or CONVERGED (from the Evaluation Summary section at the bottom)
2. **Top Improvements**: ordered by impact, with `concrete_steps`
3. **Preserve**: elements to keep unchanged
4. **Next Steps**: concrete actions with specific techniques
5. **Prior attempt diagnosis**: what was tried before and why it didn't work

### Step 6: Apply the Feedback

Based on the verdict:

- **ITERATE**: Apply the improvements from the `improvement_spec` section, following `concrete_steps` and `execution_order`. Pay special attention to `prior_attempt_awareness` — if the evaluators identified failed approaches, do NOT retry them.
- **CONVERGED**: The work meets the quality bar. Proceed to delivery.

## Output Structure Reference

The evaluation output contains these sections (all inline in the answer):

| Section | Purpose |
|---|---|
| `criteria_interpretation` | What each requirement really demands |
| `criterion_findings` | Where the work falls short, with evidence |
| `cross_answer_synthesis` | Strongest dimensions, gaps, what improvement looks like |
| `unexplored_approaches` | 1-3 fresh ideas nobody tried yet |
| `preserve` | What must survive into the next revision |
| `improvement_spec` | Design spec with `concrete_steps` per criterion |
| `verification_plan` | Checks to rerun after implementation |
| `evidence_gaps` | Missing evidence that limited the critique |
| **Evaluation Summary** | Quick-reference: verdict, top improvements, preserve, next steps |

## Example Invocations

### Pre-PR Code Review

```bash
# Create eval directory
EVAL_DIR=".massgen/eval/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_DIR"

# Write context
cat > $EVAL_DIR/context.md << 'EOF'
## Original Task
Add WebSocket support for real-time agent status updates

## What Was Done
Implemented WebSocket server in websocket_handler.py, client in webui/src/hooks/useAgentStatus.ts

## Git Info
Branch: feat/websocket-status (12 commits ahead of main)
Key files changed: websocket_handler.py, useAgentStatus.ts, types.ts

## Requirements / Acceptance Criteria
- WebSocket connection auto-reconnects on disconnect
- Status updates arrive within 500ms of agent state change
- Works with 3+ concurrent agent sessions

## Verification Results
pytest: 47 passed, 0 failed
vitest: 12 passed, 0 failed

## Key Files to Examine
massgen/websocket_handler.py, webui/src/hooks/useAgentStatus.ts
EOF

# Write criteria
cat > $EVAL_DIR/criteria.json << 'EOF'
[
  {"text": "Reconnection reliability: WebSocket auto-reconnects within 5s of disconnect with exponential backoff, no message loss during reconnect window.", "category": "must"},
  {"text": "Latency requirement: status updates arrive at the client within 500ms of agent state change under normal load.", "category": "must"},
  {"text": "Concurrency: system handles 3+ simultaneous agent sessions without message cross-contamination or dropped updates.", "category": "must"},
  {"text": "Error handling: connection failures, malformed messages, and server errors produce clear client-side feedback without crashing the UI.", "category": "should"},
  {"text": "Code quality: WebSocket handler and client hook have clean separation of concerns, no duplicated state management, and consistent error patterns.", "category": "should"}
]
EOF

# Build prompt (fill template, then invoke in background)
# ... (follow Step 3 to construct prompt from template)
uv run massgen --automation --no-parse-at-references --cwd-context ro \
  --eval-criteria $EVAL_DIR/criteria.json \
  --output-file $EVAL_DIR/result.md \
  "$(cat $EVAL_DIR/prompt.md)" \
  > $EVAL_DIR/output.log 2>&1 &

# Extract LOG_DIR and open web viewer for the user
sleep 2
LOG_DIR=$(grep -m1 '^LOG_DIR:' $EVAL_DIR/output.log | cut -d' ' -f2)
uv run massgen viewer "$LOG_DIR" --web &
```

### Architecture Evaluation

Write a context file focused on architecture decisions, generate criteria weighted toward design patterns and extensibility, and invoke as above.

### Post-Implementation Quality Check

After completing a feature, write a context file summarizing what was done and generate general-purpose criteria to catch issues across all dimensions.

## Reference Files

- `references/eval_criteria_guide.md` — how to write good evaluation criteria (format, tiers, examples)
- `references/evaluation_prompt_template.md` — the full prompt template with placeholders
- `massgen/subagent_types/round_evaluator/SUBAGENT.md` — source methodology for the evaluation approach
- `massgen/skills/massgen-develops-massgen/SKILL.md` — reference pattern for invoking MassGen via `--automation`
