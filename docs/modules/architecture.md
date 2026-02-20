# Architecture

## Core Flow

```text
cli.py -> orchestrator.py -> chat_agent.py -> backend/*.py
                |
        coordination_tracker.py (voting, consensus)
                |
        mcp_tools/ (tool execution)
```

## Key Components

**Orchestrator** (`orchestrator.py`): Central coordinator managing parallel agent execution, voting, and consensus detection. Handles coordination phases: initial_answer -> enforcement (voting) -> presentation.

**Backends** (`backend/`): Provider-specific implementations. All inherit from `base.py`. Add new backends by:
1. Create `backend/new_provider.py` inheriting from base
2. Register in `backend/__init__.py`
3. Add model mappings to `massgen/utils.py`
4. Add capabilities to `backend/capabilities.py`
5. Update `config_validator.py`

See also: [Backend Registration Checklist in CLAUDE.md Memory](../../CLAUDE.md)

**MCP Integration** (`mcp_tools/`): Model Context Protocol for external tools. `client.py` handles multi-server connections, `security.py` validates operations. Some tools have dual paths: SDK (in-process, for ClaudeCode) and stdio (config.toml-based, for Codex). **Stdio MCP servers run inside Docker where `massgen` is NOT installed** — never import from `massgen` in stdio servers. Pre-compute any needed values in the orchestrator and pass via JSON specs files. Also note Codex sometimes sends tool args as JSON strings instead of dicts — always add a `json.loads()` fallback.

**Streaming Buffer** (`backend/_streaming_buffer_mixin.py`): Tracks partial responses during streaming for compression recovery.

## Backend Hierarchy

```text
base.py (abstract interface)
    +-- base_with_custom_tool_and_mcp.py (tool + MCP support)
            |-- response.py (OpenAI Response API)
            |-- chat_completions.py (generic OpenAI-compatible)
            |-- claude.py (Anthropic)
            |-- claude_code.py (Claude Code SDK)
            |-- gemini.py (Google)
            +-- grok.py (xAI)
```

## Coordination as Evolutionary Search

MassGen's coordination loop maps to a genetic algorithm where the "population" is the set of agent answers and each round is a generation:

| GA Concept | MassGen Equivalent |
|---|---|
| **Population** | Current set of agent answers |
| **Selection** | Voting — agents evaluate and vote for the strongest answer |
| **Crossover** | Synthesis — agents combine strengths from multiple answers into a new one |
| **Mutation** | Variation — agents try different approaches for key parts of the solution |
| **Fitness** | Checklist evaluation (gap analysis, ideal version comparison) |
| **Speciation** | Persona generation — agents with different perspectives/methodologies explore different regions of the solution space |

### Diversity mechanisms

Without active diversity pressure, agents converge on the same approach and produce increasingly similar answers (the equivalent of a GA losing population diversity). MassGen maintains diversity through two layers:

1. **Persona generation** (explicit, configurable): When enabled, assigns agents different perspectives (`diversity_mode: perspective`), solution types (`implementation`), or working methodologies (`methodology`). This is the strongest diversity lever — it shapes how agents think from the start.

2. **Default prompt instructions** (always active): The evaluation prompts include a "Fresh Approach Consideration" that encourages agents to vary their approach for key parts rather than always refining the current best. The `new_answer` instruction explicitly presents "Vary" as a valid strategy alongside "Improve."

The balance between convergence (selection + synthesis) and diversity (variation + personas) is key. Too much convergence produces polished mediocrity. Too much diversity prevents answers from maturing. The checklist-gated voting controls when agents stop iterating, while personas and variation instructions control how different each iteration is.

## Agent Statelessness and Anonymity

Agents are STATELESS and ANONYMOUS across coordination rounds. Each round:
- Agent gets a fresh LLM invocation with no memory of previous rounds
- Agent does not know which agent it is (all identities are anonymous)
- Cross-agent information (answers, workspaces) is presented anonymously
- System prompts and branch names must NOT reveal agent identity or round history

## TUI Design Principles

**Timeline Chronology Rule**: Tool batching MUST respect chronological order. Tools should ONLY be batched when they arrive consecutively with no intervening content (thinking, text, status). When non-tool content arrives, any pending batch must be finalized before the content is added, and the next tool starts a fresh batch.

This is enforced via `ToolBatchTracker.mark_content_arrived()` in `content_handlers.py`, which is called whenever non-tool content is added to the timeline.

### TUI Debug Logging

All TUI debug logging goes through `massgen/frontend/displays/shared/tui_debug.py`. This module is the **single source of truth** for TUI debug file output.

- **Enable**: set `MASSGEN_TUI_DEBUG=1` in your environment.
- **Log file**: `<tempdir>/tui_debug.log` (uses `tempfile.gettempdir()` for cross-platform support).
- **Usage**: call `tui_log(msg)` from anywhere in the TUI layer. Widget-specific wrappers (e.g. `_wizard_log`, `_tab_log`) add a prefix tag and delegate to `tui_log`.
- **Do NOT** create ad-hoc `logging.FileHandler` instances with hard-coded paths in widget files — always use `tui_log`.
