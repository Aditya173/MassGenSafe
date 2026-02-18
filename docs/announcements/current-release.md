# MassGen v0.1.53 Release Announcement

<!--
This is the current release announcement. Copy this + feature-highlights.md to LinkedIn/X.
After posting, update the social links below.
-->

## Release Summary

We're excited to release MassGen v0.1.53, focused on Background Tools & Specialized Subagents! 🚀 Background tool execution for non-blocking long-running work. Specialized subagent types (Evaluator, Explorer) with built-in skills. Planning task verification requirements. TUI background job indicators and lifecycle controls.

## Install

```bash
pip install massgen==0.1.53
```

## Links

- **Release notes:** https://github.com/massgen/MassGen/releases/tag/v0.1.53
- **X post:** [TO BE ADDED AFTER POSTING]
- **LinkedIn post:** [TO BE ADDED AFTER POSTING]

---

## Full Announcement (for LinkedIn)

Copy everything below this line, then append content from `feature-highlights.md`:

---

We're excited to release MassGen v0.1.53, focused on Background Tools & Specialized Subagents! 🚀 Background tool execution for non-blocking long-running work. Specialized subagent types (Evaluator, Explorer) with built-in skills. Planning task verification requirements. TUI background job indicators and lifecycle controls.

**Key Features:**

**Background Tool Execution** - Non-blocking lifecycle tools:
- Start, monitor, wait, cancel, and list background jobs -- agents continue foreground work while long-running tools execute
- `start_background_tool`, `get_background_tool_status`, `get_background_tool_result`, `wait_for_background_tool`, `cancel_background_tool`, `list_background_tools`
- Compatible with custom tools and MCP server tools

**Specialized Subagents** - Purpose-built subagent types:
- **Evaluator**: Programmatic execution and verification (tests, Playwright, scripted checks)
- **Explorer**: Research and discovery (codebase search, semantic discovery, documentation analysis). Runs in background by default.
- Type discovery via `SUBAGENT.md` frontmatter for defining custom subagent types

**Planning Task Verification** - Quality assurance for planning:
- Tasks now require `verification` and `verification_method` fields by default
- `--no-require-verification` flag to opt out
- Framework-injected tasks exempt

**Also in this release:**
- TUI background job indicators: Agent status ribbon and background tasks modal with lifecycle controls
- Hook framework: Broadcasting subagent results to parent agents
- Tool argument normalization: Consistent argument handling across backends

**Bug Fixes:**
- Task plan verification improvements
- Codex reasoning config alignment

Release notes: https://github.com/massgen/MassGen/releases/tag/v0.1.53

Feature highlights:

<!-- Paste feature-highlights.md content here -->
