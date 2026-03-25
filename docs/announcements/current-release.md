# MassGen v0.1.68 Release Announcement

<!--
This is the current release announcement. Copy this + feature-highlights.md to LinkedIn/X.
After posting, update the social links below.
-->

## Release Summary

We're excited to release MassGen v0.1.68 — Checkpoint Mode! 🚀 New checkpoint coordination mode lets a main agent plan solo then delegate execution to the full multi-agent team via the `checkpoint()` tool. Plus: LLM API circuit breaker for 429 rate limit handling, WebUI checkpoint support, and LiteLLM supply chain fix.

## Install

```bash
pip install massgen==0.1.68
```

## Links

- **Release notes:** https://github.com/massgen/MassGen/releases/tag/v0.1.68
- **X post:** [TO BE ADDED AFTER POSTING]
- **LinkedIn post:** [TO BE ADDED AFTER POSTING]

---

## Full Announcement (for LinkedIn)

Copy everything below this line, then append content from `feature-highlights.md`:

---

We're excited to release MassGen v0.1.68 — Checkpoint Mode! 🚀 New checkpoint coordination mode lets a main agent plan solo then delegate execution to the full multi-agent team via the `checkpoint()` tool. Plus: LLM API circuit breaker for 429 rate limit handling, WebUI checkpoint support, and LiteLLM supply chain fix.

**Key Improvement:**

🔀 **Checkpoint Mode** - Delegator pattern for multi-agent coordination:
- Main agent plans and gathers context solo, then calls `checkpoint()` to delegate to the team
- Fresh agent instances with clean backends execute the task collaboratively
- After team consensus, main agent resumes with results and deliverable files
- WebUI support for checkpoint mode display

**Plus:**
- ⚡ **LLM API circuit breaker** — automatic 429 rate limit handling with circuit breaker pattern for Claude backend
- 🔒 **LiteLLM supply chain fix** — pinned litellm<=1.82.6 and committed uv.lock to prevent dependency attacks

**Getting Started:**

```bash
pip install massgen==0.1.68
# Try checkpoint mode
uv run massgen --web
```

Release notes: https://github.com/massgen/MassGen/releases/tag/v0.1.68

Feature highlights:

<!-- Paste feature-highlights.md content here -->
