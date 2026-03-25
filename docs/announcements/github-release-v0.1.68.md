# 🚀 Release Highlights — v0.1.68 (2026-03-25)

### 🔀 [Checkpoint Mode](https://github.com/massgen/MassGen/blob/main/docs/modules/checkpoint.md)
- **Delegator pattern**: Main agent plans solo then calls `checkpoint()` to delegate execution to the full multi-agent team
- **Fresh agent instances**: Clean backends and cloned workspaces for collaborative execution
- **Seamless handoff**: After team consensus, main agent resumes with results and deliverable files copied to its workspace
- **WebUI support**: Checkpoint mode display integrated into the modernized WebUI

### ⚡ LLM API Circuit Breaker
- **429 rate limit handling** ([#1024](https://github.com/massgen/MassGen/pull/1024)): Automatic circuit breaker pattern for Claude backend — detects 429 rate limits and backs off gracefully

### ✅ Fixes
- **LiteLLM supply chain fix** ([#1025](https://github.com/massgen/MassGen/pull/1025)): Pinned litellm<=1.82.6 and committed uv.lock to prevent dependency attacks

---

### 📖 Getting Started
- [**Quick Start Guide**](https://github.com/massgen/MassGen?tab=readme-ov-file#1--installation)
- **Try It**:
  ```bash
  pip install massgen==0.1.68
  # Try checkpoint mode
  uv run massgen --web
  ```
