# 🚀 Release Highlights — v0.1.73 (2026-04-06)

### 🧬 [Eval Criteria Evolver Subagent](https://docs.massgen.ai/en/latest/user_guide/concepts.html)
- **Criteria that improve themselves** ([#1047](https://github.com/massgen/MassGen/pull/1047)): New subagent type that evolves evaluation criteria across rounds — sharper, more opinionated criteria as the run progresses

### 🛡️ [Checkpoint Objective Mode](https://github.com/massgen/MassGen/blob/main/docs/modules/checkpoint.md)
- **Safety planning for irreversible actions** ([#1047](https://github.com/massgen/MassGen/pull/1047)): Initial draft of checkpoint MCP with `objective` mode — plan irreversible operations (deletions, deployments, financial actions) safely before executing
- **Structured plan output**: Returns ordered plan with per-step constraints and recursive recovery trees

### 👁️ Improved Eval Criteria Visibility
- **Clearer criteria visibility**: See what criteria agents are working against, more clearly

---

### 📖 Getting Started
- [**Quick Start Guide**](https://github.com/massgen/MassGen?tab=readme-ov-file#1--installation)
- **Try It**:
  ```bash
  pip install massgen==0.1.73
  uv run massgen --config @examples/features/trace_analyzer_background.yaml "Create an svg of an AI agent coding."
  ```
