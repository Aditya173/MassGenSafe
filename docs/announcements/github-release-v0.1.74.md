# 🚀 Release Highlights — v0.1.74 (2026-04-08)

### 🛡️ [Checkpoint MCP Improvements](https://github.com/massgen/MassGen/blob/main/docs/modules/checkpoint.md)
- **Standalone server enhancements** ([#1050](https://github.com/massgen/MassGen/pull/1050)): Major additions to the standalone checkpoint MCP server
- **Subprocess execution refinements**: Better isolation, workspace handling, and event relay

### 🔧 [Duplicate Tool Call Fix](https://docs.massgen.ai/en/latest/user_guide/backends.html)
- **ChatCompletions and Response API** ([#1050](https://github.com/massgen/MassGen/pull/1050)): Resolved duplicate tool call issues in `base_with_custom_tool_and_mcp.py`, ChatCompletions (including for MiniMax on OpenRouter), and Response backends

### 🐛 Fixes
- **Pre-collab criteria fix**: Refinements to evaluation criteria generation in pre-collaboration phase

---

### 📖 Getting Started
- [**Quick Start Guide**](https://github.com/massgen/MassGen?tab=readme-ov-file#1--installation)
- **Try It**:
  ```bash
  pip install massgen==0.1.74
  # Try checkpoint mode in Claude Code
  claude mcp add massgen-checkpoint-mcp -- \
    uvx --from massgen massgen-checkpoint-mcp --config path/to/config.yaml
  ```
