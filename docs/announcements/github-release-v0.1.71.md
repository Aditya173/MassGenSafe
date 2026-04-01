# 🚀 Release Highlights — v0.1.71 (2026-04-01)

### 🔍 Trace Analyzer Subagents
- **Background execution trace analysis**: Trace analyzer subagents now launch in the background after each round to analyze the previous round's execution trace and write insights into memory for next-round continuity
- **Fix trace memory**: Corrected memory handling in execution traces
- **Fix execution trace analyzer launch**: Trace analyzer now starts correctly

### 📋 Better Evaluation Criteria
- **Improved criteria generation**: Higher-quality, more opinionated criteria output

### 🧠 System Prompt Tuning
- **Better agent performance**: Adjusted system prompts for improved coordination across rounds

### 🔧 Fixes
- **Fix final injection**: Corrected injection behavior at the final stage
- **Fix eval criteria GPT pre-collab**: Resolved evaluation criteria issues with GPT models during pre-collaboration phase
- **Auto round fix for memory**: Fixed automatic round handling for memory

---

### 📖 Getting Started
- [**Quick Start Guide**](https://github.com/massgen/MassGen?tab=readme-ov-file#1--installation)
- **Try It**:
  ```bash
  pip install massgen==0.1.71
  uv run massgen --config @examples/features/trace_analyzer_background.yaml "Create an svg of an AI agent coding."
  ```
