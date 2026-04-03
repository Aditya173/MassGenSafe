# 🚀 Release Highlights — v0.1.72 (2026-04-03)

### 🦎 [Grok Backend Update](https://docs.massgen.ai/en/latest/user_guide/backends.html)
- **Backend improvements** ([#1044](https://github.com/massgen/MassGen/pull/1044)): Updated Grok backend with latest improvements

### ⚡ [Circuit Breaker Phase 2](https://docs.massgen.ai/en/latest/user_guide/backends.html)
- **Extended to all major backends** ([#1038](https://github.com/massgen/MassGen/pull/1038)): LLM API circuit breaker now covers ChatCompletions, Response API, and Gemini (was Claude-only in v0.1.68)
- **Gemini 503 handling**: Gemini backend circuit breaker also triggers on 503 errors

---

### 📖 Getting Started
- [**Quick Start Guide**](https://github.com/massgen/MassGen?tab=readme-ov-file#1--installation)
- **Try It**:
  ```bash
  pip install massgen==0.1.72
  uv run massgen --config @examples/providers/others/grok_x_search.yaml "Research the latest posts and news about AI agents in the last week, and summarize the key trends and insights."
  ```
