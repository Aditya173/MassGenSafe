# MassGen v0.1.62 — MassGen Skill & Viewer

New general-purpose MassGen Skill with 4 modes (general, evaluate, plan, spec) for use from Claude Code and other AI agents. Session viewer for real-time observation of automation runs. Backend improvements and quickstart enhancements.

## 🧩 MassGen Skill

- New general-purpose multi-agent skill with 4 modes: general, evaluate, plan, and spec
- Usable from Claude Code, Codex, and other AI coding agents via `/massgen`
- Auto-installation and auto-sync to a separate skills repository for easy distribution
- Comprehensive reference documentation and workflow guides for each mode

## 👁️ Session Viewer

- New `massgen viewer` command for real-time observation of automation sessions in the TUI
- Interactive session picker (`--pick`) and web viewing mode (`--web`)

## ⚡ Backend & Quickstart

- Claude Code backend with background task execution and SDK MCP support
- Codex backend with native filesystem access, JSONL streaming, and MCP tool integration
- Copilot runtime model discovery with metadata caching
- Headless quickstart (`--quickstart --headless`) for automated CI/CD setup
- Web quickstart (`--web-quickstart`) for browser-based configuration
- `--print-backends` table showing all supported backends with capabilities

## 📝 Evaluation & Planning

- Better planning prompts with thoroughness support (standard vs thorough)
- Removed should/could evaluation criteria to reduce output similarity

## 🚀 Try It

```bash
# Install the MassGen Skill for your AI agent
npx skills add massgen/skills --all
# Then in Claude Code, Cursor, Copilot, etc.:
#   /massgen "Your complex task"

# Or install MassGen directly
pip install massgen==0.1.62
# Try the Session Viewer
uv run massgen viewer --pick
```

**Full Changelog:** https://github.com/massgen/MassGen/blob/main/CHANGELOG.md

📖 [Documentation](https://docs.massgen.ai) · 💬 [Discord](https://discord.massgen.ai) · 🐦 [X/Twitter](https://x.massgen.ai)
