# 🚀 Release Highlights — v0.1.69 (2026-03-27)

### 🌐 [WebUI Automation Auto-Start](https://docs.massgen.ai/en/latest/user_guide/webui.html)
- **Auto-start without browser interaction**: `massgen --web --automation --config config.yaml "Your question"` begins immediately — open the URL at any point to monitor progress mid-run
- **Automatic config resolution**: Automation mode resolves config automatically when none is specified
- **Auto-end on completion**: Web automation correctly auto-ends when a skill completes

### 🤖 [MassGen Skill Redesign](https://docs.massgen.ai/en/latest/user_guide/skills.html)
- **Increased usability and WebUI integration** ([#1032](https://github.com/massgen/MassGen/pull/1032)): MassGen skill launches directly from the WebUI with live session tracking and monitoring

### ✨ Plus
- **Quickstart Wizard rework** ([#1032](https://github.com/massgen/MassGen/pull/1032)): New Welcome, Skills, API Key, Docker, and Setup Mode steps for smoother onboarding
- **Workspace Browser expansion** ([#1032](https://github.com/massgen/MassGen/pull/1032)): WorkspaceModal and improved workspace connection
- **Flexible criteria fields** ([#1032](https://github.com/massgen/MassGen/pull/1032)): `description` or `name` accepted as alternatives to `text` in evaluation criteria JSON

---

### 📖 Getting Started
- [**Quick Start Guide**](https://github.com/massgen/MassGen?tab=readme-ov-file#1--installation)
- **Try It**:
  ```bash
  pip install massgen==0.1.69
  # Auto-start a run and monitor in the WebUI
  uv run massgen --web --automation --config config.yaml "Your question"
  ```
