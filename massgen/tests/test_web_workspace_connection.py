from pathlib import Path

from massgen.frontend.web.server import _resolve_watch_session_workspaces


def test_resolve_watch_session_workspaces_prefers_final_snapshot_when_current_workspace_has_no_visible_files(
    tmp_path: Path,
) -> None:
    current_workspace = tmp_path / "workspace_live"
    (current_workspace / ".codex").mkdir(parents=True)
    (current_workspace / ".codex" / "config.toml").write_text("model = 'gpt-5.4'\n", encoding="utf-8")

    log_dir = tmp_path / "turn_1" / "attempt_1"
    final_workspace = log_dir / "final" / "agent_a" / "workspace"
    (final_workspace / "deliverables").mkdir(parents=True)
    (final_workspace / "deliverables" / "love_poem.md").write_text(
        "# Love\n\nA visible file.\n",
        encoding="utf-8",
    )

    status_data = {
        "agents": {
            "agent_a": {
                "workspace_paths": {
                    "workspace": str(current_workspace),
                },
            },
        },
    }

    resolved = _resolve_watch_session_workspaces(status_data, log_dir)

    assert resolved == [
        (
            str(final_workspace.resolve()),
            [
                {
                    "path": "deliverables/love_poem.md",
                    "size": 24,
                    "modified": resolved[0][1][0]["modified"],
                },
            ],
        ),
    ]


def test_resolve_watch_session_workspaces_keeps_current_workspace_when_it_has_visible_files(
    tmp_path: Path,
) -> None:
    current_workspace = tmp_path / "workspace_live"
    (current_workspace / "deliverables").mkdir(parents=True)
    (current_workspace / "deliverables" / "draft.md").write_text(
        "visible now\n",
        encoding="utf-8",
    )

    log_dir = tmp_path / "turn_1" / "attempt_1"
    final_workspace = log_dir / "final" / "agent_a" / "workspace"
    (final_workspace / "deliverables").mkdir(parents=True)
    (final_workspace / "deliverables" / "final.md").write_text(
        "should not replace current\n",
        encoding="utf-8",
    )

    status_data = {
        "agents": {
            "agent_a": {
                "workspace_paths": {
                    "workspace": str(current_workspace),
                },
            },
        },
    }

    resolved = _resolve_watch_session_workspaces(status_data, log_dir)

    assert resolved == [
        (
            str(current_workspace.resolve()),
            [
                {
                    "path": "deliverables/draft.md",
                    "size": 12,
                    "modified": resolved[0][1][0]["modified"],
                },
            ],
        ),
    ]
