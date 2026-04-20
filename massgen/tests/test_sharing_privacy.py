import json
from pathlib import Path
from types import SimpleNamespace
import shutil
from uuid import uuid4

import pytest

from massgen.session_exporter import export_command
from massgen.share import ShareError, TurnInfo, share_session, share_session_multi_turn


def _sandbox_dir(name: str) -> Path:
    base = Path.cwd() / ".pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"{name}_{uuid4().hex}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_minimal_attempt(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "status.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (log_dir / "metrics_summary.json").write_text(json.dumps({"meta": {"question": "q"}}), encoding="utf-8")


def test_export_command_blocks_public_sharing_by_default(monkeypatch, capsys):
    monkeypatch.setenv("MASSGEN_PRIVATE_MODE", "1")
    monkeypatch.delenv("MASSGEN_ALLOW_UNSAFE_SHARING", raising=False)
    args = SimpleNamespace(
        log_dir=None,
        turns="all",
        no_workspace=False,
        workspace_limit="500KB",
        yes=True,
        dry_run=False,
        verbose=False,
        json=True,
    )

    rc = export_command(args)
    assert rc == 1
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert "private mode" in payload["error"].lower()


def test_share_session_blocks_public_upload_by_default(monkeypatch):
    monkeypatch.setenv("MASSGEN_PRIVATE_MODE", "1")
    monkeypatch.delenv("MASSGEN_ALLOW_UNSAFE_SHARING", raising=False)
    monkeypatch.setattr("massgen.share.create_gist", lambda *args, **kwargs: "gist123")

    sandbox = _sandbox_dir("share_blocked")
    try:
        attempt_dir = sandbox / "turn_1" / "attempt_1"
        _write_minimal_attempt(attempt_dir)
        (attempt_dir / "agent_outputs").mkdir(exist_ok=True)
        (attempt_dir / "agent_outputs" / "agent_a.txt").write_text("answer", encoding="utf-8")

        with pytest.raises(ShareError, match="private mode"):
            share_session(attempt_dir)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def test_share_session_allows_upload_with_unsafe_override(monkeypatch):
    monkeypatch.setenv("MASSGEN_PRIVATE_MODE", "1")
    monkeypatch.setenv("MASSGEN_ALLOW_UNSAFE_SHARING", "1")
    monkeypatch.setattr("massgen.share.create_gist", lambda *args, **kwargs: "gist123")

    sandbox = _sandbox_dir("share_allowed")
    try:
        attempt_dir = sandbox / "turn_1" / "attempt_1"
        _write_minimal_attempt(attempt_dir)
        (attempt_dir / "agent_outputs").mkdir(exist_ok=True)
        (attempt_dir / "agent_outputs" / "agent_a.txt").write_text("answer", encoding="utf-8")

        url = share_session(attempt_dir)
        assert "gist123" in url
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def test_share_session_multi_turn_blocks_public_upload_by_default(monkeypatch):
    monkeypatch.setenv("MASSGEN_PRIVATE_MODE", "1")
    monkeypatch.delenv("MASSGEN_ALLOW_UNSAFE_SHARING", raising=False)
    monkeypatch.setattr("massgen.share.create_gist", lambda *args, **kwargs: "gist123")

    sandbox = _sandbox_dir("share_multi_blocked")
    try:
        attempt_dir = sandbox / "turn_1" / "attempt_1"
        _write_minimal_attempt(attempt_dir)
        (attempt_dir / "agent_outputs").mkdir(exist_ok=True)
        (attempt_dir / "agent_outputs" / "agent_a.txt").write_text("answer", encoding="utf-8")

        turn = TurnInfo(
            turn_number=1,
            attempt_number=1,
            total_attempts=1,
            attempt_path=attempt_dir,
        )

        with pytest.raises(ShareError, match="private mode"):
            share_session_multi_turn(sandbox, [turn])
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
