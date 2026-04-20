from types import SimpleNamespace
from unittest.mock import patch
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.fixture
def sandbox_dir():
    base = Path.cwd() / ".pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    sandbox = base / f"web_privacy_{uuid4().hex}"
    sandbox.mkdir(parents=True, exist_ok=True)
    try:
        yield sandbox
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def _private_web_env(monkeypatch, sandbox_dir: Path) -> None:
    monkeypatch.setenv("MASSGEN_PRIVATE_MODE", "1")
    monkeypatch.setenv("MASSGEN_LOCAL_API_TOKEN", "local-secret")
    monkeypatch.chdir(sandbox_dir)
    home_dir = sandbox_dir / "home"
    home_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("USERPROFILE", str(home_dir))


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer local-secret"}


def _app_with_patched_docker_status():
    from massgen.frontend.web.server import create_app

    with patch("massgen.utils.docker_diagnostics.diagnose_docker") as mock_diag:
        mock_diag.return_value = SimpleNamespace(
            is_available=False,
            status=SimpleNamespace(value="unavailable"),
            error_message="docker not running",
            resolution_steps=["start docker"],
        )
        app = create_app()
    return app


def test_api_routes_require_auth_in_private_mode(monkeypatch, sandbox_dir):
    _private_web_env(monkeypatch, sandbox_dir)
    app = _app_with_patched_docker_status()
    client = TestClient(app)

    response = client.get("/api/providers")
    assert response.status_code == 401


def test_api_routes_allow_bearer_auth_in_private_mode(monkeypatch, sandbox_dir):
    _private_web_env(monkeypatch, sandbox_dir)
    app = _app_with_patched_docker_status()
    client = TestClient(app)

    response = client.get("/api/providers", headers=_auth_headers())
    assert response.status_code == 200


def test_health_route_is_public_in_private_mode(monkeypatch, sandbox_dir):
    _private_web_env(monkeypatch, sandbox_dir)
    app = _app_with_patched_docker_status()
    client = TestClient(app)

    response = client.get("/api/health")
    assert response.status_code == 200


def test_websocket_rejects_missing_token_in_private_mode(monkeypatch, sandbox_dir):
    _private_web_env(monkeypatch, sandbox_dir)
    app = _app_with_patched_docker_status()
    client = TestClient(app)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/session-private"):
            pass

    assert exc_info.value.code == 1008


def test_websocket_allows_query_token_in_private_mode(monkeypatch, sandbox_dir):
    _private_web_env(monkeypatch, sandbox_dir)
    app = _app_with_patched_docker_status()
    client = TestClient(app)

    with client.websocket_connect("/ws/session-private?token=local-secret") as ws:
        payload = ws.receive_json()

    assert payload["type"] in {"init", "state_snapshot"}


def test_api_key_save_is_disabled_by_default_in_private_mode(monkeypatch, sandbox_dir):
    _private_web_env(monkeypatch, sandbox_dir)
    app = _app_with_patched_docker_status()
    client = TestClient(app)

    response = client.post(
        "/api/setup/api-keys",
        headers=_auth_headers(),
        json={"keys": {"OPENAI_API_KEY": "sk-test"}, "save_location": "global"},
    )
    assert response.status_code == 403


def test_share_endpoint_is_disabled_by_default_in_private_mode(monkeypatch, sandbox_dir):
    _private_web_env(monkeypatch, sandbox_dir)
    app = _app_with_patched_docker_status()
    client = TestClient(app)

    response = client.post(
        "/api/sessions/test-session/share",
        headers=_auth_headers(),
    )
    assert response.status_code == 403


def test_setup_status_includes_privacy_flags(monkeypatch, sandbox_dir):
    _private_web_env(monkeypatch, sandbox_dir)
    app = _app_with_patched_docker_status()
    client = TestClient(app)

    response = client.get("/api/setup/status", headers=_auth_headers())
    assert response.status_code == 200
    payload = response.json()
    assert payload["privacy"]["private_mode"] is True
    assert payload["privacy"]["requires_auth_token"] is True
    assert payload["privacy"]["sharing_enabled"] is False
    assert payload["privacy"]["web_key_save_enabled"] is False
