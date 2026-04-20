from fastapi.testclient import TestClient

from massgen.server.app import create_app
from massgen.server.openai.model_router import ResolvedModel


class FakeEngine:
    async def completion(self, req, resolved: ResolvedModel, *, request_id: str):
        _ = (req, resolved, request_id)
        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": 123,
            "model": "massgen",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello world",
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }


def _private_server_env(monkeypatch) -> None:
    monkeypatch.setenv("MASSGEN_PRIVATE_MODE", "1")
    monkeypatch.setenv("MASSGEN_SERVER_TOKEN", "server-secret")


def test_chat_completions_requires_auth_in_private_mode(monkeypatch):
    _private_server_env(monkeypatch)
    app = create_app(engine=FakeEngine())
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "massgen",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )

    assert resp.status_code == 401


def test_chat_completions_accepts_bearer_token_in_private_mode(monkeypatch):
    _private_server_env(monkeypatch)
    app = create_app(engine=FakeEngine())
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer server-secret"},
        json={
            "model": "massgen",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )

    assert resp.status_code == 200


def test_chat_completions_accepts_query_token_in_private_mode(monkeypatch):
    _private_server_env(monkeypatch)
    app = create_app(engine=FakeEngine())
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions?token=server-secret",
        json={
            "model": "massgen",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )

    assert resp.status_code == 200


def test_health_remains_public_in_private_mode(monkeypatch):
    _private_server_env(monkeypatch)
    app = create_app(engine=FakeEngine())
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
