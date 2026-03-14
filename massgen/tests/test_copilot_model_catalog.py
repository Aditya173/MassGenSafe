"""Tests for GitHub Copilot runtime model discovery."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from massgen.config_builder import ConfigBuilder
from massgen.frontend.web.server import create_app
from massgen.utils import model_catalog, model_matcher


class _FakeCopilotClient:
    def __init__(self, models):
        self._models = models
        self.start = AsyncMock()
        self.stop = AsyncMock()
        self.list_models = AsyncMock(return_value=models)


@pytest.mark.asyncio
async def test_get_models_for_provider_uses_copilot_sdk_runtime_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Copilot model discovery should come from the SDK's runtime list_models call."""
    fake_client = _FakeCopilotClient(
        [
            SimpleNamespace(id="claude-opus-4.6"),
            SimpleNamespace(id="gpt-5-mini"),
            SimpleNamespace(id="gemini-3-pro"),
        ],
    )
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path / "model_cache")
    monkeypatch.setitem(
        sys.modules,
        "copilot",
        SimpleNamespace(CopilotClient=lambda: fake_client),
    )

    models = await model_catalog.get_models_for_provider("copilot", use_cache=False)

    assert models == ["gpt-5-mini", "claude-opus-4.6", "gemini-3-pro"]
    fake_client.start.assert_awaited_once()
    fake_client.list_models.assert_awaited_once()
    fake_client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_model_metadata_for_provider_uses_copilot_sdk_runtime_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Copilot metadata discovery should expose reasoning metadata from the SDK."""
    fake_client = _FakeCopilotClient(
        [
            SimpleNamespace(
                id="claude-opus-4.6",
                name="Claude Opus 4.6",
                supported_reasoning_efforts=["medium", "high", "max"],
                default_reasoning_effort="high",
            ),
            SimpleNamespace(
                id="gpt-5-mini",
                name="GPT-5 mini",
                supported_reasoning_efforts=["low", "medium", "high"],
                default_reasoning_effort="medium",
            ),
        ],
    )
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path / "model_cache")
    monkeypatch.setitem(
        sys.modules,
        "copilot",
        SimpleNamespace(CopilotClient=lambda: fake_client),
    )

    metadata = await model_catalog.get_model_metadata_for_provider("copilot", use_cache=False)

    assert metadata == [
        {
            "id": "gpt-5-mini",
            "name": "GPT-5 mini",
            "supported_reasoning_efforts": ["low", "medium", "high"],
            "default_reasoning_effort": "medium",
        },
        {
            "id": "claude-opus-4.6",
            "name": "Claude Opus 4.6",
            "supported_reasoning_efforts": ["medium", "high", "max"],
            "default_reasoning_effort": "high",
        },
    ]


def test_get_all_models_for_provider_prefers_copilot_runtime_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config builder fuzzy matching should reuse the shared dynamic Copilot catalog."""
    monkeypatch.setattr(
        model_catalog,
        "get_models_for_provider_sync",
        lambda provider_type, use_cache=True: ["gpt-5-mini", "claude-opus-4.6"],
    )

    models = model_matcher.get_all_models_for_provider("copilot", use_api=True)

    assert models == ["gpt-5-mini", "claude-opus-4.6"]


def test_provider_models_endpoint_returns_copilot_dynamic_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing provider-models API should expose Copilot runtime discovery."""

    async def _fake_get_models_for_provider(provider: str, use_cache: bool = True) -> list[str]:
        assert provider == "copilot"
        assert use_cache is True
        return ["gpt-5-mini", "claude-opus-4.6"]

    monkeypatch.setattr(
        "massgen.utils.model_catalog.get_models_for_provider",
        _fake_get_models_for_provider,
    )

    app = create_app()
    client = TestClient(app)

    response = client.get("/api/providers/copilot/models")

    assert response.status_code == 200
    assert response.json() == {
        "provider_id": "copilot",
        "models": ["gpt-5-mini", "claude-opus-4.6"],
        "source": "dynamic",
    }


def test_provider_model_metadata_endpoint_returns_copilot_dynamic_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The metadata endpoint should expose Copilot runtime model capabilities."""

    async def _fake_get_model_metadata_for_provider(
        provider: str,
        use_cache: bool = True,
    ) -> list[dict[str, object]]:
        assert provider == "copilot"
        assert use_cache is True
        return [
            {
                "id": "gpt-5-mini",
                "name": "GPT-5 mini",
                "supported_reasoning_efforts": ["low", "medium", "high"],
                "default_reasoning_effort": "medium",
            },
            {
                "id": "claude-opus-4.6",
                "name": "Claude Opus 4.6",
                "supported_reasoning_efforts": ["medium", "high", "max"],
                "default_reasoning_effort": "high",
            },
        ]

    monkeypatch.setattr(
        "massgen.utils.model_catalog.get_model_metadata_for_provider",
        _fake_get_model_metadata_for_provider,
    )

    app = create_app()
    client = TestClient(app)

    response = client.get("/api/providers/copilot/models/metadata")

    assert response.status_code == 200
    assert response.json() == {
        "provider_id": "copilot",
        "models": [
            {
                "id": "gpt-5-mini",
                "name": "GPT-5 mini",
                "supported_reasoning_efforts": ["low", "medium", "high"],
                "default_reasoning_effort": "medium",
            },
            {
                "id": "claude-opus-4.6",
                "name": "Claude Opus 4.6",
                "supported_reasoning_efforts": ["medium", "high", "max"],
                "default_reasoning_effort": "high",
            },
        ],
        "source": "dynamic",
    }


def test_select_model_smart_uses_runtime_copilot_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """The questionary quickstart path should use runtime Copilot models too."""
    monkeypatch.setattr(
        "massgen.config_builder.get_all_models_for_provider",
        lambda provider_type: ["gpt-5-mini", "claude-opus-4.6"],
    )

    captured: dict[str, object] = {}

    class _FakePrompt:
        def ask(self):
            return "claude-opus-4.6"

    def _fake_autocomplete(prompt, choices, **kwargs):
        captured["prompt"] = prompt
        captured["choices"] = list(choices)
        return _FakePrompt()

    monkeypatch.setattr("massgen.config_builder.questionary.autocomplete", _fake_autocomplete)

    builder = ConfigBuilder()
    model = builder.select_model_smart(
        "copilot",
        ["gpt-4.1", "gpt-5-mini"],
        current_model="gpt-5-mini",
        prompt="  Model:",
    )

    assert model == "claude-opus-4.6"
    assert captured["choices"] == ["gpt-5-mini", "claude-opus-4.6"]
