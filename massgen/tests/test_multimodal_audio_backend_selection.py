"""Tests for audio generation backend selection and fallback behavior."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from massgen.tool._multimodal_tools.generation import _audio as audio_generation
from massgen.tool._multimodal_tools.generation._base import (
    GenerationConfig,
    GenerationResult,
    MediaType,
)
from massgen.tool._multimodal_tools.generation._selector import select_backend_and_model


def test_audio_auto_selection_prefers_elevenlabs_when_key_is_present(monkeypatch: pytest.MonkeyPatch):
    """Audio auto-selection should prefer ElevenLabs over OpenAI when both are available."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-elevenlabs-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    backend, model = select_backend_and_model(
        media_type=MediaType.AUDIO,
        preferred_backend=None,
        preferred_model=None,
        config=None,
    )

    assert backend == "elevenlabs"
    assert model == "eleven_multilingual_v2"


def test_audio_auto_selection_falls_back_to_openai_when_no_elevenlabs_key(monkeypatch: pytest.MonkeyPatch):
    """Audio auto-selection should use OpenAI when ElevenLabs API key is unavailable."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    backend, model = select_backend_and_model(
        media_type=MediaType.AUDIO,
        preferred_backend=None,
        preferred_model=None,
        config=None,
    )

    assert backend == "openai"
    assert model == "gpt-4o-mini-tts"


@pytest.mark.asyncio
async def test_generate_audio_routes_to_elevenlabs_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """generate_audio should route to ElevenLabs when explicitly requested."""
    config = GenerationConfig(
        prompt="hello world",
        output_path=tmp_path / "hello.mp3",
        media_type=MediaType.AUDIO,
        backend="elevenlabs",
    )

    elevenlabs_result = GenerationResult(
        success=True,
        output_path=config.output_path,
        media_type=MediaType.AUDIO,
        backend_name="elevenlabs",
        model_used="eleven_multilingual_v2",
        file_size_bytes=1234,
    )
    mock_elevenlabs = AsyncMock(return_value=elevenlabs_result)
    mock_openai = AsyncMock()

    monkeypatch.setattr(audio_generation, "_generate_speech_elevenlabs", mock_elevenlabs)
    monkeypatch.setattr(audio_generation, "_generate_audio_openai", mock_openai)

    result = await audio_generation.generate_audio(config)

    assert result.success is True
    assert result.backend_name == "elevenlabs"
    mock_elevenlabs.assert_awaited_once_with(config)
    mock_openai.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_audio_falls_back_to_openai_when_elevenlabs_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """If ElevenLabs fails, generate_audio should fall back to OpenAI when available."""
    config = GenerationConfig(
        prompt="hello world",
        output_path=tmp_path / "hello.mp3",
        media_type=MediaType.AUDIO,
        backend="elevenlabs",
    )

    elevenlabs_failure = GenerationResult(
        success=False,
        backend_name="elevenlabs",
        model_used="eleven_multilingual_v2",
        error="ElevenLabs unavailable",
    )
    openai_result = GenerationResult(
        success=True,
        output_path=config.output_path,
        media_type=MediaType.AUDIO,
        backend_name="openai",
        model_used="gpt-4o-mini-tts",
        file_size_bytes=1000,
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    mock_elevenlabs = AsyncMock(return_value=elevenlabs_failure)
    mock_openai = AsyncMock(return_value=openai_result)

    monkeypatch.setattr(audio_generation, "_generate_speech_elevenlabs", mock_elevenlabs)
    monkeypatch.setattr(audio_generation, "_generate_audio_openai", mock_openai)

    result = await audio_generation.generate_audio(config)

    assert result.success is True
    assert result.backend_name == "openai"
    mock_elevenlabs.assert_awaited_once_with(config)
    mock_openai.assert_awaited_once_with(config)


@pytest.mark.asyncio
async def test_generate_audio_openai_awaits_stream_to_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Regression: OpenAI streaming write must be awaited so file exists before stat()."""
    config = GenerationConfig(
        prompt="hello world",
        output_path=tmp_path / "speech.mp3",
        media_type=MediaType.AUDIO,
        backend="openai",
    )

    async def _write_output(path: Path) -> None:
        Path(path).write_bytes(b"audio-bytes")

    mock_response = SimpleNamespace(stream_to_file=AsyncMock(side_effect=_write_output))

    class _FakeStreamContext:
        async def __aenter__(self):
            return mock_response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    create_mock = Mock(return_value=_FakeStreamContext())
    mock_client = SimpleNamespace(
        audio=SimpleNamespace(
            speech=SimpleNamespace(
                with_streaming_response=SimpleNamespace(create=create_mock),
            ),
        ),
    )
    mock_async_openai = Mock(return_value=mock_client)

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(audio_generation, "AsyncOpenAI", mock_async_openai)

    result = await audio_generation._generate_audio_openai(config)

    assert result.success is True
    assert result.file_size_bytes == len(b"audio-bytes")
    create_mock.assert_called_once_with(model="gpt-4o-mini-tts", voice="alloy", input=config.prompt)
    mock_response.stream_to_file.assert_awaited_once_with(config.output_path)
