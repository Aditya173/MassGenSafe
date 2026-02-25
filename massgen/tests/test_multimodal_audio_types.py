"""Tests for audio type routing: speech, music, and sound effects."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from massgen.tool._multimodal_tools.generation import _audio as audio_module
from massgen.tool._multimodal_tools.generation._base import (
    GenerationConfig,
    GenerationResult,
    MediaType,
)


def _make_config(tmp_path: Path, audio_type: str = "speech", **kwargs) -> GenerationConfig:
    """Helper to build a GenerationConfig for audio tests."""
    defaults = {
        "prompt": "test prompt",
        "output_path": tmp_path / "out.mp3",
        "media_type": MediaType.AUDIO,
        "backend": "elevenlabs",
        "extra_params": {"audio_type": audio_type},
    }
    defaults.update(kwargs)
    return GenerationConfig(**defaults)


def _ok_result(backend: str = "elevenlabs", **kwargs) -> GenerationResult:
    return GenerationResult(
        success=True,
        media_type=MediaType.AUDIO,
        backend_name=backend,
        file_size_bytes=1234,
        **kwargs,
    )


def _fail_result(backend: str = "elevenlabs", error: str = "failed") -> GenerationResult:
    return GenerationResult(success=False, backend_name=backend, error=error)


# --- Routing tests ---


@pytest.mark.asyncio
async def test_audio_type_defaults_to_speech(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """When audio_type is absent from extra_params, generate_audio routes to speech."""
    config = GenerationConfig(
        prompt="hello",
        output_path=tmp_path / "hello.mp3",
        media_type=MediaType.AUDIO,
        backend="elevenlabs",
        extra_params={},  # no audio_type
    )
    mock_speech = AsyncMock(return_value=_ok_result())
    monkeypatch.setattr(audio_module, "_generate_speech_elevenlabs", mock_speech)

    result = await audio_module.generate_audio(config)

    assert result.success is True
    mock_speech.assert_awaited_once_with(config)


@pytest.mark.asyncio
async def test_music_routes_to_elevenlabs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """audio_type='music' routes to _generate_music_elevenlabs."""
    config = _make_config(tmp_path, audio_type="music")
    mock_music = AsyncMock(return_value=_ok_result())
    monkeypatch.setattr(audio_module, "_generate_music_elevenlabs", mock_music)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

    result = await audio_module.generate_audio(config)

    assert result.success is True
    mock_music.assert_awaited_once_with(config)


@pytest.mark.asyncio
async def test_sfx_routes_to_elevenlabs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """audio_type='sound_effect' routes to _generate_sfx_elevenlabs."""
    config = _make_config(tmp_path, audio_type="sound_effect")
    mock_sfx = AsyncMock(return_value=_ok_result())
    monkeypatch.setattr(audio_module, "_generate_sfx_elevenlabs", mock_sfx)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

    result = await audio_module.generate_audio(config)

    assert result.success is True
    mock_sfx.assert_awaited_once_with(config)


# --- ElevenLabs-only enforcement ---


@pytest.mark.asyncio
async def test_music_requires_elevenlabs_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Music generation returns error when no ELEVENLABS_API_KEY."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _make_config(tmp_path, audio_type="music", backend="openai")

    result = await audio_module.generate_audio(config)

    assert result.success is False
    assert "ElevenLabs" in result.error
    assert "ELEVENLABS_API_KEY" in result.error


@pytest.mark.asyncio
async def test_sfx_requires_elevenlabs_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Sound effect generation returns error when no ELEVENLABS_API_KEY."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _make_config(tmp_path, audio_type="sound_effect", backend="openai")

    result = await audio_module.generate_audio(config)

    assert result.success is False
    assert "ElevenLabs" in result.error


@pytest.mark.asyncio
async def test_music_auto_upgrades_backend_to_elevenlabs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """When backend='openai' but audio_type='music' and ELEVENLABS_API_KEY exists, auto-route."""
    config = _make_config(tmp_path, audio_type="music", backend="openai")
    mock_music = AsyncMock(return_value=_ok_result())
    monkeypatch.setattr(audio_module, "_generate_music_elevenlabs", mock_music)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

    result = await audio_module.generate_audio(config)

    assert result.success is True
    mock_music.assert_awaited_once_with(config)


@pytest.mark.asyncio
async def test_music_no_openai_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Music has no OpenAI fallback even if OPENAI_API_KEY is set."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    config = _make_config(tmp_path, audio_type="music", backend="openai")

    result = await audio_module.generate_audio(config)

    assert result.success is False
    assert "only supported via ElevenLabs" in result.error


# --- Parameter forwarding tests ---


@pytest.mark.asyncio
async def test_music_duration_converted_to_ms(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Music generation converts duration seconds to milliseconds for the SDK."""
    config = _make_config(tmp_path, audio_type="music", duration=30)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

    # Mock the AsyncElevenLabs client
    mock_compose = AsyncMock(return_value=_async_byte_iter(b"audio-data"))
    mock_client = _mock_elevenlabs_client(music_compose=mock_compose)
    monkeypatch.setattr(audio_module, "AsyncElevenLabs", lambda **kw: mock_client)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    result = await audio_module._generate_music_elevenlabs(config)

    assert result.success is True
    call_kwargs = mock_compose.call_args[1]
    assert call_kwargs["music_length_ms"] == 30000


@pytest.mark.asyncio
async def test_music_force_instrumental_defaults_true(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """force_instrumental defaults to True for music generation."""
    config = _make_config(tmp_path, audio_type="music")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

    mock_compose = AsyncMock(return_value=_async_byte_iter(b"audio-data"))
    mock_client = _mock_elevenlabs_client(music_compose=mock_compose)
    monkeypatch.setattr(audio_module, "AsyncElevenLabs", lambda **kw: mock_client)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    result = await audio_module._generate_music_elevenlabs(config)

    assert result.success is True
    call_kwargs = mock_compose.call_args[1]
    assert call_kwargs["force_instrumental"] is True


@pytest.mark.asyncio
async def test_sfx_duration_clamped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """SFX duration is clamped to 0.5-30 range."""
    config = _make_config(tmp_path, audio_type="sound_effect", duration=60)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

    mock_convert = AsyncMock(return_value=_async_byte_iter(b"sfx-data"))
    mock_client = _mock_elevenlabs_client(sfx_convert=mock_convert)
    monkeypatch.setattr(audio_module, "AsyncElevenLabs", lambda **kw: mock_client)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    result = await audio_module._generate_sfx_elevenlabs(config)

    assert result.success is True
    call_kwargs = mock_convert.call_args[1]
    assert call_kwargs["duration_seconds"] == 30.0


@pytest.mark.asyncio
async def test_sfx_extra_params_forwarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """prompt_influence from extra_params is forwarded to SFX generation."""
    config = _make_config(
        tmp_path,
        audio_type="sound_effect",
        extra_params={"audio_type": "sound_effect", "prompt_influence": 0.8},
    )
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

    mock_convert = AsyncMock(return_value=_async_byte_iter(b"sfx-data"))
    mock_client = _mock_elevenlabs_client(sfx_convert=mock_convert)
    monkeypatch.setattr(audio_module, "AsyncElevenLabs", lambda **kw: mock_client)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    result = await audio_module._generate_sfx_elevenlabs(config)

    assert result.success is True
    call_kwargs = mock_convert.call_args[1]
    assert call_kwargs["prompt_influence"] == 0.8


# --- Integration: generate_media threads audio_type ---


@pytest.mark.asyncio
async def test_generate_media_threads_audio_type(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """audio_type parameter from generate_media arrives in config.extra_params."""
    from unittest.mock import patch

    captured_configs: list[GenerationConfig] = []

    async def _capture_generate_audio(config: GenerationConfig) -> GenerationResult:
        captured_configs.append(config)
        config.output_path.write_bytes(b"fake-audio")
        return GenerationResult(
            success=True,
            output_path=config.output_path,
            media_type=MediaType.AUDIO,
            backend_name="elevenlabs",
            model_used="elevenlabs-music",
            file_size_bytes=10,
        )

    with (
        patch(
            "massgen.context.task_context.load_task_context_with_warning",
            return_value=(None, None),
        ),
        patch(
            "massgen.tool._multimodal_tools.generation.generate_media.select_backend_and_model",
            return_value=("elevenlabs", "eleven_multilingual_v2"),
        ),
        patch(
            "massgen.tool._multimodal_tools.generation.generate_media.generate_audio",
            new=_capture_generate_audio,
        ),
    ):
        from massgen.tool._multimodal_tools.generation.generate_media import (
            generate_media,
        )

        result = await generate_media(
            prompt="epic cinematic soundtrack",
            mode="audio",
            audio_type="music",
            agent_cwd=str(tmp_path),
            allowed_paths=[str(tmp_path)],
        )

    import json

    payload = json.loads(result.output_blocks[0].data)
    assert payload["success"] is True

    assert len(captured_configs) == 1
    assert captured_configs[0].extra_params.get("audio_type") == "music"


# --- Helpers ---


async def _async_byte_iter(data: bytes):
    """Create an async iterator over byte chunks (simulates SDK response)."""
    yield data


def _mock_elevenlabs_client(
    tts_convert=None,
    music_compose=None,
    sfx_convert=None,
):
    """Build a mock AsyncElevenLabs client with configurable method mocks."""
    from types import SimpleNamespace

    return SimpleNamespace(
        text_to_speech=SimpleNamespace(
            convert=tts_convert or AsyncMock(return_value=_async_byte_iter(b"speech")),
        ),
        music=SimpleNamespace(
            compose=music_compose or AsyncMock(return_value=_async_byte_iter(b"music")),
        ),
        text_to_sound_effects=SimpleNamespace(
            convert=sfx_convert or AsyncMock(return_value=_async_byte_iter(b"sfx")),
        ),
    )
