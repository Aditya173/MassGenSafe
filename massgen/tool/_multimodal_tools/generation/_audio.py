"""Audio generation backends: ElevenLabs (TTS, music, SFX) and OpenAI TTS."""

from elevenlabs import AsyncElevenLabs
from openai import AsyncOpenAI

from massgen.logger_config import logger
from massgen.tool._multimodal_tools.generation._base import (
    GenerationConfig,
    GenerationResult,
    MediaType,
    get_api_key,
    get_default_model,
    has_api_key,
)

# Available voices for OpenAI TTS
OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "sage"]

# Supported audio formats
AUDIO_FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]


async def generate_audio(config: GenerationConfig) -> GenerationResult:
    """Generate audio using the selected backend.

    Dispatches based on ``audio_type`` in ``config.extra_params``:
    - ``"speech"`` (default): Text-to-speech via ElevenLabs or OpenAI.
    - ``"music"``: Music generation via ElevenLabs only.
    - ``"sound_effect"``: Sound effect generation via ElevenLabs only.

    Args:
        config: GenerationConfig with prompt (text), output_path, voice, etc.

    Returns:
        GenerationResult with success status and file info
    """
    audio_type = config.extra_params.get("audio_type", "speech")
    backend = (config.backend or "openai").lower()

    # Music and sound effects are ElevenLabs-only.
    if audio_type in {"music", "sound_effect"}:
        if backend not in {"elevenlabs", "eleven_labs"} and has_api_key("elevenlabs"):
            backend = "elevenlabs"
        if not has_api_key("elevenlabs"):
            return GenerationResult(
                success=False,
                backend_name=backend,
                error=(f"{audio_type} generation is only supported via " "ElevenLabs. Set ELEVENLABS_API_KEY."),
            )
        if audio_type == "music":
            return await _generate_music_elevenlabs(config)
        return await _generate_sfx_elevenlabs(config)

    # Speech: ElevenLabs preferred with OpenAI fallback.
    if backend in {"elevenlabs", "eleven_labs"}:
        elevenlabs_result = await _generate_speech_elevenlabs(config)
        if elevenlabs_result.success:
            return elevenlabs_result

        if get_api_key("openai"):
            logger.warning(
                f"ElevenLabs audio generation failed ({elevenlabs_result.error}). " "Falling back to OpenAI TTS.",
            )
            return await _generate_audio_openai(config)

        return elevenlabs_result

    if backend != "openai":
        logger.warning(f"Unknown audio backend '{backend}', falling back to OpenAI TTS.")

    return await _generate_audio_openai(config)


# ---------------------------------------------------------------------------
# ElevenLabs backends (SDK)
# ---------------------------------------------------------------------------


async def _generate_speech_elevenlabs(config: GenerationConfig) -> GenerationResult:
    """Generate speech using the ElevenLabs Text-to-Speech API."""
    api_key = get_api_key("elevenlabs")
    if not api_key:
        return GenerationResult(
            success=False,
            backend_name="elevenlabs",
            error="ElevenLabs API key not found. Set ELEVENLABS_API_KEY environment variable.",
        )

    model = config.model or get_default_model("elevenlabs", MediaType.AUDIO) or "eleven_multilingual_v2"
    voice_id = config.voice or "Rachel"

    try:
        client = AsyncElevenLabs(api_key=api_key)
        audio_iterator = await client.text_to_speech.convert(
            voice_id=voice_id,
            text=config.prompt,
            model_id=model,
        )

        chunks: list[bytes] = []
        async for chunk in audio_iterator:
            chunks.append(chunk)
        audio_bytes = b"".join(chunks)

        config.output_path.write_bytes(audio_bytes)
        file_size = config.output_path.stat().st_size

        return GenerationResult(
            success=True,
            output_path=config.output_path,
            media_type=MediaType.AUDIO,
            backend_name="elevenlabs",
            model_used=model,
            file_size_bytes=file_size,
            metadata={
                "audio_type": "speech",
                "voice": config.voice,
                "resolved_voice_id": voice_id,
                "format": "mp3",
                "text_length": len(config.prompt),
            },
        )
    except Exception as e:
        logger.exception(f"ElevenLabs TTS generation failed: {e}")
        return GenerationResult(
            success=False,
            backend_name="elevenlabs",
            model_used=model,
            error=f"ElevenLabs TTS error: {e}",
        )


async def _generate_music_elevenlabs(config: GenerationConfig) -> GenerationResult:
    """Generate music using the ElevenLabs Music API."""
    api_key = get_api_key("elevenlabs")
    if not api_key:
        return GenerationResult(
            success=False,
            backend_name="elevenlabs",
            error="ElevenLabs API key not found. Set ELEVENLABS_API_KEY environment variable.",
        )

    try:
        client = AsyncElevenLabs(api_key=api_key)

        duration_ms = (config.duration or 30) * 1000
        force_instrumental = config.extra_params.get("force_instrumental", True)

        audio_iterator = await client.music.compose(
            prompt=config.prompt,
            music_length_ms=duration_ms,
            force_instrumental=force_instrumental,
        )

        chunks: list[bytes] = []
        async for chunk in audio_iterator:
            chunks.append(chunk)
        audio_bytes = b"".join(chunks)

        config.output_path.write_bytes(audio_bytes)
        file_size = config.output_path.stat().st_size

        return GenerationResult(
            success=True,
            output_path=config.output_path,
            media_type=MediaType.AUDIO,
            backend_name="elevenlabs",
            model_used="elevenlabs-music",
            file_size_bytes=file_size,
            metadata={
                "audio_type": "music",
                "duration_ms": duration_ms,
                "force_instrumental": force_instrumental,
                "format": "mp3",
            },
        )
    except Exception as e:
        logger.exception(f"ElevenLabs music generation failed: {e}")
        return GenerationResult(
            success=False,
            backend_name="elevenlabs",
            error=f"ElevenLabs music error: {e}",
        )


async def _generate_sfx_elevenlabs(config: GenerationConfig) -> GenerationResult:
    """Generate sound effects using the ElevenLabs Sound Effects API."""
    api_key = get_api_key("elevenlabs")
    if not api_key:
        return GenerationResult(
            success=False,
            backend_name="elevenlabs",
            error="ElevenLabs API key not found. Set ELEVENLABS_API_KEY environment variable.",
        )

    try:
        client = AsyncElevenLabs(api_key=api_key)

        prompt_influence = config.extra_params.get("prompt_influence", 0.3)
        kwargs: dict = {
            "text": config.prompt,
            "prompt_influence": prompt_influence,
        }
        if config.duration is not None:
            kwargs["duration_seconds"] = max(0.5, min(float(config.duration), 30.0))

        audio_iterator = await client.text_to_sound_effects.convert(**kwargs)

        chunks: list[bytes] = []
        async for chunk in audio_iterator:
            chunks.append(chunk)
        audio_bytes = b"".join(chunks)

        config.output_path.write_bytes(audio_bytes)
        file_size = config.output_path.stat().st_size

        return GenerationResult(
            success=True,
            output_path=config.output_path,
            media_type=MediaType.AUDIO,
            backend_name="elevenlabs",
            model_used="elevenlabs-sfx",
            file_size_bytes=file_size,
            metadata={
                "audio_type": "sound_effect",
                "duration_seconds": kwargs.get("duration_seconds"),
                "prompt_influence": prompt_influence,
                "format": "mp3",
            },
        )
    except Exception as e:
        logger.exception(f"ElevenLabs SFX generation failed: {e}")
        return GenerationResult(
            success=False,
            backend_name="elevenlabs",
            error=f"ElevenLabs SFX error: {e}",
        )


# ---------------------------------------------------------------------------
# OpenAI backend (speech only, unchanged)
# ---------------------------------------------------------------------------


async def _generate_audio_openai(config: GenerationConfig) -> GenerationResult:
    """Generate audio using OpenAI's TTS API.

    Uses streaming response for efficient file handling.

    Args:
        config: GenerationConfig with prompt (text to speak), output path, voice, etc.

    Returns:
        GenerationResult with generated audio info
    """
    api_key = get_api_key("openai")
    if not api_key:
        return GenerationResult(
            success=False,
            backend_name="openai",
            error="OpenAI API key not found. Set OPENAI_API_KEY environment variable.",
        )

    try:
        client = AsyncOpenAI(api_key=api_key)
        model = config.model or get_default_model("openai", MediaType.AUDIO)
        voice = config.voice or "alloy"

        # Validate voice
        if voice not in OPENAI_VOICES:
            logger.warning(
                f"Unknown voice '{voice}', using 'alloy'. " f"Available: {', '.join(OPENAI_VOICES)}",
            )
            voice = "alloy"

        # Determine format from output path extension
        ext = config.output_path.suffix.lstrip(".").lower()
        if ext not in AUDIO_FORMATS:
            ext = "mp3"  # Default format

        # Prepare request parameters
        request_params = {
            "model": model,
            "voice": voice,
            "input": config.prompt,
        }

        # Add instructions if provided (only for gpt-4o-mini-tts)
        instructions = config.extra_params.get("instructions")
        if instructions and model == "gpt-4o-mini-tts":
            request_params["instructions"] = instructions

        # Use streaming response for efficient file handling
        async with client.audio.speech.with_streaming_response.create(**request_params) as response:
            await response.stream_to_file(config.output_path)

        # Get file info
        file_size = config.output_path.stat().st_size

        return GenerationResult(
            success=True,
            output_path=config.output_path,
            media_type=MediaType.AUDIO,
            backend_name="openai",
            model_used=model,
            file_size_bytes=file_size,
            metadata={
                "voice": voice,
                "format": ext,
                "text_length": len(config.prompt),
                "instructions": instructions,
            },
        )

    except Exception as e:
        logger.exception(f"OpenAI TTS generation failed: {e}")
        return GenerationResult(
            success=False,
            backend_name="openai",
            error=f"OpenAI TTS error: {e}",
        )
