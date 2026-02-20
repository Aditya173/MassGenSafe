"""
Audio generation backends: OpenAI TTS.

This module contains all audio generation implementations that are
routed through by generate_media when mode="audio".
"""

from openai import AsyncOpenAI

from massgen.logger_config import logger
from massgen.tool._multimodal_tools.generation._base import (
    GenerationConfig,
    GenerationResult,
    MediaType,
    get_api_key,
    get_default_model,
)

# Available voices for OpenAI TTS
OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "sage"]

# Supported audio formats
AUDIO_FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]


async def generate_audio(config: GenerationConfig) -> GenerationResult:
    """Generate audio using the selected backend.

    Currently only OpenAI TTS is supported.

    Args:
        config: GenerationConfig with prompt (text), output_path, voice, etc.

    Returns:
        GenerationResult with success status and file info
    """
    # Currently only OpenAI TTS is supported (backend selection ignored for now)
    _ = config.backend  # Reserved for future multi-backend support
    return await _generate_audio_openai(config)


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
            response.stream_to_file(config.output_path)

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
            error=f"OpenAI TTS error: {str(e)}",
        )
