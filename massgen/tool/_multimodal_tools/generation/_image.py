"""
Image generation backends: OpenAI, Google Imagen, OpenRouter.

This module contains all image generation implementations that are
routed through by generate_media when mode="image".
"""

import base64

import requests
from openai import AsyncOpenAI

from massgen.logger_config import logger
from massgen.tool._multimodal_tools.generation._base import (
    GenerationConfig,
    GenerationResult,
    MediaType,
    get_api_key,
    get_default_model,
)


async def generate_image(config: GenerationConfig) -> GenerationResult:
    """Generate an image using the selected backend.

    Routes to the appropriate backend based on config.backend.

    Args:
        config: GenerationConfig with prompt, output_path, backend, etc.

    Returns:
        GenerationResult with success status and file info
    """
    backend = config.backend or "openai"  # Default if not specified

    if backend == "google":
        return await _generate_image_google(config)
    elif backend == "openrouter":
        return await _generate_image_openrouter(config)
    else:  # openai (default)
        return await _generate_image_openai(config)


async def _generate_image_openai(config: GenerationConfig) -> GenerationResult:
    """Generate image using OpenAI's Responses API.

    Uses the image_generation tool via the responses endpoint.
    This approach supports multi-turn conversations and is the recommended
    method for agentic image generation workflows.

    Args:
        config: GenerationConfig with prompt and output path

    Returns:
        GenerationResult with generated image info
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
        model = config.model or get_default_model("openai", MediaType.IMAGE)

        # Build input content (supports optional input_images for image-to-image)
        if config.input_images:
            input_content = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": config.prompt}, *config.input_images],
                },
            ]
        else:
            input_content = config.prompt

        # Generate image using OpenAI Responses API (async)
        response = await client.responses.create(
            model=model,
            input=input_content,
            tools=[{"type": "image_generation"}],
        )

        # Extract image data from response
        image_data = [output.result for output in response.output if output.type == "image_generation_call"]

        if not image_data:
            return GenerationResult(
                success=False,
                backend_name="openai",
                model_used=model,
                error="No image data in response",
            )

        # Save the first image
        image_bytes = base64.b64decode(image_data[0])
        config.output_path.write_bytes(image_bytes)

        return GenerationResult(
            success=True,
            output_path=config.output_path,
            media_type=MediaType.IMAGE,
            backend_name="openai",
            model_used=model,
            file_size_bytes=len(image_bytes),
            metadata={"total_images": len(image_data)},
        )

    except Exception as e:
        logger.exception(f"OpenAI image generation failed: {e}")
        return GenerationResult(
            success=False,
            backend_name="openai",
            error=f"OpenAI API error: {str(e)}",
        )


async def _generate_image_google(config: GenerationConfig) -> GenerationResult:
    """Generate image using Google Imagen API.

    Uses the google-genai SDK to generate images via Imagen 3.

    Args:
        config: GenerationConfig with prompt and output path

    Returns:
        GenerationResult with generated image info
    """
    api_key = get_api_key("google")
    if not api_key:
        return GenerationResult(
            success=False,
            backend_name="google",
            error="Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.",
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model = config.model or get_default_model("google", MediaType.IMAGE)

        # Prepare config
        gen_config = types.GenerateImagesConfig(
            number_of_images=1,
            output_mime_type="image/png",
        )

        # Add aspect ratio if specified
        if config.aspect_ratio:
            gen_config.aspect_ratio = config.aspect_ratio

        # Generate image
        response = client.models.generate_images(
            model=model,
            prompt=config.prompt,
            config=gen_config,
        )

        if not response.generated_images:
            return GenerationResult(
                success=False,
                backend_name="google",
                model_used=model,
                error="No images generated",
            )

        # Save the first image
        generated_image = response.generated_images[0]
        generated_image.image.save(str(config.output_path))

        file_size = config.output_path.stat().st_size

        return GenerationResult(
            success=True,
            output_path=config.output_path,
            media_type=MediaType.IMAGE,
            backend_name="google",
            model_used=model,
            file_size_bytes=file_size,
            metadata={"total_images": len(response.generated_images)},
        )

    except Exception as e:
        logger.exception(f"Google Imagen generation failed: {e}")
        return GenerationResult(
            success=False,
            backend_name="google",
            error=f"Google Imagen error: {str(e)}",
        )


async def _generate_image_openrouter(config: GenerationConfig) -> GenerationResult:
    """Generate image using OpenRouter API.

    Uses the chat completions endpoint with modalities=["image", "text"].

    Args:
        config: GenerationConfig with prompt and output path

    Returns:
        GenerationResult with generated image info
    """
    api_key = get_api_key("openrouter")
    if not api_key:
        return GenerationResult(
            success=False,
            backend_name="openrouter",
            error="OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.",
        )

    try:
        model = config.model or get_default_model("openrouter", MediaType.IMAGE)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://massgen.dev",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": config.prompt}],
            "modalities": ["image", "text"],
        }

        # Add aspect ratio if specified
        if config.aspect_ratio:
            payload["image_config"] = {"aspect_ratio": config.aspect_ratio}

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        # Extract image from response
        if not result.get("choices"):
            return GenerationResult(
                success=False,
                backend_name="openrouter",
                model_used=model,
                error="No choices in response",
            )

        message = result["choices"][0].get("message", {})
        images = message.get("images", [])

        if not images:
            # Check if content contains base64 image
            content = message.get("content", "")
            if "data:image" in content:
                # Extract base64 from data URL
                import re

                match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", content)
                if match:
                    image_bytes = base64.b64decode(match.group(1))
                    config.output_path.write_bytes(image_bytes)

                    return GenerationResult(
                        success=True,
                        output_path=config.output_path,
                        media_type=MediaType.IMAGE,
                        backend_name="openrouter",
                        model_used=model,
                        file_size_bytes=len(image_bytes),
                    )

            return GenerationResult(
                success=False,
                backend_name="openrouter",
                model_used=model,
                error="No image in response",
            )

        # Process first image
        image_url = images[0].get("image_url", {}).get("url", "")
        if not image_url:
            return GenerationResult(
                success=False,
                backend_name="openrouter",
                model_used=model,
                error="No image URL in response",
            )

        # Extract base64 data from data URL
        if image_url.startswith("data:"):
            base64_data = image_url.split(",")[1]
            image_bytes = base64.b64decode(base64_data)
        else:
            # Fetch from URL
            img_response = requests.get(image_url, timeout=60)
            img_response.raise_for_status()
            image_bytes = img_response.content

        # Save image
        config.output_path.write_bytes(image_bytes)

        return GenerationResult(
            success=True,
            output_path=config.output_path,
            media_type=MediaType.IMAGE,
            backend_name="openrouter",
            model_used=model,
            file_size_bytes=len(image_bytes),
            metadata={"total_images": len(images)},
        )

    except requests.exceptions.RequestException as e:
        logger.exception(f"OpenRouter image generation failed: {e}")
        return GenerationResult(
            success=False,
            backend_name="openrouter",
            error=f"OpenRouter API error: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"OpenRouter image generation failed: {e}")
        return GenerationResult(
            success=False,
            backend_name="openrouter",
            error=f"OpenRouter error: {str(e)}",
        )
