"""
Dynamic model catalog fetcher for chat completion providers.
Fetches model lists from provider APIs with caching.

Based on research of official provider APIs.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

# Cache directory
CACHE_DIR = Path.home() / ".massgen" / "model_cache"
CACHE_DURATION = timedelta(hours=24)  # Cache for 24 hours


def ensure_cache_dir():
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_path(provider: str, cache_kind: str = "models") -> Path:
    """Get cache file path for a provider and cache kind."""
    return CACHE_DIR / f"{provider}_{cache_kind}.json"


def is_cache_valid(cache_path: Path) -> bool:
    """Check if cache file exists and is still valid."""
    if not cache_path.exists():
        return False

    try:
        with open(cache_path) as f:
            data = json.load(f)
            cached_at = datetime.fromisoformat(data.get("cached_at", ""))
            return datetime.now() - cached_at < CACHE_DURATION
    except (json.JSONDecodeError, ValueError, KeyError):
        return False


def read_cache(cache_path: Path, key: str = "models") -> list[Any] | None:
    """Read a cached list payload."""
    try:
        with open(cache_path) as f:
            data = json.load(f)
            cached_value = data.get(key, [])
            if isinstance(cached_value, list):
                return cached_value
            return None
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def write_cache(cache_path: Path, values: list[Any], key: str = "models"):
    """Write a cached list payload."""
    ensure_cache_dir()
    data = {key: values, "cached_at": datetime.now().isoformat()}
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)


async def fetch_openrouter_models(api_key: str | None = None) -> list[str]:
    """Fetch model list from OpenRouter API.

    OpenRouter's /models endpoint works without authentication.

    Args:
        api_key: OpenRouter API key (optional, not required for listing models)

    Returns:
        List of model IDs
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # OpenRouter allows listing models without auth
            headers = {}
            if api_key or os.getenv("OPENROUTER_API_KEY"):
                headers["Authorization"] = f"Bearer {api_key or os.getenv('OPENROUTER_API_KEY')}"

            response = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            tool_supporting_models = []
            for model in models:
                supported_params = model.get("supported_parameters", [])
                # Check if model supports tool calling
                if "tools" in supported_params:
                    tool_supporting_models.append(model["id"])
            return tool_supporting_models
    except (httpx.HTTPError, KeyError, ValueError):
        return []


async def fetch_poe_models() -> list[str]:
    """Fetch model list from POE API.

    Returns:
        List of model IDs
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("https://api.poe.com/v1/models")
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
    except (httpx.HTTPError, KeyError, ValueError):
        return []


async def fetch_together_models(
    api_key: str | None = None,
    default_model: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
) -> list[str]:
    """Fetch model list from Together AI API.

    Filters to only chat/language models, sorted by creation date (newest first).

    Args:
        api_key: Together API key
        default_model: Model to put at the top of the list

    Returns:
        List of model IDs
    """
    if not api_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.together.xyz/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()

            # Together returns list directly or in data field
            models_data = data if isinstance(data, list) else data.get("data", [])

            # Filter to chat and language models only (exclude image, embedding, moderation, rerank)
            chat_types = {"chat", "language", "code"}
            models_data = [m for m in models_data if m.get("type") in chat_types]

            # Sort by created timestamp descending (newest first)
            models_data.sort(key=lambda m: m.get("created", 0), reverse=True)

            models = [model["id"] for model in models_data]

            # Move default model to top if present
            if default_model and default_model in models:
                models.remove(default_model)
                models.insert(0, default_model)

            return models
    except (httpx.HTTPError, KeyError, ValueError):
        return []


def _is_chat_model(model_id: str, provider: str = "openai") -> bool:
    """Check if a model is a chat/text model (not specialized).

    Filters out across all providers:
    - Audio/speech models (whisper, tts, *-audio*, orpheus)
    - Image/video models (dall-e, sora, *-image*)
    - Embedding models (text-embedding-*, embed)
    - Moderation/safety models (*-guard*, *-safeguard*, *-moderation*)
    - Fine-tuned models (ft:*)

    Provider-specific filtering is also applied.
    """
    model_lower = model_id.lower()

    # Universal exclude patterns (apply to all providers)
    universal_exclude_prefixes = [
        "whisper",  # speech recognition
        "tts-",  # text-to-speech
        "text-embedding",  # embeddings
        "dall-e",  # image generation
        "sora",  # video generation
        "ft:",  # fine-tuned
    ]

    universal_exclude_contains = [
        "-guard-",  # safety models
        "-guard",  # safety models (at end)
        "-safeguard",  # safety models
        "-moderation",  # moderation
        "-audio-",  # audio
        "-transcribe",  # transcription
        "-tts",  # text-to-speech
        "-embed",  # embeddings
        "orpheus",  # speech synthesis (Groq)
    ]

    # Check universal excludes
    for prefix in universal_exclude_prefixes:
        if model_lower.startswith(prefix):
            return False

    for pattern in universal_exclude_contains:
        if pattern in model_lower:
            return False

    # Provider-specific filtering
    if provider == "openai":
        # OpenAI-specific excludes
        openai_exclude_prefixes = [
            "babbage",  # legacy
            "davinci",  # legacy
            "computer-use",  # computer use
            "codex-mini-latest",  # standalone codex
            "gpt-image",  # image generation
            "gpt-audio",  # audio
            "gpt-realtime",  # realtime
            "chatgpt-image",  # image generation
        ]

        openai_exclude_contains = [
            "-audio",  # audio models
            "-realtime",  # realtime models
            "-image-",  # image models
            "-search-api",  # search API (not chat)
            "-deep-research",  # deep research (not standard chat)
            "-instruct",  # instruct models (legacy)
        ]

        for prefix in openai_exclude_prefixes:
            if model_lower.startswith(prefix):
                return False

        for pattern in openai_exclude_contains:
            if pattern in model_lower:
                return False

        # OpenAI: only keep known chat model prefixes
        valid_prefixes = ["gpt-", "o1", "o3", "o4", "chatgpt-4o"]
        return any(model_lower.startswith(p) for p in valid_prefixes)

    # For other providers, if it passed universal filters, it's likely a chat model
    return True


async def fetch_openai_compatible_models(
    base_url: str,
    api_key: str | None = None,
    sort_by_created: bool = False,
    default_model: str | None = None,
    filter_chat_models: bool = False,
    provider: str = "openai",
) -> list[str]:
    """Fetch model list from OpenAI-compatible API endpoint.

    Args:
        base_url: Base URL of the API (e.g., "https://api.groq.com/openai/v1")
        api_key: API key for authentication
        sort_by_created: Sort by creation date (newest first)
        default_model: Model to put at the top of the list
        filter_chat_models: Filter to only chat models
        provider: Provider name for provider-specific filtering

    Returns:
        List of model IDs
    """
    if not api_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            models_data = data.get("data", [])

            # Filter to chat models if requested
            if filter_chat_models:
                models_data = [m for m in models_data if _is_chat_model(m["id"], provider)]

            if sort_by_created:
                # Sort by created timestamp descending (newest first)
                models_data.sort(key=lambda m: m.get("created", 0), reverse=True)

            models = [model["id"] for model in models_data]

            # Move default model to top if specified
            if default_model and default_model in models:
                models.remove(default_model)
                models.insert(0, default_model)

            return models
    except (httpx.HTTPError, KeyError, ValueError):
        return []


def _extract_model_id(model: Any) -> str | None:
    """Extract a normalized model id from SDK or dict model payloads."""
    if model is None:
        return None

    model_id = getattr(model, "id", None)
    if model_id is None and isinstance(model, dict):
        model_id = model.get("id")

    if model_id is None:
        return None

    normalized = str(model_id).strip()
    return normalized or None


def _extract_model_field(model: Any, field_name: str) -> Any:
    """Extract a field from SDK or dict model payloads."""
    value = getattr(model, field_name, None)
    if value is None and isinstance(model, dict):
        value = model.get(field_name)
    return value


def _normalize_reasoning_efforts(value: Any) -> list[str]:
    """Normalize reasoning-effort metadata from SDK payloads."""
    if not value:
        return []

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        effort = str(item).strip()
        if not effort or effort in seen:
            continue
        seen.add(effort)
        normalized.append(effort)
    return normalized


def _prioritize_default_model(
    items: list[Any],
    default_model: str | None,
    key_fn,
) -> list[Any]:
    """Move the default model to the front when present."""
    if not default_model:
        return items

    default_index = next(
        (index for index, item in enumerate(items) if key_fn(item) == default_model),
        None,
    )
    if default_index is None:
        return items

    prioritized = list(items)
    default_item = prioritized.pop(default_index)
    prioritized.insert(0, default_item)
    return prioritized


def _normalize_copilot_model_metadata(
    models_data: list[Any],
    *,
    default_model: str | None,
) -> list[dict[str, Any]]:
    """Normalize Copilot SDK model payloads into stable metadata dicts."""
    metadata: list[dict[str, Any]] = []
    seen: set[str] = set()

    for model in models_data:
        model_id = _extract_model_id(model)
        if not model_id or model_id in seen:
            continue

        seen.add(model_id)
        name = _extract_model_field(model, "name")
        default_reasoning_effort = _extract_model_field(model, "default_reasoning_effort")
        supported_reasoning_efforts = _normalize_reasoning_efforts(
            _extract_model_field(model, "supported_reasoning_efforts"),
        )

        metadata.append(
            {
                "id": model_id,
                "name": str(name).strip() if name else model_id,
                "supported_reasoning_efforts": supported_reasoning_efforts,
                "default_reasoning_effort": (str(default_reasoning_effort).strip() if default_reasoning_effort else None),
            },
        )

    return _prioritize_default_model(metadata, default_model, lambda item: item["id"])


async def fetch_copilot_model_metadata(
    default_model: str | None = "gpt-5-mini",
) -> list[dict[str, Any]]:
    """Fetch runtime-available Copilot model metadata from the SDK."""
    try:
        from copilot import CopilotClient
    except ImportError:
        return []

    client = CopilotClient()
    try:
        await client.start()
        models_data = await client.list_models()
        return _normalize_copilot_model_metadata(
            list(models_data or []),
            default_model=default_model,
        )
    except Exception:
        return []
    finally:
        try:
            await client.stop()
        except Exception:
            pass


async def fetch_copilot_models(default_model: str | None = "gpt-5-mini") -> list[str]:
    """Fetch runtime-available models from the GitHub Copilot SDK.

    The SDK resolves availability using the local Copilot runtime, so results can
    vary by installed CLI/runtime version, authentication state, plan, and org
    restrictions. Failures return an empty list so callers can fall back to the
    static capabilities registry.
    """
    metadata = await fetch_copilot_model_metadata(default_model=default_model)
    return [model["id"] for model in metadata]


async def get_models_for_provider(provider: str, use_cache: bool = True) -> list[str]:
    """Get model list for a provider, using cache if available.

    Args:
        provider: Provider name (e.g., "openrouter", "groq")
        use_cache: Whether to use cached results

    Returns:
        List of model IDs
    """
    cache_path = get_cache_path(provider)

    # Try cache first
    if use_cache and is_cache_valid(cache_path):
        cached_models = read_cache(cache_path)
        if cached_models:
            return cached_models

    # Fetch from API based on provider
    models = []

    if provider == "openrouter":
        models = await fetch_openrouter_models()
    elif provider == "poe":
        models = await fetch_poe_models()
    elif provider == "groq":
        # Filter out whisper, guard, and orpheus models
        models = await fetch_openai_compatible_models(
            "https://api.groq.com/openai/v1",
            os.getenv("GROQ_API_KEY"),
            sort_by_created=True,
            filter_chat_models=True,
            provider="groq",
        )
    elif provider == "cerebras":
        models = await fetch_openai_compatible_models("https://api.cerebras.ai/v1", os.getenv("CEREBRAS_API_KEY"))
    elif provider == "together":
        # Use dedicated fetcher that filters by type field
        models = await fetch_together_models(os.getenv("TOGETHER_API_KEY"))
    elif provider == "nebius":
        models = await fetch_openai_compatible_models(
            "https://api.studio.nebius.com/v1",
            os.getenv("NEBIUS_API_KEY"),
        )
    elif provider == "fireworks":
        # Fireworks uses OpenAI-compatible endpoint
        models = await fetch_openai_compatible_models(
            "https://api.fireworks.ai/inference/v1",
            os.getenv("FIREWORKS_API_KEY"),
        )
    elif provider == "moonshot":
        # Moonshot/Kimi uses OpenAI-compatible endpoint
        models = await fetch_openai_compatible_models("https://api.moonshot.cn/v1", os.getenv("MOONSHOT_API_KEY"))
    elif provider == "nvidia_nim":
        # Nvidia NIM uses OpenAI-compatible endpoint
        models = await fetch_openai_compatible_models("https://integrate.api.nvidia.com/v1", os.getenv("NGC_API_KEY"))
    elif provider == "qwen":
        # Qwen uses DashScope API (OpenAI-compatible)
        models = await fetch_openai_compatible_models(
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            os.getenv("QWEN_API_KEY"),
        )
    elif provider == "openai":
        # Filter to chat models, sort by created date (newest first), recommended model at top
        models = await fetch_openai_compatible_models(
            "https://api.openai.com/v1",
            os.getenv("OPENAI_API_KEY"),
            sort_by_created=True,
            default_model="gpt-5.4",
            filter_chat_models=True,
            provider="openai",
        )
    elif provider == "copilot":
        from massgen.backend.capabilities import BACKEND_CAPABILITIES

        caps = BACKEND_CAPABILITIES.get("copilot")
        models = await fetch_copilot_models(
            default_model=caps.default_model if caps else "gpt-5-mini",
        )

    # Cache the results
    if models:
        write_cache(cache_path, models)

    return models


async def get_model_metadata_for_provider(
    provider: str,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Get model metadata for a provider, using cache if available."""
    cache_path = get_cache_path(provider, "model_metadata")

    if use_cache and is_cache_valid(cache_path):
        cached_metadata = read_cache(cache_path, key="metadata")
        if cached_metadata:
            return cached_metadata

    metadata: list[dict[str, Any]] = []
    if provider == "copilot":
        from massgen.backend.capabilities import BACKEND_CAPABILITIES

        caps = BACKEND_CAPABILITIES.get("copilot")
        metadata = await fetch_copilot_model_metadata(
            default_model=caps.default_model if caps else "gpt-5-mini",
        )

    if metadata:
        write_cache(cache_path, metadata, key="metadata")

    return metadata


def get_model_metadata_for_provider_sync(
    provider: str,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Synchronous wrapper for get_model_metadata_for_provider."""
    from massgen.utils.async_helpers import run_async_safely

    try:
        return run_async_safely(
            get_model_metadata_for_provider(provider, use_cache),
            timeout=15,
        )
    except Exception:
        return []


def get_models_for_provider_sync(provider: str, use_cache: bool = True) -> list[str]:
    """Synchronous wrapper for get_models_for_provider.

    Args:
        provider: Provider name (e.g., "openrouter", "groq")
        use_cache: Whether to use cached results

    Returns:
        List of model IDs
    """
    from massgen.utils.async_helpers import run_async_safely

    try:
        return run_async_safely(get_models_for_provider(provider, use_cache), timeout=15)
    except Exception:
        # If async fails, return empty list
        return []
