"""Privacy and local-auth configuration for MassGen surfaces."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from os import getenv
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    value = getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _extract_bearer_token(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    if not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header[7:].strip()
    return token or None


_GENERATED_TOKENS: dict[str, str] = {}


def _resolve_token(*, env_name: str, cache_key: str, private_mode: bool) -> tuple[str | None, bool]:
    value = (getenv(env_name) or "").strip()
    if value:
        return value, False
    if not private_mode:
        return None, False
    if cache_key not in _GENERATED_TOKENS:
        _GENERATED_TOKENS[cache_key] = secrets.token_urlsafe(24)
    return _GENERATED_TOKENS[cache_key], True


@dataclass(frozen=True)
class PrivacySettings:
    """Unified privacy/auth settings used by WebUI and OpenAI-compatible server."""

    private_mode: bool
    allow_remote_access: bool
    allow_unsafe_sharing: bool
    disable_web_key_save: bool
    local_api_token: str | None
    local_api_token_generated: bool
    server_token: str | None
    server_token_generated: bool

    @property
    def web_key_save_enabled(self) -> bool:
        return not self.disable_web_key_save

    @classmethod
    def from_env(cls) -> PrivacySettings:
        private_mode = _env_bool("MASSGEN_PRIVATE_MODE", True)
        allow_remote_access = _env_bool("MASSGEN_ALLOW_REMOTE_ACCESS", False)
        allow_unsafe_sharing = _env_bool("MASSGEN_ALLOW_UNSAFE_SHARING", False)
        disable_web_key_save = _env_bool("MASSGEN_DISABLE_WEB_KEY_SAVE", private_mode)
        local_api_token, local_generated = _resolve_token(
            env_name="MASSGEN_LOCAL_API_TOKEN",
            cache_key="local_api_token",
            private_mode=private_mode,
        )
        server_token, server_generated = _resolve_token(
            env_name="MASSGEN_SERVER_TOKEN",
            cache_key="server_token",
            private_mode=private_mode,
        )
        return cls(
            private_mode=private_mode,
            allow_remote_access=allow_remote_access,
            allow_unsafe_sharing=allow_unsafe_sharing,
            disable_web_key_save=disable_web_key_save,
            local_api_token=local_api_token,
            local_api_token_generated=local_generated,
            server_token=server_token,
            server_token_generated=server_generated,
        )


def _extract_query_token(source: Any) -> str | None:
    try:
        token = source.get("token")
    except Exception:
        return None
    if not token:
        return None
    token = str(token).strip()
    return token or None


def extract_token_from_http(request: Any) -> str | None:
    auth_token = _extract_bearer_token(request.headers.get("authorization"))
    if auth_token:
        return auth_token
    return _extract_query_token(request.query_params)


def extract_token_from_websocket(websocket: Any) -> str | None:
    auth_token = _extract_bearer_token(websocket.headers.get("authorization"))
    if auth_token:
        return auth_token
    return _extract_query_token(websocket.query_params)


def token_matches(provided_token: str | None, expected_token: str | None) -> bool:
    if not expected_token:
        return False
    if not provided_token:
        return False
    return secrets.compare_digest(provided_token, expected_token)
