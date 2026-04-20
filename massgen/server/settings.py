"""
Server settings for MassGen HTTP server.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv

from massgen.privacy import PrivacySettings


@dataclass(frozen=True)
class ServerSettings:
    """Settings for the MassGen HTTP server."""

    host: str = "127.0.0.1"
    port: int = 4000
    default_config: str | None = None
    debug: bool = False
    private_mode: bool = True
    allow_remote_access: bool = False
    allow_unsafe_sharing: bool = False
    server_token: str | None = None
    server_token_generated: bool = False

    @classmethod
    def from_env(cls) -> ServerSettings:
        """Load settings from environment variables."""

        def _get_bool(name: str, default: bool = False) -> bool:
            v = getenv(name)
            if v is None:
                return default
            return v.strip().lower() in {"1", "true", "yes", "y", "on"}

        privacy = PrivacySettings.from_env()

        host = getenv("MASSGEN_SERVER_HOST", cls.host)
        port_s = getenv("MASSGEN_SERVER_PORT")
        port = int(port_s) if port_s else cls.port
        default_config = getenv("MASSGEN_SERVER_DEFAULT_CONFIG", cls.default_config or "")

        if privacy.private_mode and not privacy.allow_remote_access and host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"

        return cls(
            host=host,
            port=port,
            default_config=default_config or None,
            debug=_get_bool("MASSGEN_SERVER_DEBUG", cls.debug),
            private_mode=privacy.private_mode,
            allow_remote_access=privacy.allow_remote_access,
            allow_unsafe_sharing=privacy.allow_unsafe_sharing,
            server_token=privacy.server_token,
            server_token_generated=privacy.server_token_generated,
        )
