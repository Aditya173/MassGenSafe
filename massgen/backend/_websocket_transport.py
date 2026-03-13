"""
WebSocket transport for the OpenAI Responses API.
Persistent connection for response.create events.
See https://developers.openai.com/api/docs/guides/websocket-mode/
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets
from websockets.protocol import State as WSState

from ..logger_config import logger

DEFAULT_WS_URL = "wss://api.openai.com/v1/responses"
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_BASE_DELAY = 1.0


class WebSocketConnectionError(Exception):
    """Raised when a WebSocket connection cannot be established."""


def _extract_error_details(event: dict[str, Any]) -> tuple[str, str]:
    """Extract message and code from websocket error events."""
    nested_error = event.get("error")
    if isinstance(nested_error, dict):
        return (
            nested_error.get("message", "Unknown error"),
            nested_error.get("code", ""),
        )
    return (event.get("message", "Unknown error"), event.get("code", ""))


class WebSocketResponseTransport:
    """Persistent WebSocket transport for the OpenAI Responses API."""

    def __init__(
        self,
        api_key: str,
        url: str = DEFAULT_WS_URL,
        organization: str | None = None,
    ):
        self.api_key = api_key
        self.url = url
        self.organization = organization
        self._ws = None

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _build_response_create_event(self, payload: dict[str, Any]) -> str:
        """Wrap an API params dict as a response.create WebSocket event."""
        event = {"type": "response.create", **payload}
        return json.dumps(event)

    async def connect(self) -> None:
        """Establish the WebSocket connection with retry logic."""
        headers = self._build_headers()
        last_error: Exception | None = None

        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            try:
                self._ws = await websockets.connect(
                    self.url,
                    additional_headers=headers,
                    max_size=None,
                    ping_interval=30,
                    ping_timeout=10,
                )
                logger.info(
                    f"[WebSocket] Connected to {self.url} (attempt {attempt + 1})",
                )
                return
            except Exception as e:
                last_error = e
                if attempt < MAX_RECONNECT_ATTEMPTS - 1:
                    delay = RECONNECT_BASE_DELAY * (2**attempt)
                    logger.warning(
                        f"[WebSocket] Connection attempt {attempt + 1} failed: {e}. Retrying in {delay}s...",
                    )
                    await asyncio.sleep(delay)

        raise WebSocketConnectionError(
            f"Failed to connect to {self.url} after {MAX_RECONNECT_ATTEMPTS} attempts: {last_error}",
        )

    async def send_and_receive(
        self,
        api_params: dict[str, Any],
    ):
        """Send a response.create event and yield parsed response events.

        Args:
            api_params: The API params dict (same as HTTP body, minus stream/background).

        Yields:
            Parsed event dicts with a "type" field matching the HTTP SSE event types
            (e.g. "response.output_text.delta", "response.completed").
        """
        if self._ws is None:
            raise WebSocketConnectionError("Not connected. Call connect() first.")

        message = self._build_response_create_event(api_params)
        await self._ws.send(message)
        logger.debug("[WebSocket] Sent response.create event")

        async for raw_message in self._ws:
            event = json.loads(raw_message)
            event_type = event.get("type", "")
            logger.debug(f"[WebSocket] Received event: {event_type}")

            if event_type == "error":
                error_msg, error_code = _extract_error_details(event)
                logger.error(
                    f"[WebSocket] Server error: {error_msg} (code={error_code})",
                )
                raise WebSocketConnectionError(
                    f"WebSocket response.create failed: {error_msg}" + (f" (code={error_code})" if error_code else ""),
                )

            yield event

            if event_type in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                break

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                await self._ws.close()
                logger.info("[WebSocket] Connection closed")
            except Exception as e:
                logger.warning(f"[WebSocket] Error closing connection: {e}")
            finally:
                self._ws = None

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.state == WSState.OPEN
