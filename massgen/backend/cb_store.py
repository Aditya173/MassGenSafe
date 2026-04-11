"""State stores for LLM circuit breakers."""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Protocol, runtime_checkable

DEFAULT_CIRCUIT_BREAKER_STATE: dict[str, Any] = {
    "state": "closed",
    "failure_count": 0,
    "last_failure_time": 0.0,
    "open_until": 0.0,
    "half_open_probe_active": False,
}


def _default_state() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_CIRCUIT_BREAKER_STATE)


@runtime_checkable
class CircuitBreakerStore(Protocol):
    """Protocol for shared circuit breaker state stores."""

    def get_state(self, backend: str) -> dict:
        """Return the circuit breaker state for a backend."""

    def set_state(self, backend: str, state: dict) -> None:
        """Persist the complete circuit breaker state for a backend."""

    def cas_state(self, backend: str, expected_state: str, updates: dict) -> bool:
        """Apply updates if the current state field matches expected_state."""

    def increment_failure(self, backend: str) -> int:
        """Atomically increment and return the backend failure count."""

    def clear(self, backend: str) -> None:
        """Remove persisted state for a backend."""


class InMemoryStore:
    """Thread-safe in-process circuit breaker store."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._storage: dict[str, dict] = {}

    def get_state(self, backend: str) -> dict:
        with self._lock:
            if backend not in self._storage:
                self._storage[backend] = _default_state()
            return copy.deepcopy(self._storage[backend])

    def set_state(self, backend: str, state: dict) -> None:
        with self._lock:
            complete_state = _default_state()
            complete_state.update(copy.deepcopy(state))
            self._storage[backend] = complete_state

    def cas_state(self, backend: str, expected_state: str, updates: dict) -> bool:
        with self._lock:
            if backend not in self._storage:
                self._storage[backend] = _default_state()
            if self._storage[backend].get("state") != expected_state:
                return False
            self._storage[backend].update(copy.deepcopy(updates))
            return True

    def increment_failure(self, backend: str) -> int:
        with self._lock:
            state = self.get_state(backend)
            state["failure_count"] = int(state["failure_count"]) + 1
            self.set_state(backend, state)
            return int(state["failure_count"])

    def clear(self, backend: str) -> None:
        with self._lock:
            self._storage.pop(backend, None)


class RedisStore:
    """Redis hash-backed circuit breaker store."""

    _CAS_SCRIPT = """
local current_state = redis.call("HGET", KEYS[1], "state")
if current_state == false then
    current_state = "closed"
end
if current_state ~= ARGV[1] then
    return 0
end
for i = 3, #ARGV, 2 do
redis.call("HSET", KEYS[1], ARGV[i], ARGV[i + 1])
end
redis.call("EXPIRE", KEYS[1], ARGV[2])
return 1
"""

    _INCREMENT_SCRIPT = """
local count = redis.call("HINCRBY", KEYS[1], "failure_count", 1)
if redis.call("HGET", KEYS[1], "state") == false then
    redis.call(
        "HSET",
        KEYS[1],
        "state",
        "closed",
        "last_failure_time",
        "0.0",
        "open_until",
        "0.0",
        "half_open_probe_active",
        "False"
    )
end
local existing_ttl = redis.call("TTL", KEYS[1])
local new_ttl = tonumber(ARGV[1])
if existing_ttl == -2 then existing_ttl = 0 end
if existing_ttl > new_ttl then new_ttl = existing_ttl end
redis.call("EXPIRE", KEYS[1], new_ttl)
return count
"""

    def __init__(
        self,
        redis_client: Any,
        ttl: int = 3600,
        key_prefix: str = "massgen:cb",
    ) -> None:
        self._client = redis_client
        self._ttl = ttl
        self._key_prefix = key_prefix

    def get_state(self, backend: str) -> dict:
        raw_state = self._client.hgetall(self._key(backend))
        state = _default_state()
        for raw_key, raw_value in raw_state.items():
            key = self._decode(raw_key)
            value = self._decode(raw_value)
            if key == "failure_count":
                state[key] = int(value)
            elif key in {"last_failure_time", "open_until"}:
                state[key] = float(value)
            elif key == "half_open_probe_active":
                state[key] = value == "True"
            elif key == "state":
                state[key] = value
        return state

    def set_state(self, backend: str, state: dict) -> None:
        key = self._key(backend)
        mapping = {field: self._to_redis_value(state[field]) for field in DEFAULT_CIRCUIT_BREAKER_STATE}
        self._client.hset(key, mapping=mapping)
        self._client.expire(key, self._compute_ttl(state))

    def cas_state(self, backend: str, expected_state: str, updates: dict) -> bool:
        args: list[Any] = [expected_state, str(self._compute_ttl(updates))]
        for field, value in updates.items():
            args.extend([field, self._to_redis_value(value)])
        try:
            result = self._client.eval(self._CAS_SCRIPT, 1, self._key(backend), *args)
        except Exception as exc:
            if not self._script_unavailable(exc):
                raise
            return self._cas_state_without_lua(backend, expected_state, updates)
        return int(result) == 1

    def increment_failure(self, backend: str) -> int:
        try:
            result = self._client.eval(
                self._INCREMENT_SCRIPT,
                1,
                self._key(backend),
                str(self._ttl),
            )
        except Exception as exc:
            if not self._script_unavailable(exc):
                raise
            return self._increment_failure_without_lua(backend)
        return int(result)

    def clear(self, backend: str) -> None:
        self._client.delete(self._key(backend))

    def _key(self, backend: str) -> str:
        return f"{self._key_prefix}:{backend}"

    @staticmethod
    def _decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _to_redis_value(value: Any) -> str:
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value)

    @staticmethod
    def _script_unavailable(exc: Exception) -> bool:
        # Match only known Lua/EVAL unavailability signatures.
        # Require "unknown command" context to avoid classifying READONLY,
        # ACL-denied, proxy, or other operational errors as Lua unavailability.
        message = str(exc).lower()
        unknown_command = "unknown command" in message or "unknown redis command" in message or "err unknown command" in message
        mentions_script_command = "eval" in message or "evalsha" in message
        return "lupa" in message or (unknown_command and mentions_script_command)

    def _compute_ttl(self, updates: dict) -> int:
        if updates.get("state") == "open":
            open_until = float(updates.get("open_until", 0))
            remaining = int(open_until - time.monotonic())
            return max(1, self._ttl, remaining + 60)
        return max(1, self._ttl)

    def _cas_state_without_lua(
        self,
        backend: str,
        expected_state: str,
        updates: dict,
    ) -> bool:
        key = self._key(backend)
        for attempt in range(3):
            pipe = self._client.pipeline(True)
            try:
                pipe.watch(key)
                current = self._client.hget(key, "state")
                if current is not None:
                    current = self._decode(current)
                else:
                    current = "closed"
                if current != expected_state:
                    pipe.reset()
                    return False
                pipe.multi()
                for field, value in updates.items():
                    pipe.hset(key, field, self._to_redis_value(value))
                effective_ttl = self._compute_ttl(updates)
                pipe.expire(key, effective_ttl)
                pipe.execute()
                return True
            except Exception as exc:
                err_msg = str(exc).lower()
                if "watch" in err_msg or "multi" in err_msg or "wrongtype" in err_msg or "execabort" in err_msg:
                    time.sleep(0.001 * (2**attempt))
                    continue
                raise
        return False

    def _increment_failure_without_lua(self, backend: str) -> int:
        key = self._key(backend)
        for attempt in range(3):
            pipe = self._client.pipeline(True)
            try:
                pipe.watch(key)
                state = self.get_state(backend)
                state["failure_count"] = int(state["failure_count"]) + 1
                pipe.multi()
                mapping = {field: self._to_redis_value(state[field]) for field in DEFAULT_CIRCUIT_BREAKER_STATE}
                pipe.hset(key, mapping=mapping)
                pipe.expire(key, self._compute_ttl(state))
                pipe.execute()
                return int(state["failure_count"])
            except Exception as exc:
                err_msg = str(exc).lower()
                if "watch" in err_msg or "execabort" in err_msg:
                    time.sleep(0.001 * (2**attempt))
                    continue
                raise
        raise RuntimeError(
            f"Failed to atomically increment failure count for {backend!r} after 3 retries",
        )


def make_store(backend: str = "memory", **kwargs: Any) -> CircuitBreakerStore:
    """Create a circuit breaker state store."""
    if backend == "memory":
        return InMemoryStore()
    if backend == "redis":
        return RedisStore(
            kwargs["redis_client"],
            ttl=kwargs.get("ttl", 3600),
        )
    raise ValueError(f"Unknown circuit breaker store backend: {backend}")
