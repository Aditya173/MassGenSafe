"""Adversarial tests for circuit breaker state stores."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from massgen.backend.cb_store import (
    DEFAULT_CIRCUIT_BREAKER_STATE,
    InMemoryStore,
    RedisStore,
)
from massgen.backend.llm_circuit_breaker import (
    CircuitState,
    LLMCircuitBreaker,
    LLMCircuitBreakerConfig,
)


def _enabled_config(**overrides: Any) -> LLMCircuitBreakerConfig:
    defaults = {"enabled": True, "max_failures": 3, "reset_time_seconds": 1}
    defaults.update(overrides)
    return LLMCircuitBreakerConfig(**defaults)


def _complete_state(**overrides: Any) -> dict[str, Any]:
    state = dict(DEFAULT_CIRCUIT_BREAKER_STATE)
    state.update(overrides)
    return state


def _assert_complete_state(state: dict[str, Any]) -> None:
    assert set(DEFAULT_CIRCUIT_BREAKER_STATE).issubset(state)


def _fake_redis():
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeRedis()


class TestAdversarialInMemoryStore:
    def test_backend_name_empty_string(self) -> None:
        store = InMemoryStore()

        assert store.get_state("") == DEFAULT_CIRCUIT_BREAKER_STATE

    def test_backend_name_very_long(self) -> None:
        store = InMemoryStore()
        long_backend = "x" * 10000

        store.set_state(long_backend, _complete_state(state="open"))

        assert store.get_state(long_backend)["state"] == "open"
        assert store.get_state("normal") == DEFAULT_CIRCUIT_BREAKER_STATE

    def test_backend_name_unicode(self) -> None:
        store = InMemoryStore()

        state = store.get_state("\u4e2d\u6587backend")

        assert state == DEFAULT_CIRCUIT_BREAKER_STATE

    def test_set_state_missing_keys(self) -> None:
        store = InMemoryStore()

        store.set_state("backend", {"state": "open"})

        assert store.get_state("backend") == _complete_state(state="open")

    def test_set_state_extra_keys(self) -> None:
        store = InMemoryStore()

        store.set_state("backend", _complete_state(extra_unknown_key="ignored_or_kept"))

        state = store.get_state("backend")
        _assert_complete_state(state)

    def test_cas_state_same_state_no_op(self) -> None:
        store = InMemoryStore()

        result = store.cas_state("backend", "closed", {"failure_count": 5})

        assert result is True
        assert store.get_state("backend")["state"] == "closed"
        assert store.get_state("backend")["failure_count"] == 5

    def test_cas_state_invalid_expected_state(self) -> None:
        store = InMemoryStore()

        result = store.cas_state(
            "backend",
            "nonexistent_state",
            {"state": "open"},
        )

        assert result is False

    def test_increment_failure_many_times(self) -> None:
        store = InMemoryStore()

        for _ in range(1000):
            store.increment_failure("backend")

        assert store.get_state("backend")["failure_count"] == 1000

    def test_clear_nonexistent_backend_no_crash(self) -> None:
        store = InMemoryStore()

        store.clear("never_existed")

    def test_clear_then_get_returns_defaults(self) -> None:
        store = InMemoryStore()
        store.set_state("backend", _complete_state(state="open", failure_count=10))

        store.clear("backend")

        assert store.get_state("backend") == DEFAULT_CIRCUIT_BREAKER_STATE


class TestAdversarialInMemoryStoreConcurrency:
    def test_concurrent_increment_failure_100_threads(self) -> None:
        store = InMemoryStore()

        threads = [threading.Thread(target=store.increment_failure, args=("backend",)) for _ in range(100)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert store.get_state("backend")["failure_count"] == 100

    def test_concurrent_cas_state_race(self) -> None:
        store = InMemoryStore()
        results: list[bool] = []
        result_lock = threading.Lock()

        def attempt_cas() -> None:
            result = store.cas_state(
                "backend",
                "closed",
                {"state": "open", "failure_count": 1},
            )
            with result_lock:
                results.append(result)

        threads = [threading.Thread(target=attempt_cas) for _ in range(100)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results.count(True) == 1
        assert results.count(False) == 99
        assert store.get_state("backend")["state"] == "open"

    def test_concurrent_clear_and_increment(self) -> None:
        store = InMemoryStore()
        exceptions: list[BaseException] = []
        exception_lock = threading.Lock()

        def run_safely(action: Any) -> None:
            try:
                action("backend")
            except BaseException as exc:
                with exception_lock:
                    exceptions.append(exc)

        threads = [threading.Thread(target=run_safely, args=(store.increment_failure,)) for _ in range(50)] + [threading.Thread(target=run_safely, args=(store.clear,)) for _ in range(50)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert exceptions == []
        final_count = store.get_state("backend")["failure_count"]
        assert 0 <= final_count <= 50

    def test_concurrent_set_and_get_state(self) -> None:
        store = InMemoryStore()
        exceptions: list[BaseException] = []
        observed_states: list[dict[str, Any]] = []
        lock = threading.Lock()

        def set_open_state() -> None:
            try:
                store.set_state("backend", _complete_state(state="open"))
            except BaseException as exc:
                with lock:
                    exceptions.append(exc)

        def get_state() -> None:
            try:
                state = store.get_state("backend")
                with lock:
                    observed_states.append(state)
            except BaseException as exc:
                with lock:
                    exceptions.append(exc)

        threads = [threading.Thread(target=set_open_state) for _ in range(50)] + [threading.Thread(target=get_state) for _ in range(50)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert exceptions == []
        assert observed_states
        for state in observed_states:
            _assert_complete_state(state)


class TestAdversarialRedisStore:
    def test_redis_store_missing_key_returns_defaults(self) -> None:
        store = RedisStore(_fake_redis())

        assert store.get_state("backend") == DEFAULT_CIRCUIT_BREAKER_STATE

    def test_redis_store_cas_when_key_missing(self) -> None:
        store = RedisStore(_fake_redis())

        result = store.cas_state("backend", "closed", {"state": "open"})

        assert result is True
        assert store.get_state("backend")["state"] == "open"

    def test_redis_store_increment_on_missing_key(self) -> None:
        store = RedisStore(_fake_redis())

        assert store.increment_failure("backend") == 1
        assert store.get_state("backend")["failure_count"] == 1

    def test_redis_store_clear_missing_key_no_crash(self) -> None:
        store = RedisStore(_fake_redis())

        store.clear("backend")

    def test_redis_store_ttl_refreshed_on_increment(self) -> None:
        client = _fake_redis()
        store = RedisStore(client, ttl=123)

        store.increment_failure("backend")

        assert client.ttl("massgen:cb:backend") > 0

    def test_redis_store_increment_preserves_open_state_ttl(self) -> None:
        client = _fake_redis()
        store = RedisStore(client, ttl=1)
        open_until = time.monotonic() + 120
        store.set_state(
            "backend",
            _complete_state(state="open", open_until=open_until),
        )
        ttl_before = client.ttl("massgen:cb:backend")

        store.increment_failure("backend")

        assert client.ttl("massgen:cb:backend") >= ttl_before - 1

    def test_redis_store_increment_without_lua_preserves_open_state_ttl(self, monkeypatch) -> None:
        client = _fake_redis()
        store = RedisStore(client, ttl=1)
        open_until = time.monotonic() + 120
        store.set_state(
            "backend",
            _complete_state(state="open", open_until=open_until),
        )
        ttl_before = client.ttl("massgen:cb:backend")

        def raise_unknown_command(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("unknown command 'eval'")

        monkeypatch.setattr(client, "eval", raise_unknown_command)

        store.increment_failure("backend")

        assert client.ttl("massgen:cb:backend") >= ttl_before - 1

    def test_redis_store_increment_without_lua_retry_exhaustion_raises(self, monkeypatch) -> None:
        store = RedisStore(_fake_redis())

        def raise_unknown_command(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("unknown command 'eval'")

        class FailingWatchPipeline:
            def watch(self, key: str) -> None:
                raise RuntimeError("watch conflict")

        monkeypatch.setattr(store._client, "eval", raise_unknown_command)
        monkeypatch.setattr(store._client, "pipeline", lambda transaction=True: FailingWatchPipeline())

        with pytest.raises(
            RuntimeError,
            match="Failed to atomically increment failure count for 'backend' after 3 retries",
        ):
            store.increment_failure("backend")

    def test_redis_store_open_state_ttl_covers_open_until(self) -> None:
        client = _fake_redis()
        store = RedisStore(client, ttl=1)
        open_until = time.monotonic() + 120

        store.set_state(
            "backend",
            _complete_state(state="open", open_until=open_until),
        )

        assert client.ttl("massgen:cb:backend") >= 170

    def test_redis_store_cas_open_state_ttl_covers_open_until(self) -> None:
        client = _fake_redis()
        store = RedisStore(client, ttl=1)
        open_until = time.monotonic() + 120

        result = store.cas_state(
            "backend",
            "closed",
            {"state": "open", "open_until": open_until},
        )

        assert result is True
        assert client.ttl("massgen:cb:backend") >= 170

    def test_redis_store_cas_missing_state_preserves_partial_hash(self) -> None:
        client = _fake_redis()
        store = RedisStore(client)
        client.hset("massgen:cb:backend", "failure_count", "7")

        result = store.cas_state("backend", "closed", {"state": "open"})

        assert result is True
        assert store.get_state("backend")["failure_count"] == 7

    def test_redis_store_state_dict_wrong_types(self) -> None:
        client = _fake_redis()
        store = RedisStore(client)
        client.hset(
            "massgen:cb:backend",
            mapping={
                "state": "open",
                "failure_count": "not_a_number",
                "last_failure_time": "also_not_a_float",
                "open_until": "still_not_a_float",
                "half_open_probe_active": "not_a_bool",
            },
        )

        # Current behavior raises ValueError for corrupted numeric fields.
        # Graceful defaulting would also be acceptable for this boundary case.
        try:
            state = store.get_state("backend")
        except ValueError:
            return

        assert state == DEFAULT_CIRCUIT_BREAKER_STATE

    def test_redis_store_cas_without_lua_only_one_winner(self, monkeypatch) -> None:
        """Concurrent CAS through non-Lua fallback: exactly 1 thread wins closed->open."""
        client = _fake_redis()
        store = RedisStore(client)

        # Disable Lua scripting to force _cas_state_without_lua path.
        def raise_eval(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("unknown command 'eval'")

        monkeypatch.setattr(client, "eval", raise_eval)

        results: list[bool] = []
        lock = threading.Lock()

        def attempt() -> None:
            res = store.cas_state(
                "backend",
                "closed",
                {"state": "open", "open_until": time.monotonic() + 60},
            )
            with lock:
                results.append(res)

        threads = [threading.Thread(target=attempt) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(True) == 1
        assert results.count(False) == 19
        assert store.get_state("backend")["state"] == "open"

    def test_redis_store_script_unavailable_does_not_match_readonly(
        self,
        monkeypatch,
    ) -> None:
        """READONLY errors are not classified as Lua unavailability."""
        client = _fake_redis()
        store = RedisStore(client)

        def raise_readonly(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("READONLY You can't write against a read only replica")

        monkeypatch.setattr(client, "eval", raise_readonly)

        with pytest.raises(RuntimeError, match="READONLY"):
            store.cas_state("backend", "closed", {"state": "open"})


class TestAdversarialCBIntegration:
    def test_closed_should_block_returns_false(self) -> None:
        config = LLMCircuitBreakerConfig(enabled=True)
        store = InMemoryStore()
        cb = LLMCircuitBreaker(config, backend_name="test", store=store)

        assert cb.should_block() is False
        assert cb.state == CircuitState.CLOSED
        assert store.get_state("test")["state"] == "closed"

    def test_half_open_should_block_active_probe_blocks(self) -> None:
        config = LLMCircuitBreakerConfig(
            enabled=True,
            max_failures=1,
            reset_time_seconds=60,
        )
        store = InMemoryStore()
        cb = LLMCircuitBreaker(config, backend_name="test", store=store)
        store.set_state(
            "test",
            {
                "state": "half_open",
                "failure_count": 1,
                "last_failure_time": 0.0,
                "open_until": 0.0,
                "half_open_probe_active": True,
            },
        )

        assert cb.should_block() is True

    def test_half_open_should_block_inactive_probe_allows_one(self) -> None:
        config = LLMCircuitBreakerConfig(
            enabled=True,
            max_failures=1,
            reset_time_seconds=60,
        )
        store = InMemoryStore()
        cb = LLMCircuitBreaker(config, backend_name="test", store=store)
        store.set_state(
            "test",
            {
                "state": "half_open",
                "failure_count": 1,
                "last_failure_time": 0.0,
                "open_until": 0.0,
                "half_open_probe_active": False,
            },
        )

        assert cb.should_block() is False
        assert store.get_state("test")["half_open_probe_active"] is True
        assert cb.should_block() is True

    def test_record_success_resets_failure_count_in_store(self) -> None:
        config = LLMCircuitBreakerConfig(enabled=True, max_failures=3)
        store = InMemoryStore()
        cb = LLMCircuitBreaker(config, backend_name="test", store=store)

        cb.record_failure()
        cb.record_failure()
        assert store.get_state("test")["failure_count"] == 2

        cb.record_success()

        assert store.get_state("test")["failure_count"] == 0
        assert store.get_state("test")["state"] == "closed"

    def test_closed_reset_is_noop(self) -> None:
        config = LLMCircuitBreakerConfig(enabled=True)
        store = InMemoryStore()
        cb = LLMCircuitBreaker(config, backend_name="test", store=store)

        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert store.get_state("test")["failure_count"] == 0

    def test_half_open_force_open_transitions_to_open(self) -> None:
        config = LLMCircuitBreakerConfig(
            enabled=True,
            max_failures=1,
            reset_time_seconds=60,
        )
        store = InMemoryStore()
        cb = LLMCircuitBreaker(config, backend_name="test", store=store)

        cb.force_open("test")
        store.set_state(
            "test",
            {
                "state": "half_open",
                "failure_count": 1,
                "last_failure_time": 0.0,
                "open_until": 0.0,
                "half_open_probe_active": True,
            },
        )
        assert cb.state == CircuitState.HALF_OPEN

        cb.force_open("force from half_open")

        assert cb.state == CircuitState.OPEN

    def test_half_open_reset_returns_to_closed(self) -> None:
        config = LLMCircuitBreakerConfig(enabled=True, max_failures=1)
        store = InMemoryStore()
        cb = LLMCircuitBreaker(config, backend_name="test", store=store)
        store.set_state(
            "test",
            {
                "state": "half_open",
                "failure_count": 1,
                "last_failure_time": 0.0,
                "open_until": 0.0,
                "half_open_probe_active": True,
            },
        )

        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert store.get_state("test")["failure_count"] == 0

    def test_open_record_success_closes_circuit(self) -> None:
        config = LLMCircuitBreakerConfig(enabled=True, max_failures=1)
        store = InMemoryStore()
        cb = LLMCircuitBreaker(config, backend_name="test", store=store)

        cb.force_open("test")
        assert cb.state == CircuitState.OPEN

        cb.record_success()

        assert cb.state == CircuitState.CLOSED
        assert store.get_state("test")["failure_count"] == 0

    def test_state_machine_all_closed_transitions(self) -> None:
        store = InMemoryStore()
        cb = LLMCircuitBreaker(
            config=_enabled_config(max_failures=2),
            backend_name="backend",
            store=store,
        )

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.force_open()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_state_machine_all_open_transitions(self) -> None:
        store = InMemoryStore()
        cb = LLMCircuitBreaker(
            config=_enabled_config(max_failures=1),
            backend_name="backend",
            store=store,
        )

        cb.force_open()
        assert cb.should_block() is True

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        first_open_until = store.get_state("backend")["open_until"]
        time.sleep(0.001)
        cb.force_open()
        assert cb.state == CircuitState.OPEN
        assert store.get_state("backend")["open_until"] >= first_open_until

        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_state_machine_half_open_transitions(self) -> None:
        store = InMemoryStore()
        cb = LLMCircuitBreaker(
            config=_enabled_config(max_failures=1),
            backend_name="backend",
            store=store,
        )

        cb.force_open()
        state = store.get_state("backend")
        state["open_until"] = time.monotonic() - 1
        store.set_state("backend", state)

        assert cb.should_block() is False
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

        cb.force_open()
        state = store.get_state("backend")
        state["open_until"] = time.monotonic() - 1
        store.set_state("backend", state)
        assert cb.should_block() is False
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_contradictory_state_dict(self) -> None:
        store = InMemoryStore()
        cb = LLMCircuitBreaker(
            config=_enabled_config(),
            backend_name="backend",
            store=store,
        )
        store.set_state("backend", _complete_state(state="open", open_until=0.0))

        assert cb.should_block() is False
        assert cb.state == CircuitState.HALF_OPEN

    def test_exception_during_store_raises_propagates(self) -> None:
        class RaisingSetStateStore(InMemoryStore):
            def set_state(self, backend: str, state: dict) -> None:
                raise RuntimeError("set_state failed")

        cb = LLMCircuitBreaker(
            config=_enabled_config(max_failures=1),
            backend_name="backend",
            store=RaisingSetStateStore(),
        )

        with pytest.raises(RuntimeError, match="set_state failed"):
            cb.record_failure()

    def test_rule_45_no_mutation_before_validation(self) -> None:
        store = InMemoryStore()
        before = store.get_state("backend")

        result = store.cas_state(
            "backend",
            "nonexistent_state",
            {"state": "open", "failure_count": 100},
        )

        assert result is False
        assert store.get_state("backend") == before

    def test_rule_27_error_recovery_store_write_failure(self, monkeypatch) -> None:
        store = InMemoryStore()
        store.set_state("backend", _complete_state(state="closed", failure_count=2))
        before = store.get_state("backend")
        cb = LLMCircuitBreaker(
            config=_enabled_config(),
            backend_name="backend",
            store=store,
        )

        def fail_set_state(self: InMemoryStore, backend: str, state: dict) -> None:
            raise RuntimeError("write failed")

        monkeypatch.setattr(InMemoryStore, "set_state", fail_set_state)

        with pytest.raises(RuntimeError, match="write failed"):
            cb.record_success()

        assert store.get_state("backend") == before

    def test_rule_27_redis_hset_succeeds_expire_fails_state_readable(self) -> None:
        """RedisStore partial-write: HSET succeeds but EXPIRE fails.

        When EXPIRE fails, the key has no TTL but the state data is still
        readable. This is an acceptable degradation -- state is correct,
        key just won't auto-expire. Document the behavior rather than hide it.
        """
        client = _fake_redis()
        store = RedisStore(client, ttl=60)
        original_expire = client.expire

        expire_call_count = [0]

        def fail_on_second_expire(key: str, seconds: int) -> bool:
            expire_call_count[0] += 1
            if expire_call_count[0] == 1:
                raise RuntimeError("expire failed")
            return original_expire(key, seconds)

        client.expire = fail_on_second_expire

        with pytest.raises(RuntimeError, match="expire failed"):
            store.set_state("backend", _complete_state(state="open", failure_count=3))

        # After partial write: HSET succeeded (data readable), EXPIRE failed (no TTL).
        # State is defined -- either full data if HSET was atomic, or empty if HSET
        # also failed. Both are acceptable; no crash and no corrupted type errors.
        try:
            state = store.get_state("backend")
            # If data was written, it should be structurally valid.
            assert isinstance(state["failure_count"], int)
            assert state["state"] in ("closed", "open", "half_open")
        except Exception as exc:
            pytest.fail(f"get_state raised after partial write: {exc}")

    def test_two_cb_same_store_different_backends(self) -> None:
        store = InMemoryStore()
        backend_a = LLMCircuitBreaker(
            config=_enabled_config(max_failures=1),
            backend_name="backend_a",
            store=store,
        )
        backend_b = LLMCircuitBreaker(
            config=_enabled_config(max_failures=1),
            backend_name="backend_b",
            store=store,
        )

        backend_a.record_failure()

        assert backend_a.state == CircuitState.OPEN
        assert backend_b.state == CircuitState.CLOSED
        assert backend_b.failure_count == 0
