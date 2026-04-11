"""Optional Prometheus metrics for circuit breaker behavior."""

from __future__ import annotations

from threading import RLock
from typing import Any


class CircuitBreakerMetrics:
    """Prometheus metrics for circuit breaker state and request outcomes.

    The class is safe to use from multiple threads. Metric initialization and
    updates are protected by an instance-level re-entrant lock. If
    ``prometheus_client`` is not installed, all recording methods become
    no-ops and :meth:`get_registry` returns ``None``.
    """

    _STATE_VALUES = {
        "CLOSED": 0,
        "HALF_OPEN": 1,
        "OPEN": 2,
    }
    # Buckets cover sub-second fast paths through long-running LLM streaming calls.
    # Values above 600s fall into the +Inf bucket.
    _LATENCY_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600)

    def __init__(self) -> None:
        self._available: bool | None = None
        self._lock = RLock()
        self._registry: Any = None
        self._state_transitions: Any = None
        self._requests: Any = None
        self._request_latency: Any = None
        self._current_state: Any = None

    def record_state_transition(
        self,
        backend: str,
        from_state: str,
        to_state: str,
    ) -> None:
        """Record a circuit breaker state transition.

        Args:
            backend: Stable, low-cardinality backend identifier (e.g. "claude",
                "gemini", "openai"). Avoid dynamic values such as URLs, session IDs,
                or tenant names -- each distinct value creates a new time series.
            from_state: Previous circuit state (e.g. "closed", "open", "half_open").
            to_state: New circuit state.
        """
        if not self._ensure_metrics():
            return

        with self._lock:
            self._state_transitions.labels(
                backend=backend,
                from_state=from_state,
                to_state=to_state,
            ).inc()
            self._current_state.labels(backend=backend).set(
                self._state_value(to_state)
            )

    def record_request(
        self,
        backend: str,
        outcome: str,
        latency_seconds: float,
    ) -> None:
        """Record a circuit breaker request outcome and latency.

        Args:
            backend: Stable, low-cardinality backend identifier. See
                ``record_state_transition`` for cardinality guidance.
            outcome: One of "success", "failure", "rejected_open",
                "rejected_half_open". Custom values are accepted but kept
                low-cardinality.
            latency_seconds: Wall-clock duration of one API attempt in seconds
                (per-attempt, not total including retries or retry sleeps).
                Negative values (e.g. from clock skew) are passed to the
                histogram as-is.
        """
        if not self._ensure_metrics():
            return

        with self._lock:
            self._requests.labels(backend=backend, outcome=outcome).inc()
            self._request_latency.labels(backend=backend).observe(
                latency_seconds
            )

    def get_registry(self) -> Any:
        """Return this instance's custom Prometheus registry, if available."""
        if not self._ensure_metrics():
            return None

        return self._registry

    def _ensure_metrics(self) -> bool:
        """Initialize Prometheus metrics lazily."""
        if self._available is False:
            return False
        if self._available is True:
            return True

        with self._lock:
            if self._available is not None:
                return self._available

            try:
                from prometheus_client import (  # type: ignore[import-not-found]
                    CollectorRegistry,
                    Counter,
                    Gauge,
                    Histogram,
                )
            except ImportError:
                self._available = False
                return False

            self._registry = CollectorRegistry()
            self._state_transitions = Counter(
                "cb_state_transitions_total",
                "Circuit breaker state transitions.",
                ["backend", "from_state", "to_state"],
                registry=self._registry,
            )
            self._requests = Counter(
                "cb_requests_total",
                "Circuit breaker requests by outcome.",
                ["backend", "outcome"],
                registry=self._registry,
            )
            self._request_latency = Histogram(
                "cb_request_latency_seconds",
                "Circuit breaker request latency in seconds.",
                ["backend"],
                buckets=self._LATENCY_BUCKETS,
                registry=self._registry,
            )
            self._current_state = Gauge(
                "cb_current_state",
                "Circuit breaker current state: 0=CLOSED, 1=HALF_OPEN, 2=OPEN.",
                ["backend"],
                registry=self._registry,
            )
            self._available = True
            return True

    def _state_value(self, state: str) -> int:
        """Return the numeric gauge value for a circuit breaker state."""
        return self._STATE_VALUES.get(state.upper(), -1)
