"""
Circuit breaker for AI provider requests.

Protects AERIS from long blocking timeouts when the remote AI service
is experiencing sustained outages, rate limits, or connectivity failures.

States:
  CLOSED    -> Normal operation. Requests flow through.
  OPEN      -> Threshold of consecutive failures reached. Fast-fail immediately
               without attempting network I/O during the cooldown window.
  HALF_OPEN -> Cooldown elapsed. Allows a single probe request to check if the
               service has recovered. If successful, transitions to CLOSED;
               if failed, returns to OPEN.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict

logger = logging.getLogger("aeris.ai.circuit_breaker")


class CircuitBreaker:
    """Thread-safe circuit breaker implementation."""

    STATE_CLOSED = "CLOSED"
    STATE_OPEN = "OPEN"
    STATE_HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(0.1, float(cooldown_seconds))

        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Return the current circuit state, computing potential OPEN -> HALF_OPEN transitions."""
        with self._lock:
            if self._state == self.STATE_OPEN:
                if (time.monotonic() - self._last_failure_time) >= self.cooldown_seconds:
                    self._state = self.STATE_HALF_OPEN
                    logger.info("Circuit breaker cooldown elapsed -> transition to HALF_OPEN")
            return self._state

    @property
    def failure_count(self) -> int:
        """Return the current consecutive failure count."""
        with self._lock:
            return self._failure_count

    def can_execute(self) -> bool:
        """Return True if a request is allowed to proceed, False if it should fast-fail."""
        current_state = self.state  # evaluates OPEN -> HALF_OPEN transition under lock
        return current_state in (self.STATE_CLOSED, self.STATE_HALF_OPEN)

    def record_success(self) -> None:
        """Record a successful provider request, resetting failures and closing the circuit."""
        with self._lock:
            if self._state != self.STATE_CLOSED or self._failure_count > 0:
                logger.info(
                    "Circuit breaker recovered: transitioning from %s to CLOSED (was %d failures)",
                    self._state,
                    self._failure_count,
                )
            self._state = self.STATE_CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a provider failure, tripping to OPEN if failure threshold is reached."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == self.STATE_HALF_OPEN:
                logger.warning(
                    "Probe request failed in HALF_OPEN state -> circuit re-opened for %.1fs",
                    self.cooldown_seconds,
                )
                self._state = self.STATE_OPEN
            elif self._failure_count >= self.failure_threshold:
                if self._state != self.STATE_OPEN:
                    logger.warning(
                        "Circuit breaker tripped to OPEN after %d consecutive failures (cooldown=%.1fs)",
                        self._failure_count,
                        self.cooldown_seconds,
                    )
                self._state = self.STATE_OPEN

    def reset(self) -> None:
        """Force reset the circuit breaker to clean CLOSED state."""
        with self._lock:
            self._state = self.STATE_CLOSED
            self._failure_count = 0
            self._last_failure_time = 0.0

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic dictionary without secrets."""
        # Read via public accessors so we never nest the non-reentrant lock.
        return {
            "state": self.state,
            "consecutive_failures": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
        }
