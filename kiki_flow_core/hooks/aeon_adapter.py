"""Read-only adapter to micro-kiki Aeon memory with a circuit breaker."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class CircuitBreakerOpenError(Exception):
    """Raised when the breaker is open and the call is rejected."""


class AeonAdapter:
    """Wrap an Aeon-fetching callable with a circuit breaker on consecutive failures."""

    def __init__(
        self,
        fetcher: Callable[[int], list[dict[str, Any]]],
        breaker_threshold: int = 3,
        breaker_cooldown_s: float = 300.0,
    ) -> None:
        self.fetcher = fetcher
        self.breaker_threshold = breaker_threshold
        self.breaker_cooldown_s = breaker_cooldown_s
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def _is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.breaker_cooldown_s:
            self._opened_at = None
            self._consecutive_failures = 0
            return False
        return True

    def fetch_recent_episodes(self, window_h: int) -> list[dict[str, Any]]:
        if self._is_open():
            raise CircuitBreakerOpenError(
                f"Aeon breaker open until cooldown ({self.breaker_cooldown_s}s)"
            )
        try:
            result = self.fetcher(window_h)
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.breaker_threshold:
                self._opened_at = time.monotonic()
            raise
        else:
            self._consecutive_failures = 0
            return result
