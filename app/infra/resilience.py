from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    """Fail-open after N errors; half-open after reset_seconds."""

    def __init__(self, failure_threshold: int = 3, reset_seconds: float = 15.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self._failures = 0
        self._open_until = 0.0

    def _raise_if_open(self) -> None:
        now = time.monotonic()
        if self._open_until and now < self._open_until:
            raise CircuitOpenError("circuit open")
        if self._open_until and now >= self._open_until:
            self._open_until = 0.0
            self._failures = 0

    def record_success(self) -> None:
        self._failures = 0
        self._open_until = 0.0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._open_until = time.monotonic() + self.reset_seconds

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        self._raise_if_open()
        try:
            result = await fn()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


async def with_timeout(awaitable: Awaitable[T], seconds: float) -> T:
    return await asyncio.wait_for(awaitable, timeout=seconds)
