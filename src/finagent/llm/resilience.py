"""Rate limiting and retry-with-backoff, with injectable clock/sleep for fast tests."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import TypeVar

R = TypeVar("R")


class RateLimiter:
    """Sliding-window limiter: at most ``max_calls`` per 60 seconds, thread-safe."""

    def __init__(
        self,
        max_calls_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_calls_per_minute < 1:
            raise ValueError("max_calls_per_minute must be >= 1")
        self._max = max_calls_per_minute
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a call is allowed, then record it."""
        while True:
            with self._lock:
                now = self._clock()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self._max:
                    self._calls.append(now)
                    return
                wait = 60.0 - (now - self._calls[0])
            self._sleep(max(wait, 0.0))


def retry_with_backoff(
    fn: Callable[[], R],
    *,
    is_retryable: Callable[[BaseException], bool],
    max_retries: int,
    base_delay_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> R:
    """Call ``fn``; on a retryable error wait base * 2**attempt seconds and try again."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries or not is_retryable(exc):
                raise
            sleep(base_delay_s * (2**attempt))
            attempt += 1
