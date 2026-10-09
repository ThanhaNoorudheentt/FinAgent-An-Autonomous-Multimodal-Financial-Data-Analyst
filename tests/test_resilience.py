from __future__ import annotations

import pytest

from finagent.llm.resilience import RateLimiter, retry_with_backoff


class FlakyError(Exception):
    pass


def test_retry_succeeds_after_transient_failures() -> None:
    attempts = {"n": 0}
    sleeps: list[float] = []

    def fn() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise FlakyError
        return "ok"

    result = retry_with_backoff(
        fn,
        is_retryable=lambda e: isinstance(e, FlakyError),
        max_retries=5,
        base_delay_s=2.0,
        sleep=sleeps.append,
    )
    assert result == "ok"
    assert sleeps == [2.0, 4.0]  # exponential backoff


def test_retry_gives_up_after_max_retries() -> None:
    def fn() -> None:
        raise FlakyError

    with pytest.raises(FlakyError):
        retry_with_backoff(
            fn,
            is_retryable=lambda _e: True,
            max_retries=2,
            base_delay_s=0.0,
            sleep=lambda _s: None,
        )


def test_non_retryable_error_raises_immediately() -> None:
    sleeps: list[float] = []

    def fn() -> None:
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        retry_with_backoff(
            fn, is_retryable=lambda _e: False, max_retries=5, base_delay_s=1.0, sleep=sleeps.append
        )
    assert sleeps == []


def test_rate_limiter_waits_when_window_full() -> None:
    now = {"t": 0.0}
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now["t"] += s

    limiter = RateLimiter(2, clock=lambda: now["t"], sleep=sleep)
    limiter.acquire()
    limiter.acquire()
    assert slept == []
    limiter.acquire()  # third call must wait for the first to leave the 60s window
    assert slept == [60.0]


def test_rate_limiter_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="max_calls_per_minute"):
        RateLimiter(0)
