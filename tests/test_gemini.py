from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from finagent.config import Settings
from finagent.llm.base import Image, generate_model
from finagent.llm.cache import DiskCache
from finagent.llm.gemini import GeminiClient, is_retryable
from finagent.llm.resilience import RateLimiter
from finagent.schemas import Answer


class ApiError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


class StubModels:
    def __init__(self, text: str = "hello", fail_codes: list[int] | None = None) -> None:
        self.text = text
        self.fail_codes = list(fail_codes or [])
        self.generate_calls = 0
        self.embed_calls: list[list[str]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.generate_calls += 1
        if self.fail_codes:
            raise ApiError(self.fail_codes.pop(0))
        return SimpleNamespace(text=self.text)

    def embed_content(self, *, model: str, contents: list[str]) -> Any:
        self.embed_calls.append(contents)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[float(len(t)), 1.0]) for t in contents]
        )


def make_client(tmp_path: Path, models: StubModels) -> GeminiClient:
    settings = Settings(cache_dir=tmp_path, retry_base_delay_s=0.0, max_retries=3)
    return GeminiClient(
        settings,
        sdk_client=SimpleNamespace(models=models),
        cache=DiskCache(tmp_path),
        limiter=RateLimiter(1000, sleep=lambda _s: None),
    )


def test_is_retryable_status_codes() -> None:
    assert is_retryable(ApiError(429))
    assert is_retryable(ApiError(503))
    assert not is_retryable(ApiError(400))
    assert not is_retryable(ValueError("x"))


def test_generate_is_cached(tmp_path: Path) -> None:
    models = StubModels(text="42")
    client = make_client(tmp_path, models)
    assert client.generate("q") == "42"
    assert client.generate("q") == "42"
    assert models.generate_calls == 1
    client.generate("different")
    assert models.generate_calls == 2


def test_images_are_part_of_cache_key(tmp_path: Path) -> None:
    models = StubModels()
    client = make_client(tmp_path, models)
    client.generate("q", images=[Image(b"one")])
    client.generate("q", images=[Image(b"two")])
    assert models.generate_calls == 2


def test_transient_errors_are_retried(tmp_path: Path) -> None:
    models = StubModels(text="ok", fail_codes=[429, 503])
    client = make_client(tmp_path, models)
    assert client.generate("q") == "ok"
    assert models.generate_calls == 3


def test_permanent_errors_are_not_retried(tmp_path: Path) -> None:
    models = StubModels(fail_codes=[400])
    client = make_client(tmp_path, models)
    with pytest.raises(ApiError):
        client.generate("q")
    assert models.generate_calls == 1


def test_embed_caches_per_text_and_preserves_order(tmp_path: Path) -> None:
    models = StubModels()
    client = make_client(tmp_path, models)
    first = client.embed(["aa", "b"])
    assert first == [[2.0, 1.0], [1.0, 1.0]]
    second = client.embed(["b", "ccc", "aa"])
    assert second == [[1.0, 1.0], [3.0, 1.0], [2.0, 1.0]]
    assert models.embed_calls == [["aa", "b"], ["ccc"]]  # only the miss was sent


def test_missing_api_key_gives_clear_error() -> None:
    with pytest.raises(RuntimeError, match="FINAGENT_GOOGLE_API_KEY"):
        GeminiClient(Settings(google_api_key=None, _env_file=None))  # type: ignore[call-arg]


def test_generate_model_parses_fenced_json(tmp_path: Path) -> None:
    models = StubModels(text='```json\n{"answer": "94"}\n```')
    client = make_client(tmp_path, models)
    result = generate_model(client, "q", Answer)
    assert result.answer == "94"


# The message below is the real shape of the daily-quota error from the Gemini free tier.
DAILY_QUOTA_MSG = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED', 'details': "
    "[{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'}, "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '66807s'}]}}"
)
PER_MINUTE_MSG = (
    "429 RESOURCE_EXHAUSTED. {'details': [{'quotaId': 'GenerateRequestsPerMinutePerProject', "
    "'@type': 'RetryInfo', 'retryDelay': '12s'}]}"
)


class QuotaApiError(Exception):
    def __init__(self, message: str, code: int = 429) -> None:
        super().__init__(message)
        self.code = code


def test_daily_quota_is_detected_but_per_minute_limit_is_not() -> None:
    from finagent.llm.gemini import is_daily_quota_exhausted

    assert is_daily_quota_exhausted(QuotaApiError(DAILY_QUOTA_MSG))
    assert not is_daily_quota_exhausted(QuotaApiError(PER_MINUTE_MSG))
    assert not is_daily_quota_exhausted(QuotaApiError(DAILY_QUOTA_MSG, code=500))
    # A very long server-suggested wait counts even without the "PerDay" marker.
    assert is_daily_quota_exhausted(QuotaApiError("429 {'retryDelay': '7200s'}"))


def test_per_minute_429_is_still_retried_but_daily_is_not() -> None:
    from finagent.llm.gemini import is_retryable

    assert is_retryable(QuotaApiError(PER_MINUTE_MSG))
    assert not is_retryable(QuotaApiError(DAILY_QUOTA_MSG))


def test_daily_quota_fails_fast_with_clear_error(tmp_path: Path) -> None:
    from finagent.llm.base import QuotaExhaustedError

    class QuotaModels(StubModels):
        def generate_content(self, **kwargs: Any) -> Any:
            self.generate_calls += 1
            raise QuotaApiError(DAILY_QUOTA_MSG)

    models = QuotaModels()
    client = make_client(tmp_path, models)
    with pytest.raises(QuotaExhaustedError, match=r"resets in about 18\.6 h"):
        client.generate("q")
    assert models.generate_calls == 1  # no pointless retries
