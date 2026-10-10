"""Gemini (Google AI Studio) implementation of :class:`LLMClient`.

Adds disk caching, a rate limiter, and retry-with-backoff around the ``google-genai`` SDK.
The SDK client can be injected, which keeps this class testable without network access.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

from pydantic import BaseModel

from finagent.config import Settings
from finagent.llm.base import Image, QuotaExhaustedError
from finagent.llm.cache import DiskCache, make_key
from finagent.llm.resilience import RateLimiter, retry_with_backoff

logger = logging.getLogger(__name__)

R = TypeVar("R")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_EMBED_BATCH_SIZE = 100  # Gemini embedding endpoint batch limit


_RETRY_DELAY_RE = re.compile(r"retryDelay\W+(\d+(?:\.\d+)?)s")
_LONG_DELAY_S = 600.0  # a server-suggested wait beyond this is not worth retrying in-process


def _retry_delay_s(exc: BaseException) -> float | None:
    match = _RETRY_DELAY_RE.search(str(exc))
    return float(match.group(1)) if match else None


def is_daily_quota_exhausted(exc: BaseException) -> bool:
    """True for 429s caused by a per-day quota (or any wait the server says is very long)."""
    if getattr(exc, "code", None) != 429:
        return False
    if "PerDay" in str(exc):
        return True
    delay = _retry_delay_s(exc)
    return delay is not None and delay > _LONG_DELAY_S


def is_retryable(exc: BaseException) -> bool:
    """True for transient API errors (per-minute rate limits and server-side failures)."""
    return getattr(exc, "code", None) in _RETRYABLE_STATUS and not is_daily_quota_exhausted(exc)


class GeminiClient:
    """LLM client backed by the Gemini API."""

    def __init__(
        self,
        settings: Settings,
        *,
        sdk_client: Any | None = None,
        cache: DiskCache | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._settings = settings
        self._sdk = sdk_client if sdk_client is not None else self._build_sdk_client(settings)
        self._cache = cache if cache is not None else DiskCache(settings.cache_dir)
        self._limiter = limiter or RateLimiter(settings.requests_per_minute)

    @staticmethod
    def _build_sdk_client(settings: Settings) -> Any:
        if settings.google_api_key is None:
            raise RuntimeError(
                "No API key found. Set FINAGENT_GOOGLE_API_KEY (get one at aistudio.google.com)."
            )
        from google import genai  # imported lazily so tests/fake runs need no SDK setup

        return genai.Client(api_key=settings.google_api_key.get_secret_value())

    # ------------------------------------------------------------------ generation
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: list[Image] | None = None,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        key = make_key(
            kind="generate",
            model=self._settings.llm_model,
            prompt=prompt,
            system=system,
            images=[(i.mime_type, i.data) for i in images or []],
            schema=schema.model_json_schema() if schema else None,
            temperature=temperature,
        )
        cached = self._cache.get(key)
        if isinstance(cached, str):
            return cached

        text = self._call_with_resilience(
            lambda: self._generate_once(prompt, system, images, schema, temperature)
        )
        self._cache.set(key, text)
        return text

    def _generate_once(
        self,
        prompt: str,
        system: str | None,
        images: list[Image] | None,
        schema: type[BaseModel] | None,
        temperature: float,
    ) -> str:
        from google.genai import types

        contents: list[Any] = [
            types.Part.from_bytes(data=img.data, mime_type=img.mime_type) for img in images or []
        ]
        contents.append(prompt)
        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if system:
            config_kwargs["system_instruction"] = system
        if schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema
        response = self._sdk.models.generate_content(
            model=self._settings.llm_model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = response.text
        if not isinstance(text, str):
            raise RuntimeError("Gemini returned no text (response may have been blocked).")
        return text

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, caching per text and batching only the cache misses."""
        keys = [make_key(kind="embed", model=self._settings.embed_model, text=t) for t in texts]
        vectors: dict[int, list[float]] = {}
        missing: list[int] = []
        for i, key in enumerate(keys):
            cached = self._cache.get(key)
            if isinstance(cached, list):
                vectors[i] = cached
            else:
                missing.append(i)

        for start in range(0, len(missing), _EMBED_BATCH_SIZE):
            batch_idx = missing[start : start + _EMBED_BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_idx]
            result = self._call_with_resilience(partial(self._embed_once, batch_texts))
            for i, vec in zip(batch_idx, result, strict=True):
                vectors[i] = vec
                self._cache.set(keys[i], vec)
        return [vectors[i] for i in range(len(texts))]

    def _embed_once(self, texts: list[str]) -> list[list[float]]:
        response = self._sdk.models.embed_content(model=self._settings.embed_model, contents=texts)
        return [list(e.values) for e in response.embeddings]

    # ------------------------------------------------------------------ plumbing
    def _call_with_resilience(self, fn: Callable[[], R]) -> R:
        def limited() -> R:
            self._limiter.acquire()
            return fn()

        try:
            return retry_with_backoff(
                limited,
                is_retryable=is_retryable,
                max_retries=self._settings.max_retries,
                base_delay_s=self._settings.retry_base_delay_s,
            )
        except Exception as exc:
            if is_daily_quota_exhausted(exc):
                delay = _retry_delay_s(exc)
                resets = f" (resets in about {delay / 3600:.1f} h)" if delay else ""
                raise QuotaExhaustedError(
                    f"Daily API quota exhausted for {self._settings.llm_model}{resets}. "
                    "Quotas are per model: wait, enable billing, or switch models via "
                    "FINAGENT_LLM_MODEL / --judge-model."
                ) from exc
            raise
