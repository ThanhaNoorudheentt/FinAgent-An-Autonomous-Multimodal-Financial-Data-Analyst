"""Deterministic in-memory client for tests and offline development."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable

from pydantic import BaseModel

from finagent.llm.base import Image

Responder = Callable[[str], str]


def _constant_responder(text: str) -> Responder:
    def respond(_prompt: str) -> str:
        return text

    return respond


class FakeClient:
    """Implements :class:`LLMClient` without any network access.

    ``responder`` maps a prompt to a response string; every call is recorded in ``calls``.
    Embeddings are deterministic, unit-norm vectors derived from a hash of the text.
    """

    def __init__(self, responder: Responder | str = "", *, embed_dim: int = 8) -> None:
        self._responder: Responder = (
            responder if callable(responder) else _constant_responder(responder)
        )
        self._embed_dim = embed_dim
        self.calls: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: list[Image] | None = None,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(prompt)
        return self._responder(prompt)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [(digest[i % len(digest)] / 255.0) - 0.5 for i in range(self._embed_dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]
