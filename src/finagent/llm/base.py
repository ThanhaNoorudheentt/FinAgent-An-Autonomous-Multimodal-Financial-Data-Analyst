"""The LLM interface. The rest of the code base depends on this, never on a vendor SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class QuotaExhaustedError(RuntimeError):
    """A provider quota that will not recover within minutes (e.g. a daily limit) is used up.

    Callers should stop and resume later rather than retry: retrying only wastes time.
    """


@dataclass(frozen=True)
class Image:
    """Raw image bytes (a PDF page render, a chart crop, ...)."""

    data: bytes
    mime_type: str = "image/png"


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface needed by FinAgent. Implementations: GeminiClient, FakeClient."""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: list[Image] | None = None,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return the model's text. If ``schema`` is given, the text is JSON for that schema."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""
        ...


def generate_model(
    client: LLMClient,
    prompt: str,
    schema: type[T],
    *,
    system: str | None = None,
    images: list[Image] | None = None,
) -> T:
    """Generate and validate a structured response as a pydantic model."""
    raw = client.generate(prompt, system=system, images=images, schema=schema)
    return schema.model_validate_json(_strip_code_fence(raw))


def _strip_code_fence(text: str) -> str:
    """Remove a leading/trailing markdown code fence if the model added one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()
