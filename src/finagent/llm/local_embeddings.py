"""Local sentence-transformer embeddings; this module makes no network API calls.

The model is downloaded by Hugging Face on first use if it is not already cached.
Subsequent embedding inference runs locally.
"""

from __future__ import annotations

from typing import Any


class SentenceTransformerClient:
    """Embedding-only client backed by a local Sentence Transformers model."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require sentence-transformers. Install the project with "
                "`python -m pip install -e '.[local-embeddings]'`."
            ) from exc
        self.model_name = model_name
        self._model: Any = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode texts locally and return normalized vectors in input order."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result: list[list[float]] = vectors.tolist()
        return result
