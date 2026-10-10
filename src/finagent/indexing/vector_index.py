"""A small exact-search vector index (numpy). Per-document indexes are tiny, so no ANN needed."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from finagent.schemas import Chunk

Matrix = npt.NDArray[np.float32]


class VectorIndex:
    """Cosine-similarity index over a list of chunks."""

    def __init__(self, chunks: list[Chunk], vectors: Sequence[Sequence[float]] | Matrix) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if not chunks and matrix.size == 0:  # e.g. an image-only PDF with no extractable text
            matrix = np.zeros((0, 0), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise ValueError("need one 2-D vector row per chunk")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._matrix: Matrix = (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32)
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: Sequence[float], k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` (chunk position, cosine score) pairs, best first."""
        if not self.chunks:
            return []
        q = np.asarray(query, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        scores = self._matrix @ (q / (norm or 1.0))
        top = np.argsort(-scores)[: min(k, len(self.chunks))]
        return [(int(i), float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._matrix)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")

    @classmethod
    def load(cls, directory: Path) -> VectorIndex:
        with (directory / "chunks.jsonl").open(encoding="utf-8") as fh:
            chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]
        return cls(chunks, np.load(directory / "vectors.npy"))
