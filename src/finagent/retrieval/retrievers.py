"""Retrievers share one interface so RAG systems can swap them freely."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from finagent.indexing.vector_index import VectorIndex
from finagent.llm.base import LLMClient
from finagent.retrieval.bm25 import BM25
from finagent.schemas import Chunk, ScoredChunk


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[ScoredChunk]: ...


class DenseRetriever:
    """Embedding similarity search."""

    def __init__(self, index: VectorIndex, client: LLMClient) -> None:
        self._index = index
        self._client = client

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        [vector] = self._client.embed([query])
        return [
            ScoredChunk(chunk=self._index.chunks[i], score=s)
            for i, s in self._index.search(vector, k)
        ]


class BM25Retriever:
    """Lexical search; needs no embeddings."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._bm25 = BM25([c.text for c in chunks])

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        scores = self._bm25.scores(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [ScoredChunk(chunk=self._chunks[i], score=scores[i]) for i in order if scores[i] > 0]


class HybridRetriever:
    """Reciprocal Rank Fusion of several retrievers (robust, needs no score calibration)."""

    def __init__(
        self, retrievers: Sequence[Retriever], *, rrf_k: int = 60, candidates: int = 30
    ) -> None:
        self._retrievers = retrievers
        self._rrf_k = rrf_k
        self._candidates = candidates

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for retriever in self._retrievers:
            for rank, hit in enumerate(retriever.retrieve(query, max(k, self._candidates))):
                cid = hit.chunk.chunk_id
                by_id[cid] = hit.chunk
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self._rrf_k + rank + 1)
        ranked = sorted(fused, key=lambda c: fused[c], reverse=True)[:k]
        return [ScoredChunk(chunk=by_id[c], score=fused[c]) for c in ranked]
