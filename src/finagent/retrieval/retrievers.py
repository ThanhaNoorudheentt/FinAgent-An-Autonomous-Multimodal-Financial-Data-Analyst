"""Retrievers share one interface so RAG systems can swap them freely."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from finagent.indexing.vector_index import VectorIndex
from finagent.ingestion.structure import detect_statement_intent
from finagent.llm.base import EmbeddingClient
from finagent.retrieval.bm25 import BM25
from finagent.retrieval.synonyms import expand_query
from finagent.schemas import Chunk, ScoredChunk


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[ScoredChunk]: ...


class DenseRetriever:
    """Embedding similarity search."""

    def __init__(self, index: VectorIndex, client: EmbeddingClient) -> None:
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

    def __init__(self, chunks: list[Chunk], *, expand: bool = False) -> None:
        self._chunks = chunks
        self._bm25 = BM25([c.text for c in chunks])
        self._expand = expand

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        scores = self._bm25.scores(expand_query(query) if self._expand else query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [ScoredChunk(chunk=self._chunks[i], score=scores[i]) for i in order if scores[i] > 0]


def rrf_fuse(
    ranked_lists: Sequence[Sequence[ScoredChunk]], *, k: int, rrf_k: int = 60
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion: robust to differing score scales, rewards agreement."""
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits):
            cid = hit.chunk.chunk_id
            by_id[cid] = hit.chunk
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
    ranked = sorted(fused, key=lambda c: fused[c], reverse=True)[:k]
    return [ScoredChunk(chunk=by_id[c], score=fused[c]) for c in ranked]


class HybridRetriever:
    """Reciprocal Rank Fusion of several retrievers."""

    def __init__(
        self, retrievers: Sequence[Retriever], *, rrf_k: int = 60, candidates: int = 30
    ) -> None:
        self._retrievers = retrievers
        self._rrf_k = rrf_k
        self._candidates = candidates

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        lists = [r.retrieve(query, max(k, self._candidates)) for r in self._retrievers]
        return rrf_fuse(lists, k=k, rrf_k=self._rrf_k)


class StatementAwareRetriever:
    """Put chunks from the statement a question names (balance sheet, ...) first.

    If the question says e.g. "...shown in the cash flow statement", the best-matching chunks
    *within the pages tagged as that statement* are ranked ahead of the base retriever's results.
    Questions that name no statement fall through to the base retriever unchanged.
    """

    def __init__(self, base: Retriever, chunks: list[Chunk], *, per_statement: int = 3) -> None:
        self._base = base
        self._per_statement = per_statement
        self._by_tag: dict[str, BM25Retriever] = {}
        for tag in {t for c in chunks for t in c.tags}:
            self._by_tag[tag] = BM25Retriever([c for c in chunks if tag in c.tags], expand=True)

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        tags = [t for t in detect_statement_intent(query) if t in self._by_tag]
        priority: list[ScoredChunk] = []
        for tag in tags:
            priority.extend(self._by_tag[tag].retrieve(query, self._per_statement))
        seen = {h.chunk.chunk_id for h in priority}
        rest = [
            h for h in self._base.retrieve(query, k + len(priority)) if h.chunk.chunk_id not in seen
        ]
        return (priority + rest)[:k]
