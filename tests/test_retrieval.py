from __future__ import annotations

from pathlib import Path

import pytest

from finagent.indexing.vector_index import VectorIndex
from finagent.llm.fake import FakeClient
from finagent.retrieval.bm25 import BM25, tokenize
from finagent.retrieval.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from finagent.schemas import Chunk


def chunks_from(texts: list[str]) -> list[Chunk]:
    return [Chunk(chunk_id=f"d:p{i}:c0", doc_id="d", page=i, text=t) for i, t in enumerate(texts)]


def test_tokenize_keeps_numbers_and_drops_stopwords() -> None:
    assert tokenize("The total of $1,577.5 for FY2018") == ["total", "1,577.5", "fy2018"]


def test_bm25_ranks_matching_document_first() -> None:
    bm = BM25(["cash flow statement purchases of property", "risk factors competition", "goodwill"])
    scores = bm.scores("purchases property")
    assert scores.index(max(scores)) == 0
    assert scores[1] == scores[2] == 0.0


def test_bm25_handles_empty_corpus_and_unknown_terms() -> None:
    assert BM25([]).scores("anything") == []
    assert BM25(["abc"]).scores("zzz") == [0.0]


def test_bm25_retriever_drops_zero_scores() -> None:
    r = BM25Retriever(chunks_from(["alpha beta", "gamma"]))
    hits = r.retrieve("alpha", 5)
    assert [h.chunk.page for h in hits] == [0]


def test_vector_index_search_and_roundtrip(tmp_path: Path) -> None:
    chunks = chunks_from(["a", "b", "c"])
    index = VectorIndex(chunks, [[1, 0], [0, 1], [0.9, 0.1]])
    assert [i for i, _ in index.search([1, 0], 2)] == [0, 2]
    index.save(tmp_path / "idx")
    loaded = VectorIndex.load(tmp_path / "idx")
    assert loaded.chunks == chunks
    assert [i for i, _ in loaded.search([0, 1], 1)] == [1]


def test_vector_index_validates_shape() -> None:
    with pytest.raises(ValueError, match="one 2-D vector row per chunk"):
        VectorIndex(chunks_from(["a", "b"]), [[1.0, 0.0]])


def test_vector_index_empty_search() -> None:
    assert VectorIndex([], []).search([1.0], 3) == []


def test_dense_retriever_finds_identical_text() -> None:
    client = FakeClient()
    chunks = chunks_from(["alpha", "beta", "gamma"])
    index = VectorIndex(chunks, client.embed([c.text for c in chunks]))
    [top] = DenseRetriever(index, client).retrieve("beta", 1)
    assert top.chunk.text == "beta"


class StaticRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, k: int):  # type: ignore[no-untyped-def]
        from finagent.schemas import ScoredChunk

        return [ScoredChunk(chunk=c, score=1.0) for c in self._chunks[:k]]


def test_hybrid_rrf_rewards_agreement_between_retrievers() -> None:
    a, b, c = chunks_from(["a", "b", "c"])
    hybrid = HybridRetriever([StaticRetriever([a, b]), StaticRetriever([b, c])])
    ranked = [h.chunk.text for h in hybrid.retrieve("q", 3)]
    assert ranked[0] == "b"  # appears in both lists, so it beats either list's sole #1
    assert set(ranked) == {"a", "b", "c"}
