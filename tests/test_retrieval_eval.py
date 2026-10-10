from __future__ import annotations

from finagent.evaluation.retrieval import evaluate_retrieval
from finagent.retrieval.retrievers import Retriever
from finagent.schemas import Chunk, FinanceBenchEvidence, FinanceBenchExample, ScoredChunk


def example(doc: str, pages: list[int]) -> FinanceBenchExample:
    return FinanceBenchExample(
        financebench_id=f"fb_{doc}",
        company="X",
        doc_name=doc,
        question_type="t",
        question="q",
        answer="a",
        evidence=[
            FinanceBenchEvidence(evidence_text="t", doc_name=doc, evidence_page_num=p)
            for p in pages
        ],
    )


class RankedPages:
    """Retriever that returns chunks from the given pages, in order."""

    def __init__(self, doc: str, pages: list[int]) -> None:
        self._chunks = [
            Chunk(chunk_id=f"{doc}:p{p}:c0", doc_id=doc, page=p, text="x") for p in pages
        ]

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        return [ScoredChunk(chunk=c, score=1.0) for c in self._chunks[:k]]


def test_recall_at_k_and_mrr() -> None:
    def factory(doc: str) -> Retriever:
        return RankedPages(doc, {"A": [9, 7, 5], "B": [1, 2, 3]}[doc])

    # A: gold page 7 sits at rank 2.  B: gold pages 8 and 3 -> only page 3 (rank 3) is found.
    report = evaluate_retrieval([example("A", [7]), example("B", [8, 3])], factory, ks=(1, 2, 3))
    assert report.n == 2
    assert report.recall_at_k == {1: 0.0, 2: 0.5, 3: 0.75}  # mean of (A: 0,1,1) and (B: 0,0,.5)
    assert report.mrr == (1 / 2 + 1 / 3) / 2


def test_missing_filings_are_skipped_not_fatal() -> None:
    def factory(doc: str) -> Retriever:
        if doc == "MISSING":
            raise FileNotFoundError(doc)
        return RankedPages(doc, [4])

    report = evaluate_retrieval([example("MISSING", [4]), example("OK", [4])], factory, ks=(1,))
    assert (report.n, report.n_skipped) == (1, 1)
    assert report.recall_at_k == {1: 1.0}


def test_empty_input() -> None:
    report = evaluate_retrieval([], lambda d: RankedPages(d, []), ks=(1, 5))
    assert (report.n, report.mrr, report.recall_at_k) == (0, 0.0, {1: 0.0, 5: 0.0})
