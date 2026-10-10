"""Retrieval-only evaluation: how often is the gold evidence page retrieved?

Needs no generation or judge calls, so it is free with BM25 and cheap with embeddings. Use it to
choose chunking and retrievers *before* spending LLM budget on end-to-end runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from finagent.evaluation.metrics import evidence_recall
from finagent.schemas import Evidence, FinanceBenchExample
from finagent.systems.rag import RetrieverFactory


@dataclass(frozen=True)
class RetrievalReport:
    n: int
    n_skipped: int  # questions whose filing PDF was not available
    recall_at_k: dict[int, float]  # k = number of chunks retrieved; recall is over gold pages
    mrr: float  # mean reciprocal rank of the first chunk on a gold page (0 if absent)


def evaluate_retrieval(
    examples: Iterable[FinanceBenchExample],
    factory: RetrieverFactory,
    *,
    ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> RetrievalReport:
    max_k = max(ks)
    totals = dict.fromkeys(ks, 0.0)
    rr_total = 0.0
    n = skipped = 0
    for ex in examples:
        try:
            retriever = factory(ex.doc_name)
        except FileNotFoundError:
            skipped += 1
            continue
        hits = retriever.retrieve(ex.question, max_k)
        found = [Evidence(doc_id=h.chunk.doc_id, page=h.chunk.page, content="") for h in hits]
        gold = {(g.doc_name, g.evidence_page_num) for g in ex.evidence}
        for k in ks:
            totals[k] += evidence_recall(found[:k], ex.evidence)
        rr_total += next(
            (1.0 / (i + 1) for i, e in enumerate(found) if (e.doc_id, e.page) in gold), 0.0
        )
        n += 1
    return RetrievalReport(
        n=n,
        n_skipped=skipped,
        recall_at_k={k: (totals[k] / n if n else 0.0) for k in ks},
        mrr=rr_total / n if n else 0.0,
    )
