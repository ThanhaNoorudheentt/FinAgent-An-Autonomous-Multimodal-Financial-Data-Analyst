"""Retrieval-only evaluation: how often is the gold evidence page retrieved?

Needs no generation or judge calls, so it is free with BM25 and cheap with embeddings. Use it to
choose chunking and retrievers *before* spending LLM budget on end-to-end runs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from finagent.evaluation.metrics import evidence_recall
from finagent.schemas import Evidence, FinanceBenchExample
from finagent.systems.rag import RetrieverFactory


@dataclass(frozen=True)
class RetrievalRow:
    """Per-question result, kept so misses can be inspected rather than just averaged."""

    id: str
    question_type: str
    question_reasoning: str
    gold_pages: list[int]
    retrieved_pages: list[int]  # page of each retrieved chunk, best first (repeats kept)
    first_gold_rank: int | None  # 1-based rank of the first chunk on a gold page
    recall_at_k: dict[int, float]


@dataclass(frozen=True)
class RetrievalReport:
    n: int
    n_skipped: int  # questions whose filing PDF was not available
    recall_at_k: dict[int, float]  # k = number of chunks retrieved; recall is over gold pages
    mrr: float  # mean reciprocal rank of the first chunk on a gold page (0 if absent)
    rows: list[RetrievalRow] = field(default_factory=list)


def evaluate_retrieval(
    examples: Iterable[FinanceBenchExample],
    factory: RetrieverFactory,
    *,
    ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> RetrievalReport:
    max_k = max(ks)
    rows: list[RetrievalRow] = []
    skipped = 0
    for ex in examples:
        try:
            retriever = factory(ex.doc_name)
        except FileNotFoundError:
            skipped += 1
            continue
        hits = retriever.retrieve(ex.question, max_k)
        found = [Evidence(doc_id=h.chunk.doc_id, page=h.chunk.page, content="") for h in hits]
        gold = {(g.doc_name, g.evidence_page_num) for g in ex.evidence}
        rank = next((i + 1 for i, e in enumerate(found) if (e.doc_id, e.page) in gold), None)
        rows.append(
            RetrievalRow(
                id=ex.financebench_id,
                question_type=ex.question_type,
                question_reasoning=ex.question_reasoning or "(none)",
                gold_pages=sorted({g.evidence_page_num for g in ex.evidence}),
                retrieved_pages=[e.page for e in found if e.page is not None],
                first_gold_rank=rank,
                recall_at_k={k: evidence_recall(found[:k], ex.evidence) for k in ks},
            )
        )
    n = len(rows)
    return RetrievalReport(
        n=n,
        n_skipped=skipped,
        recall_at_k={k: (sum(r.recall_at_k[k] for r in rows) / n if n else 0.0) for k in ks},
        mrr=(sum(1.0 / r.first_gold_rank for r in rows if r.first_gold_rank) / n) if n else 0.0,
        rows=rows,
    )


def breakdown(
    rows: Iterable[RetrievalRow], key: Callable[[RetrievalRow], str], k: int
) -> dict[str, tuple[int, float]]:
    """Mean recall@k per group, as ``{group: (count, mean_recall)}`` sorted by group name."""
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row.recall_at_k[k])
    return {g: (len(v), sum(v) / len(v)) for g, v in sorted(groups.items())}
