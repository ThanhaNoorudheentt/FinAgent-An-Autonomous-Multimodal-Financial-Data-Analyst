"""FinanceBench scoring: LLM-judged correctness plus retrieval (evidence) recall."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from finagent.evaluation.judge import judge_answer
from finagent.evaluation.metrics import evidence_recall
from finagent.llm.base import LLMClient
from finagent.schemas import Answer, FinanceBenchExample


def make_judge_scorer(
    judge: LLMClient, *, log_path: Path | None = None
) -> Callable[[FinanceBenchExample, Answer], bool]:
    """Build a scorer that grades answers with an LLM judge.

    Verdict reasons are appended to ``log_path`` (JSONL) so wrong answers can be inspected.
    Prefer a judge model different from the answering model when possible, to limit
    self-preference bias.
    """

    def score(example: FinanceBenchExample, answer: Answer) -> bool:
        verdict = judge_answer(judge, example.question, example.answer, answer.answer)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                record = {
                    "id": example.financebench_id,
                    "correct": verdict.correct,
                    "reason": verdict.reason,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return verdict.correct

    return score


def evidence_metrics(example: FinanceBenchExample, answer: Answer) -> dict[str, float]:
    """Retrieval recall: share of gold (doc, page) evidence present among retrieved chunks."""
    return {"evidence_recall": evidence_recall(answer.evidence, example.evidence)}
