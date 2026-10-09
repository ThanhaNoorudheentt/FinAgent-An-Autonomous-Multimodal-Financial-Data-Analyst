"""Benchmark runner: resumable, error-tolerant, and writes one JSON record per example."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from finagent.schemas import Answer

logger = logging.getLogger(__name__)

E = TypeVar("E")


@dataclass(frozen=True)
class EvalSummary:
    n: int
    n_correct: int
    n_errors: int
    mean_latency_s: float

    @property
    def accuracy(self) -> float:
        """Errors count as incorrect so crashes can never inflate the score."""
        return self.n_correct / self.n if self.n else 0.0


def _read_records(path: Path) -> dict[str, dict[str, Any]]:
    """Latest record per id (later lines override earlier ones, e.g. after a retry)."""
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                records[rec["id"]] = rec
    return records


def summarize(path: Path) -> EvalSummary:
    records = list(_read_records(path).values())
    n = len(records)
    return EvalSummary(
        n=n,
        n_correct=sum(1 for r in records if r["correct"]),
        n_errors=sum(1 for r in records if r["error"] is not None),
        mean_latency_s=(sum(r["latency_s"] for r in records) / n) if n else 0.0,
    )


def run_evaluation(
    examples: Sequence[E],
    *,
    answer_fn: Callable[[E], Answer],
    score_fn: Callable[[E, Answer], bool],
    id_fn: Callable[[E], str],
    out_path: Path,
    limit: int | None = None,
) -> EvalSummary:
    """Run ``answer_fn`` over ``examples`` and append results to ``out_path`` (JSONL).

    Already-completed examples are skipped, so an interrupted run (or a Colab session reset)
    resumes where it stopped and never re-spends API budget. Examples that previously
    errored are retried. One failing example never aborts the run.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {i for i, r in _read_records(out_path).items() if r["error"] is None}
    todo = [ex for ex in list(examples)[:limit] if id_fn(ex) not in done]
    logger.info("%d examples to run (%d already done)", len(todo), len(done))

    with out_path.open("a", encoding="utf-8") as fh:
        for idx, ex in enumerate(todo, start=1):
            start = time.perf_counter()
            record: dict[str, Any] = {"id": id_fn(ex), "correct": False, "answer": None}
            try:
                answer = answer_fn(ex)
                record["answer"] = answer.model_dump()
                record["correct"] = bool(score_fn(ex, answer))
                record["error"] = None
            except Exception as exc:  # one bad example must not kill the whole run
                logger.exception("Example %s failed", record["id"])
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["latency_s"] = round(time.perf_counter() - start, 3)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            logger.info("[%d/%d] %s correct=%s", idx, len(todo), record["id"], record["correct"])

    return summarize(out_path)
