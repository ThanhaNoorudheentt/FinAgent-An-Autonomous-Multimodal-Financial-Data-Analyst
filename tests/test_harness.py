from __future__ import annotations

from pathlib import Path

from finagent.evaluation.harness import run_evaluation, summarize
from finagent.schemas import Answer


def test_run_scores_and_summarizes(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    summary = run_evaluation(
        [1, 2, 3, 4],
        answer_fn=lambda x: Answer(answer=str(x)),
        score_fn=lambda x, a: x % 2 == 0,
        id_fn=str,
        out_path=out,
    )
    assert (summary.n, summary.n_correct, summary.n_errors) == (4, 2, 0)
    assert summary.accuracy == 0.5


def test_errors_are_recorded_and_count_as_incorrect(tmp_path: Path) -> None:
    def answer_fn(x: int) -> Answer:
        if x == 2:
            raise RuntimeError("boom")
        return Answer(answer="ok")

    summary = run_evaluation(
        [1, 2],
        answer_fn=answer_fn,
        score_fn=lambda _x, _a: True,
        id_fn=str,
        out_path=tmp_path / "r.jsonl",
    )
    assert summary.n == 2
    assert summary.n_correct == 1
    assert summary.n_errors == 1


def test_resume_skips_done_and_retries_errors(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    calls: list[int] = []
    fail = {"on": True}

    def answer_fn(x: int) -> Answer:
        calls.append(x)
        if x == 2 and fail["on"]:
            raise RuntimeError("transient")
        return Answer(answer="ok")

    kwargs = {"score_fn": lambda _x, _a: True, "id_fn": str, "out_path": out}
    run_evaluation([1, 2, 3], answer_fn=answer_fn, **kwargs)  # type: ignore[arg-type]
    assert calls == [1, 2, 3]

    calls.clear()
    fail["on"] = False
    summary = run_evaluation([1, 2, 3], answer_fn=answer_fn, **kwargs)  # type: ignore[arg-type]
    assert calls == [2]  # only the previously failed example is re-run
    assert (summary.n, summary.n_correct, summary.n_errors) == (3, 3, 0)
    assert summarize(out).n == 3


def test_limit(tmp_path: Path) -> None:
    summary = run_evaluation(
        list(range(10)),
        answer_fn=lambda x: Answer(answer=str(x)),
        score_fn=lambda _x, _a: True,
        id_fn=str,
        out_path=tmp_path / "r.jsonl",
        limit=3,
    )
    assert summary.n == 3


def test_quota_exhaustion_aborts_run_and_keeps_progress(tmp_path: Path) -> None:
    import pytest

    from finagent.llm.base import QuotaExhaustedError

    out = tmp_path / "r.jsonl"

    def answer_fn(x: int) -> Answer:
        if x == 3:
            raise QuotaExhaustedError("daily limit")
        return Answer(answer="ok")

    kwargs = {"score_fn": lambda _x, _a: True, "id_fn": str, "out_path": out}
    with pytest.raises(QuotaExhaustedError):
        run_evaluation([1, 2, 3, 4, 5], answer_fn=answer_fn, **kwargs)  # type: ignore[arg-type]
    assert summarize(out).n == 2  # examples 1 and 2 saved; 3 not recorded; 4, 5 never started

    calls: list[int] = []

    def recovered(x: int) -> Answer:
        calls.append(x)
        return Answer(answer="ok")

    run_evaluation([1, 2, 3, 4, 5], answer_fn=recovered, **kwargs)  # type: ignore[arg-type]
    assert calls == [3, 4, 5]  # resume picks up exactly where it stopped
