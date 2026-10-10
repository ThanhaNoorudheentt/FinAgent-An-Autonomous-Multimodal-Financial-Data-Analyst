from __future__ import annotations

import json
from pathlib import Path

from finagent.evaluation.financebench import evidence_metrics, make_judge_scorer
from finagent.evaluation.harness import run_evaluation
from finagent.llm.fake import FakeClient
from finagent.retrieval.retrievers import BM25Retriever
from finagent.schemas import Answer, Chunk, Evidence, FinanceBenchEvidence, FinanceBenchExample
from finagent.systems.rag import RagFinanceBenchSystem


def make_example(page: int = 1) -> FinanceBenchExample:
    return FinanceBenchExample(
        financebench_id="fb_1",
        company="ACME",
        doc_name="ACME_2023_10K",
        question_type="metrics-generated",
        question="What was FY2023 capital expenditure?",
        answer="$1,577 million",
        evidence=[
            FinanceBenchEvidence(
                evidence_text="x", doc_name="ACME_2023_10K", evidence_page_num=page
            )
        ],
    )


def make_chunks() -> list[Chunk]:
    texts = ["risk factors competition", "capital expenditure purchases 1,577", "goodwill note"]
    return [
        Chunk(chunk_id=f"ACME_2023_10K:p{i}:c0", doc_id="ACME_2023_10K", page=i, text=t)
        for i, t in enumerate(texts)
    ]


def test_rag_system_prompts_with_sources_and_returns_cited_evidence() -> None:
    client = FakeClient(json.dumps({"reasoning": "read it", "answer": " $1,577 million "}))
    system = RagFinanceBenchSystem(client, lambda _doc: BM25Retriever(make_chunks()), k=2)
    answer = system.answer(make_example())
    assert answer.answer == "$1,577 million"
    assert answer.evidence[0].page == 1  # the capex chunk was retrieved first
    assert "(page 1)" in client.calls[0]
    assert "What was FY2023 capital expenditure?" in client.calls[0]


def test_rag_system_handles_no_hits() -> None:
    client = FakeClient(json.dumps({"reasoning": "", "answer": "Insufficient information"}))
    system = RagFinanceBenchSystem(client, lambda _d: BM25Retriever(make_chunks()), k=3)
    ex = make_example().model_copy(update={"question": "zzzz qqqq"})
    answer = system.answer(ex)
    assert answer.evidence == []
    assert "(none)" in client.calls[0]


def test_evidence_metrics() -> None:
    ans = Answer(answer="x", evidence=[Evidence(doc_id="ACME_2023_10K", page=1, content="")])
    assert evidence_metrics(make_example(page=1), ans) == {"evidence_recall": 1.0}
    assert evidence_metrics(make_example(page=2), ans) == {"evidence_recall": 0.0}


def test_judge_scorer_logs_reasons(tmp_path: Path) -> None:
    judge = FakeClient('{"correct": false, "reason": "wrong number"}')
    log = tmp_path / "judge.jsonl"
    score = make_judge_scorer(judge, log_path=log)
    assert score(make_example(), Answer(answer="$1,000 million")) is False
    record = json.loads(log.read_text().strip())
    assert record == {"id": "fb_1", "correct": False, "reason": "wrong number"}


def test_harness_aggregates_metrics(tmp_path: Path) -> None:
    summary = run_evaluation(
        [make_example(page=1), make_example(page=2).model_copy(update={"financebench_id": "fb_2"})],
        answer_fn=lambda _e: Answer(
            answer="a", evidence=[Evidence(doc_id="ACME_2023_10K", page=1, content="")]
        ),
        score_fn=lambda _e, _a: True,
        id_fn=lambda e: e.financebench_id,
        out_path=tmp_path / "r.jsonl",
        metrics_fn=evidence_metrics,
    )
    assert summary.mean_metrics == {"evidence_recall": 0.5}
