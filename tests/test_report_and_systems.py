from __future__ import annotations

import json

import pytest

from finagent.evaluation.splits import split_examples
from finagent.llm.base import Image
from finagent.llm.fake import FakeClient
from finagent.reporting.report import render_report
from finagent.retrieval.retrievers import BM25Retriever
from finagent.schemas import Answer, Calculation, Chunk, Evidence, FinanceBenchExample, FinQAExample
from finagent.systems.program import ProgramFinQASystem
from finagent.systems.rag import RagFinanceBenchSystem


def test_report_contains_all_sections() -> None:
    answer = Answer(
        answer="$1,577 million",
        confidence=0.9,
        calculations=[Calculation(description="abs | value", code="abs(-1577)", result=1577.0)],
        evidence=[
            Evidence(doc_id="ACME", page=59, kind="table", content="Purchases of PP&E | (1,577)")
        ],
        notes=["Numbers not traceable: 5."],
    )
    report = render_report("What was capex?", answer, doc_id="ACME")
    for expected in (
        "# FinAgent analyst report",
        "**Question:** What was capex?",
        "$1,577 million",
        "90%",
        "| 1 | abs / value | `abs(-1577)` | 1577 |",
        "ACME, page 59** (table)",
        "## Verification notes",
        "Numbers not traceable",
    ):
        assert expected in report


def test_report_minimal_answer_has_no_empty_sections() -> None:
    report = render_report("q", Answer(answer="a"), doc_id="D")
    assert "## Calculations" not in report
    assert "## Evidence" not in report
    assert "Confidence" not in report


def test_report_truncates_long_evidence() -> None:
    evidence = [Evidence(doc_id="D", page=i, content="x" * 500) for i in range(15)]
    report = render_report("q", Answer(answer="a", evidence=evidence), doc_id="D")
    assert "... and 3 more passages" in report
    assert "\u2026" in report


def test_split_is_deterministic_disjoint_and_roughly_30_70() -> None:
    ids = [f"id_{i}" for i in range(500)]
    dev = split_examples(ids, "dev", key=str)
    test = split_examples(ids, "test", key=str)
    assert set(dev).isdisjoint(test)
    assert sorted(dev + test) == sorted(ids)
    assert split_examples(ids, "all", key=str) == ids
    assert dev == split_examples(ids, "dev", key=str)
    assert 0.2 < len(dev) / len(ids) < 0.4


@pytest.fixture
def finqa() -> FinQAExample:
    return FinQAExample(
        id="X/1",
        question="what is the percentage change?",
        answer="1.6%",
        exe_ans=0.01639,
        pre_text=["a"],
        post_text=["b"],
        table=[["", "v"], ["2014", "5735"], ["2015", "5829"]],
    )


def test_program_system_computes_with_the_tool(finqa: FinQAExample) -> None:
    reply = {
        "reasoning": "change over 2014",
        "expression": "(5829 - 5735) / 5735 * 100",
        "answer_format": "percent",
        "fallback_answer": "wrong",
    }
    answer = ProgramFinQASystem(FakeClient(json.dumps(reply))).answer(finqa)
    assert answer.answer == "1.64%"
    assert answer.calculations[0].code == "(5829 - 5735) / 5735 * 100"
    assert answer.llm_calls == 1


def test_program_system_falls_back_on_calculator_error(finqa: FinQAExample) -> None:
    reply = {"reasoning": "r", "expression": "1,5 + 2", "fallback_answer": "42"}
    answer = ProgramFinQASystem(FakeClient(json.dumps(reply))).answer(finqa)
    assert answer.answer == "42"
    assert "Calculator error" in answer.notes[0]


def test_program_system_yes_no(finqa: FinQAExample) -> None:
    reply = {"reasoning": "r", "expression": "5829 > 5735", "answer_format": "yes_no"}
    assert ProgramFinQASystem(FakeClient(json.dumps(reply))).answer(finqa).answer == "yes"


def bench_example() -> FinanceBenchExample:
    return FinanceBenchExample(
        financebench_id="fb",
        company="X",
        doc_name="ACME",
        question_type="t",
        question="capex purchases?",
        answer="a",
        evidence=[],
    )


def test_multimodal_rag_attaches_images_of_top_distinct_pages() -> None:
    chunks = [
        Chunk(chunk_id=f"ACME:p{p}:c{c}", doc_id="ACME", page=p, text="capex purchases")
        for p, c in [(7, 0), (7, 1), (3, 0), (9, 0)]
    ]
    seen: dict[str, object] = {}

    def images(doc: str, pages: list[int]) -> list[Image]:
        seen["args"] = (doc, pages)
        return [Image(b"png")] * len(pages)

    client = FakeClient(json.dumps({"reasoning": "r", "answer": "a"}))
    system = RagFinanceBenchSystem(
        client, lambda _d: BM25Retriever(chunks), k=4, page_images=images, n_images=2
    )
    answer = system.answer(bench_example())
    assert seen["args"] == ("ACME", [7, 3])  # distinct pages in rank order, capped at n_images
    assert "Images of pages 7, 3 are attached" in client.calls[0]
    assert answer.llm_calls == 1
    assert system.name == "mmrag-k4"


def test_text_rag_sends_no_images() -> None:
    client = FakeClient(json.dumps({"reasoning": "r", "answer": "a"}))
    chunks = [Chunk(chunk_id="ACME:p1:c0", doc_id="ACME", page=1, text="capex purchases")]
    RagFinanceBenchSystem(client, lambda _d: BM25Retriever(chunks)).answer(bench_example())
    assert "attached" not in client.calls[0]
