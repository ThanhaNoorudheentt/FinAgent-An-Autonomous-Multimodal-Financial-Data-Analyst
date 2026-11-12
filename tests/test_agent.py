from __future__ import annotations

from pathlib import Path

import pytest

from finagent.agent.agent import INSUFFICIENT, AgentConfig, FinAgent
from finagent.agent.verifier import confidence_score, extract_numbers, verify_answer
from finagent.llm.base import QuotaExhaustedError
from finagent.llm.fake import FakeClient
from tests.helpers import Script, make_filing

QUESTION = "What was FY2018 capital expenditure in USD millions?"
FIND = {
    "thought": "look up capex",
    "action": "find_line_items",
    "query": "capex",
    "statement": "cash_flow",
}
CALC = {"thought": "capex is shown negative", "action": "calculate", "expression": "abs(-1577)"}


def final(answer: str, confidence: float = 0.9) -> dict[str, object]:
    return {"thought": "done", "action": "final_answer", "answer": answer, "confidence": confidence}


def run(tmp_path: Path, steps: list, *, config: AgentConfig | None = None, plan=None):  # type: ignore[no-untyped-def]
    client = FakeClient(Script(steps, plan))
    answer = FinAgent(client, config).answer(QUESTION, make_filing(tmp_path))
    return answer, client


def test_happy_path_uses_tools_and_records_evidence(tmp_path: Path) -> None:
    answer, client = run(tmp_path, [FIND, CALC, final("$1,577 million in FY2018")])
    assert answer.answer == "$1,577 million in FY2018"
    assert answer.confidence == 0.9  # every number is grounded
    assert answer.llm_calls == 4 == len(client.calls)  # plan + three steps
    [calc] = answer.calculations
    assert calc.code == "abs(-1577)"
    assert calc.result == 1577.0
    assert calc.description == "capex is shown negative"
    assert any("1,577" in e.content and e.kind == "table" and e.page == 1 for e in calc.inputs)
    assert {e.page for e in answer.evidence} >= {1}
    assert answer.notes == []


def test_prompt_contains_plan_context_and_prior_observations(tmp_path: Path) -> None:
    _, client = run(tmp_path, [FIND, final("$1,577 million")])
    assert "CANDIDATE LINE ITEMS" in client.calls[1]
    assert "p1 [cash_flow] Purchases of property, plant and equipment" in client.calls[1]
    assert "find_line_items(capex) ->" in client.calls[2]  # the observation is fed back


def test_ungrounded_number_lowers_confidence_and_adds_a_note(tmp_path: Path) -> None:
    answer, _ = run(tmp_path, [final("$1,900 million", confidence=0.8)])
    assert answer.confidence == pytest.approx(0.4)
    assert "1900" in answer.notes[0]


def test_invalid_step_json_is_skipped_not_fatal(tmp_path: Path) -> None:
    answer, _ = run(tmp_path, ["not json at all", final("$1,577 million")])
    assert answer.answer == "$1,577 million"
    assert any("invalid JSON" in n for n in answer.notes)
    assert answer.confidence == pytest.approx(0.9 * 0.95)  # small penalty for the failed step


def test_invalid_plan_falls_back_to_the_question(tmp_path: Path) -> None:
    answer, client = run(tmp_path, [final("$1,577 million")], plan="garbage")
    assert any("Planner output was invalid" in n for n in answer.notes)
    assert answer.evidence  # retrieval still ran on the raw question
    assert answer.llm_calls == len(client.calls)


def test_calculator_error_becomes_an_observation(tmp_path: Path) -> None:
    bad = {"thought": "oops", "action": "calculate", "expression": "1 / 0"}
    answer, client = run(tmp_path, [bad, final("$1,577 million")])
    assert "Error: division by zero" in client.calls[-1]
    assert answer.calculations == []
    assert answer.confidence == pytest.approx(0.9 * 0.95)


def test_sql_and_search_tools(tmp_path: Path) -> None:
    sql = {
        "thought": "query",
        "action": "sql",
        "sql": "SELECT label, value FROM line_items WHERE period='2018' AND value < -1500",
    }
    search = {"thought": "text", "action": "search_filing", "query": "dividends paid"}
    _, client = run(tmp_path, [sql, search, final("$1,577 million")])
    assert "Purchases of property, plant and equipment (PP&E) | -1577.0" in client.calls[-1]
    assert "(page 1)" in client.calls[-1]


def test_missing_tool_argument_is_reported(tmp_path: Path) -> None:
    _, client = run(
        tmp_path, [{"thought": "x", "action": "search_filing"}, final("$1,577 million")]
    )
    assert "Error: search_filing needs 'query'" in client.calls[-1]


def test_step_budget_forces_a_final_answer(tmp_path: Path) -> None:
    answer, client = run(
        tmp_path, [FIND, FIND, final("$1,577 million")], config=AgentConfig(max_steps=2)
    )
    assert "Step budget exhausted" in answer.notes[0]
    assert "MUST be final_answer" in client.calls[-1]
    assert answer.llm_calls == 4  # plan + 2 steps + forced final


def test_forced_final_that_is_not_final_yields_abstention(tmp_path: Path) -> None:
    answer, _ = run(tmp_path, [FIND, FIND, FIND], config=AgentConfig(max_steps=2))
    assert answer.answer == INSUFFICIENT
    assert answer.confidence == 0.0
    assert "No valid final answer" in answer.notes[-1]


def test_abstention_has_zero_confidence(tmp_path: Path) -> None:
    answer, _ = run(tmp_path, [final(INSUFFICIENT, confidence=0.9)])
    assert answer.confidence == 0.0


def test_quota_errors_propagate(tmp_path: Path) -> None:
    def boom(_prompt: str) -> str:
        raise QuotaExhaustedError("daily limit")

    with pytest.raises(QuotaExhaustedError):
        FinAgent(FakeClient(boom)).answer(QUESTION, make_filing(tmp_path))


def test_extract_numbers_and_year_and_small_count_exemptions() -> None:
    assert extract_numbers("Capex was $1,577.5 million in 2018 (-3.2%)") == [1577.5, 2018.0, 3.2]
    assert extract_numbers("FY2018") == []  # digits glued to letters are not standalone numbers
    v = verify_answer("In 2018 there were 3 segments and sales of 32,765", ["Net sales 32,765"], [])
    assert v.grounded
    assert v.checked == 1  # the year and the small count are not checked


def test_grounding_allows_rounding_unit_scaling_and_percent() -> None:
    assert verify_answer("$8.7 billion", ["PP&E net 8,738 (millions)"], []).grounded
    assert verify_answer("14.5%", [], [0.145]).grounded
    assert not verify_answer("$9.5 billion", ["PP&E net 8,738"], []).grounded
    assert verify_answer("14.46%", [], [14.463]).grounded  # computed percentage, rounded
    assert verify_answer("$1,577.0 million", ["(1,577)"], []).grounded


def test_grounding_is_strict_about_digits_the_answer_claims() -> None:
    sources = ["Capex 1,577 1,373 1,420 and capex 1,828 2,152"]
    assert not verify_answer("$1,900 million", sources, []).grounded
    assert not verify_answer("$1,580 million", sources, []).grounded  # near, but not 1,577
    assert verify_answer("$1,577 million", sources, []).grounded


def test_confidence_score() -> None:
    ok = verify_answer("5000", ["5000"], [])
    bad = verify_answer("5000", [], [])
    assert confidence_score(None, ok, abstained=False, tool_errors=0) == 0.5
    assert confidence_score(2.0, ok, abstained=False, tool_errors=0) == 1.0  # clamped
    assert confidence_score(0.8, bad, abstained=False, tool_errors=0) == 0.4
    assert confidence_score(0.8, ok, abstained=True, tool_errors=0) == 0.0
