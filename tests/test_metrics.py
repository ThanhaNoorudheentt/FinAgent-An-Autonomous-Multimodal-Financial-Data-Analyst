from __future__ import annotations

import pytest

from finagent.evaluation.metrics import answers_match, evidence_recall, parse_number
from finagent.schemas import Evidence, FinanceBenchEvidence


@pytest.mark.parametrize(
    ("text", "value", "pct"),
    [
        ("94", 94.0, False),
        ("$1,577.00", 1577.0, False),
        ("(3.2)", -3.2, False),
        ("-5", -5.0, False),
        ("\u22125", -5.0, False),
        ("14.1%", 14.1, True),
        ("about 14.1 percent", 14.1, True),
        ("-$5", -5.0, False),
        (".5", 0.5, False),
    ],
)
def test_parse_number(text: str, value: float, pct: bool) -> None:
    parsed = parse_number(text)
    assert parsed is not None
    assert parsed.value == pytest.approx(value)
    assert parsed.is_percent is pct


def test_parse_number_none_when_absent() -> None:
    assert parse_number("no digits here") is None


@pytest.mark.parametrize(
    ("pred", "gold", "expected"),
    [
        ("94", 94.0, True),
        ("94.0", "94", True),
        ("95", 94.0, False),
        ("14.1%", 0.14136, True),  # percent vs fraction
        ("0.141", 0.14136, True),
        ("14.1", 0.14136, False),  # no % sign: not silently rescaled
        ("$1,577", "$1577.00", True),
        ("(2.5)", -2.5, True),
        ("Yes.", "yes", True),
        ("no", "yes", False),
        ("n/a", 5.0, False),
    ],
)
def test_answers_match(pred: str, gold: float | str, expected: bool) -> None:
    assert answers_match(pred, gold) is expected


def test_evidence_recall() -> None:
    gold = [
        FinanceBenchEvidence(evidence_text="x", doc_name="A", evidence_page_num=3),
        FinanceBenchEvidence(evidence_text="y", doc_name="A", evidence_page_num=7),
    ]
    got = [Evidence(doc_id="A", page=3, content="..."), Evidence(doc_id="B", page=7, content="")]
    assert evidence_recall(got, gold) == 0.5
    assert evidence_recall([], gold) == 0.0
    assert evidence_recall(got, []) == 1.0


def test_evidence_recall_page_offset() -> None:
    gold = [FinanceBenchEvidence(evidence_text="x", doc_name="A", evidence_page_num=3)]
    got = [Evidence(doc_id="A", page=4, content="")]
    assert evidence_recall(got, gold) == 0.0
    assert evidence_recall(got, gold, page_offset=1) == 1.0
