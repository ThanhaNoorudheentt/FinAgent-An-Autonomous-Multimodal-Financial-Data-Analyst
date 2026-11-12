from __future__ import annotations

from finagent.ingestion.structure import classify_page, detect_statement_intent
from finagent.retrieval.synonyms import expand_query

NUMERIC_ROWS = 10


def test_classifies_primary_statements_even_with_broken_titles() -> None:
    assert classify_page(["Consolidated Statement of Cash Flow s", "x"], NUMERIC_ROWS) == (
        "cash_flow",
    )
    assert classify_page(["CONSOLIDATED BALANCE SHEET"], NUMERIC_ROWS) == ("balance_sheet",)
    assert classify_page(["Consolidated Statements of Operations"], NUMERIC_ROWS) == (
        "income_statement",
    )


def test_index_or_prose_pages_are_not_tagged() -> None:
    assert classify_page(["Consolidated Balance Sheet 55"], numeric_rows=2) == ()
    assert classify_page(["Risk factors", "competition"], NUMERIC_ROWS) == ()


def test_comprehensive_income_is_not_the_income_statement() -> None:
    assert classify_page(["Consolidated Statements of Comprehensive Income"], NUMERIC_ROWS) == ()


def test_detects_statement_named_in_a_question() -> None:
    q = "Answer using information shown in the cash flow statement: what was capex?"
    assert detect_statement_intent(q) == ("cash_flow",)
    assert detect_statement_intent("Per the balance sheet and income statement") == (
        "balance_sheet",
        "income_statement",
    )
    assert detect_statement_intent("What is free cash flow?") == ()  # not a statement reference


def test_expand_query_adds_accounting_synonyms_only_when_triggered() -> None:
    assert "purchases of property plant and equipment" in expand_query("What was FY2018 capex?")
    assert expand_query("Who is the CEO?") == "Who is the CEO?"
