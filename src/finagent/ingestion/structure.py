"""Recognise primary financial statements and which statement a question refers to."""

from __future__ import annotations

import re
from collections.abc import Sequence

from finagent.ingestion.layout import MIN_ROWS_FOR_TABLE

BALANCE_SHEET = "balance_sheet"
INCOME_STATEMENT = "income_statement"
CASH_FLOW = "cash_flow"

# Matched against the page heading with all whitespace removed, so titles broken by PDF
# extraction ("Cash Flow s") still match.
_PAGE_TITLES: dict[str, tuple[str, ...]] = {
    BALANCE_SHEET: (
        "balancesheet",
        "statementoffinancialposition",
        "statementsoffinancialposition",
        "statementoffinancialcondition",
        "statementsoffinancialcondition",
    ),
    INCOME_STATEMENT: (
        "statementofincome",
        "statementsofincome",
        "statementofoperations",
        "statementsofoperations",
        "statementofearnings",
        "statementsofearnings",
        "incomestatement",
        "statementofprofitorloss",
    ),
    CASH_FLOW: ("statementofcashflow", "statementsofcashflow"),
}

_QUERY_PATTERNS: dict[str, re.Pattern[str]] = {
    BALANCE_SHEET: re.compile(
        r"balance[\s-]*sheet|statements? of financial (?:position|condition)"
    ),
    INCOME_STATEMENT: re.compile(
        r"income statement|statements? of (?:income|operations|earnings)|profit and loss|p&l"
        r"|profit & loss"
    ),
    CASH_FLOW: re.compile(r"cash[\s-]*flows? statement|statements? of cash[\s-]*flows?"),
}

_HEADING_LINES = 12


def classify_page(rows: Sequence[str], numeric_rows: int) -> tuple[str, ...]:
    """Tag a page as a primary statement from its heading, if it also looks like a table."""
    if numeric_rows < MIN_ROWS_FOR_TABLE:
        return ()
    heading = "".join(" ".join(rows[:_HEADING_LINES]).lower().split())
    return tuple(tag for tag, titles in _PAGE_TITLES.items() if any(t in heading for t in titles))


def detect_statement_intent(question: str) -> tuple[str, ...]:
    """Statements the question explicitly names (e.g. "...shown in the cash flow statement")."""
    text = question.lower()
    return tuple(tag for tag, pattern in _QUERY_PATTERNS.items() if pattern.search(text))
