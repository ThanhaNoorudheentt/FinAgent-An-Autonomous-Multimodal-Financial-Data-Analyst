from __future__ import annotations

import sqlite3

import pytest

from finagent.tools.calculator import CalculatorError, evaluate, format_result
from finagent.tools.sql import SqlToolError, run_select


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", 7.0),
        ("(5829 - 5735) / 5735 * 100", pytest.approx(1.639, abs=1e-3)),
        ("-(2 + 3)", -5.0),
        ("pct_change(100, 125)", 25.0),
        ("pct_change(-100, -50)", 50.0),
        ("margin_pct(25, 200)", 12.5),
        ("ratio(9, 3)", 3.0),
        ("cagr_pct(100, 121, 2)", pytest.approx(10.0)),
        ("average(2, 4, 9)", 5.0),
        ("max(1, 7, 3) + min(4, 2)", 9.0),
        ("round(2.567, 1)", 2.6),
        ("2 ** 10", 1024.0),
    ],
)
def test_evaluate_arithmetic_and_helpers(expression: str, expected: float) -> None:
    assert evaluate(expression) == expected


def test_comparisons_return_bool() -> None:
    assert evaluate("5 > 3") is True
    assert evaluate("1 < 2 < 3") is True
    assert evaluate("2 == 3") is False


def test_variables() -> None:
    assert evaluate("a * 2", {"a": 4.0}) == 8.0
    with pytest.raises(CalculatorError, match="unknown name"):
        evaluate("b + 1")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "(1).__class__",
        "[x for x in range(3)]",
        "'abc'",
        "True + 1",
        "lambda: 1",
        "print(1)",
        "1 +",
        "x = 1",
    ],
)
def test_unsafe_or_invalid_expressions_are_rejected(expression: str) -> None:
    with pytest.raises(CalculatorError):
        evaluate(expression)


@pytest.mark.parametrize(
    "expression",
    ["1 / 0", "ratio(1, 0)", "pct_change(0, 5)", "9 ** 9 ** 9", "10 ** 200", "sqrt(-1)", "a" * 600],
)
def test_numeric_failures_are_clean_errors(expression: str) -> None:
    with pytest.raises(CalculatorError):
        evaluate(expression)


def test_thousands_separator_gives_a_helpful_error() -> None:
    with pytest.raises(CalculatorError, match="thousands separators"):
        evaluate("1,577 + 2")


def test_format_result() -> None:
    assert format_result(94.0) == "94"
    assert format_result(14.463) == "14.463"
    assert format_result(True) == "yes"
    assert format_result(1234567.891) == "1,234,567.89"


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE line_items (label TEXT, value REAL)")
    c.executemany("INSERT INTO line_items VALUES (?, ?)", [("Capex", -1577.0), ("Sales", 32765.0)])
    return c


def test_select_works_and_formats(conn: sqlite3.Connection) -> None:
    out = run_select(conn, "SELECT label, value FROM line_items ORDER BY value;")
    assert out.splitlines() == ["label | value", "Capex | -1577.0", "Sales | 32765.0"]
    assert run_select(conn, "SELECT * FROM line_items WHERE value > 1e9") == "(no rows)"


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE line_items",
        "DELETE FROM line_items",
        "INSERT INTO line_items VALUES ('x', 1)",
        "UPDATE line_items SET value = 0",
        "PRAGMA table_info(line_items)",
        "ATTACH DATABASE 'x.db' AS x",
        "SELECT 1; DROP TABLE line_items",
        "",
    ],
)
def test_non_select_statements_are_rejected_and_data_is_intact(
    conn: sqlite3.Connection, query: str
) -> None:
    with pytest.raises(SqlToolError):
        run_select(conn, query)
    assert conn.execute("SELECT COUNT(*) FROM line_items").fetchone() == (2,)


def test_row_limit_and_runaway_queries(conn: sqlite3.Connection) -> None:
    out = run_select(
        conn,
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n LIMIT 500) SELECT i FROM n",
        max_rows=5,
    )
    assert out.endswith("... truncated to 5 rows")
    with pytest.raises(SqlToolError):
        run_select(
            conn,
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n) SELECT COUNT(*) FROM n",
        )
