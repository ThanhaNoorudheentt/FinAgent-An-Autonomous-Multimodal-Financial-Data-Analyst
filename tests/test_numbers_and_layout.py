from __future__ import annotations

import pytest

from finagent.ingestion.layout import Word, numeric_row_count, words_to_rows
from finagent.ingestion.numbers import is_dash, parse_financial_number


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("1,577", 1577.0),
        ("(1,577)", -1577.0),
        ("-5", -5.0),
        ("\u22125.5", -5.5),
        ("$ 5,363", 5363.0),
        ("14.1%", 14.1),
        ("(3.2)%", -3.2),
        ("Net income", None),
        ("\u2014", None),
        ("", None),
        ("2018 results", None),
    ],
)
def test_parse_financial_number(cell: str, expected: float | None) -> None:
    assert parse_financial_number(cell) == expected


def test_is_dash() -> None:
    assert is_dash("\u2014")
    assert not is_dash("0")


def w(x0: float, y: float, text: str, width: float = 20.0) -> Word:
    return Word(x0, y, x0 + width, y + 8, text)


def test_words_to_rows_joins_a_visual_row_and_marks_column_gaps() -> None:
    words = [
        w(72, 100, "Net", 15),
        w(90, 100, "income", 30),
        w(300, 100, "5,363"),
        w(380, 100, "4,869"),
    ]
    assert words_to_rows(words) == ["Net income | 5,363 | 4,869"]


def test_words_to_rows_separates_rows_and_tolerates_small_vertical_jitter() -> None:
    words = [w(72, 100, "A", 10), w(300, 101.5, "1"), w(72, 120, "B", 10), w(300, 120, "2")]
    assert words_to_rows(words) == ["A | 1", "B | 2"]


def test_standalone_dollar_signs_are_dropped() -> None:
    words = [w(72, 100, "Cash", 25), w(290, 100, "$", 5), w(300, 100, "2,853")]
    assert words_to_rows(words) == ["Cash | 2,853"]


def test_numeric_row_count_ignores_prose_and_headers() -> None:
    rows = [
        "Consolidated Statement of Cash Flows",
        "(Millions) | 2018 | 2017",
        "Net income | 5 | 6",
        "Text",
    ]
    assert numeric_row_count(rows) == 2  # the year header counts as numeric cells; prose does not
