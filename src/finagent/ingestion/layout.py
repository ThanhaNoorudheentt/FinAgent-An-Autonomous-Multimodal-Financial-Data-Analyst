"""Rebuild table rows from word positions.

PDF text extraction usually emits one table cell per line, which separates a label such as
"Purchases of property, plant and equipment" from its values. Grouping words by vertical position
restores the row, and a wide horizontal gap marks a column break, giving lines like
``Purchases of PP&E | (1,577) | (1,373) | (1,420)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from finagent.ingestion.numbers import parse_financial_number

CELL_SEPARATOR = " | "
MIN_ROWS_FOR_TABLE = 8  # a page needs this many numeric rows to be treated as a table


@dataclass(frozen=True)
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


def words_to_rows(
    words: Sequence[Word], *, y_tolerance: float = 3.0, column_gap: float = 14.0
) -> list[str]:
    """Group words into visual rows; separate cells separated by more than ``column_gap`` points.

    Stand-alone ``$`` signs are dropped so ``$ 5,363`` becomes the cell ``5,363``.
    """
    kept = [w for w in words if w.text.strip() and w.text.strip() != "$"]
    ordered = sorted(kept, key=lambda w: ((w.y0 + w.y1) / 2, w.x0))
    rows: list[list[Word]] = []
    for word in ordered:
        centre = (word.y0 + word.y1) / 2
        if rows and abs(centre - _mean_centre(rows[-1])) <= y_tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])
    return [_join(sorted(row, key=lambda w: w.x0), column_gap) for row in rows]


def _mean_centre(row: list[Word]) -> float:
    return sum((w.y0 + w.y1) / 2 for w in row) / len(row)


def _join(row: list[Word], gap: float) -> str:
    parts = [row[0].text]
    for left, right in pairwise(row):
        parts.append(CELL_SEPARATOR if right.x0 - left.x1 > gap else " ")
        parts.append(right.text)
    return "".join(parts)


def numeric_row_count(rows: Sequence[str]) -> int:
    """Rows that have a label or several cells and at least one numeric value cell."""
    count = 0
    for row in rows:
        cells = row.split(CELL_SEPARATOR)
        if len(cells) >= 2 and any(parse_financial_number(c) is not None for c in cells[1:]):
            count += 1
    return count
