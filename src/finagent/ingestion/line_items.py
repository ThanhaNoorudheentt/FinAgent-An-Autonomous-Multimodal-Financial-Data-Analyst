"""Structured line items parsed from table pages, searchable by label and queryable with SQL."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass

from finagent.ingestion.layout import CELL_SEPARATOR
from finagent.ingestion.numbers import YEAR_RE, parse_financial_number
from finagent.retrieval.bm25 import BM25
from finagent.retrieval.synonyms import expand_query
from finagent.schemas import Page

_UNIT_RE = re.compile(r"\b(?:in |\(in |\(dollars in |\(\$ in |\()?(millions|thousands|billions)\b")


@dataclass(frozen=True)
class LineItem:
    doc_id: str
    page: int
    row: int
    label: str
    values: tuple[float, ...]
    periods: tuple[str | None, ...]
    unit: str  # "millions", "thousands", "billions" or "" when the page does not say
    tags: tuple[str, ...]
    raw: str


def detect_unit(text: str) -> str:
    match = _UNIT_RE.search(text.lower())
    return match.group(1) if match else ""


def parse_line_items(pages: Iterable[Page]) -> list[LineItem]:
    """Extract ``label | value | value ...`` rows from table pages.

    Each row's values are paired with the year header seen most recently on the same page.
    A row holding only numbers is attached to a directly preceding label-only row (wrapped label).
    """
    items: list[LineItem] = []
    for page in pages:
        if not page.tabular:
            continue
        unit = detect_unit(page.text)
        header: list[str] = []
        pending: str | None = None
        for row_index, line in enumerate(page.text.splitlines()):
            cells = [c.strip() for c in line.split(CELL_SEPARATOR) if c.strip()]
            years = [c for c in cells if YEAR_RE.match(c)]
            if len(years) >= 2:
                header, pending = years, None
                continue
            label: str | None = None
            value_cells = cells
            if cells and parse_financial_number(cells[0]) is None:
                label, value_cells = cells[0], cells[1:]
            values = [v for c in value_cells if (v := parse_financial_number(c)) is not None]
            if label is not None and values:
                pending = None
            elif label is not None:
                pending = label
                continue
            elif values and pending is not None:
                label, pending = pending, None
            else:
                continue
            periods = tuple(header[i] if i < len(header) else None for i in range(len(values)))
            items.append(
                LineItem(
                    page.doc_id,
                    page.page,
                    row_index,
                    label,
                    tuple(values),
                    periods,
                    unit,
                    page.tags,
                    line,
                )
            )
    return items


def format_item(item: LineItem) -> str:
    pairs = ", ".join(
        f"{period or f'col{i + 1}'}={_fmt(v)}"
        for i, (period, v) in enumerate(zip(item.periods, item.values, strict=True))
    )
    tag = f" [{item.tags[0]}]" if item.tags else ""
    unit = f" ({item.unit})" if item.unit else ""
    return f"p{item.page}{tag} {item.label}: {pairs}{unit}"


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"


class LineItemTable:
    """All line items of one filing, with label search and a read-only SQL view."""

    def __init__(self, items: list[LineItem]) -> None:
        self._items = items
        self._bm25 = BM25([i.label for i in items])
        self._conn: sqlite3.Connection | None = None

    def __len__(self) -> int:
        return len(self._items)

    def search(self, query: str, *, k: int = 8, statement: str | None = None) -> list[LineItem]:
        """Best-matching rows by label (accounting synonyms expanded), optionally one statement."""
        scores = self._bm25.scores(expand_query(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[LineItem] = []
        for i in order:
            if scores[i] <= 0 or len(out) >= k:
                break
            if statement is None or statement in self._items[i].tags:
                out.append(self._items[i])
        return out

    def sqlite(self) -> sqlite3.Connection:
        """In-memory database with one row per (line item, column), built once and cached."""
        if self._conn is None:
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            conn.execute(
                "CREATE TABLE line_items (doc_id TEXT, page INTEGER, row INTEGER, label TEXT, "
                "col INTEGER, period TEXT, value REAL, unit TEXT, statement TEXT)"
            )
            conn.executemany(
                "INSERT INTO line_items VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        it.doc_id,
                        it.page,
                        it.row,
                        it.label,
                        c,
                        it.periods[c],
                        v,
                        it.unit,
                        it.tags[0] if it.tags else "",
                    )
                    for it in self._items
                    for c, v in enumerate(it.values)
                ],
            )
            self._conn = conn
        return self._conn
