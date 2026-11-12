"""Read-only SQL over a filing's extracted line items."""

from __future__ import annotations

import sqlite3

MAX_ROWS = 40
_PROGRESS_STEP = 10_000
_MAX_PROGRESS_CALLS = 300  # aborts runaway queries (e.g. unbounded recursive CTEs)
_SQLITE_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)

SCHEMA_HELP = (
    "Table line_items(doc_id, page, row, label, col, period, value, unit, statement). "
    "One row per number in a statement row; period is the column's year header when known; "
    "statement is balance_sheet | income_statement | cash_flow or ''. "
    "Example: SELECT page, label, period, value FROM line_items WHERE label LIKE '%property%' "
    "AND statement = 'balance_sheet'"
)


class SqlToolError(ValueError):
    """The query was rejected or failed."""


def _authorizer(action: int, *_: object) -> int:
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        _SQLITE_RECURSIVE,
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def run_select(conn: sqlite3.Connection, query: str, *, max_rows: int = MAX_ROWS) -> str:
    """Run one SELECT statement and return a compact text table.

    Only reads are permitted (enforced by SQLite's authorizer, not by string matching), and runaway
    queries are aborted.
    """
    text = query.strip().rstrip(";").strip()
    if not text:
        raise SqlToolError("empty query")
    calls = {"n": 0}

    def progress() -> int:
        calls["n"] += 1
        return 1 if calls["n"] > _MAX_PROGRESS_CALLS else 0

    conn.set_authorizer(_authorizer)
    conn.set_progress_handler(progress, _PROGRESS_STEP)
    try:
        cursor = conn.execute(text)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
    except sqlite3.Error as exc:
        raise SqlToolError(str(exc)) from exc
    finally:
        conn.set_authorizer(None)
        conn.set_progress_handler(None, 0)

    if not rows:
        return "(no rows)"
    lines = [" | ".join(columns)]
    lines += [" | ".join("" if v is None else str(v) for v in row) for row in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f"... truncated to {max_rows} rows")
    return "\n".join(lines)
