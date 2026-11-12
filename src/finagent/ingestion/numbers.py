"""Parsing of numbers as they appear in financial statements."""

from __future__ import annotations

import re

_DASHES = {"-", "\u2014", "\u2013", "\u2212", "--"}
_CELL_RE = re.compile(r"^\(?\s*[-\u2212]?\s*\$?\s*(\d[\d,]*(?:\.\d+)?|\.\d+)\s*%?\s*\)?\s*%?$")
YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def parse_financial_number(cell: str) -> float | None:
    """Parse one table cell: ``(1,577)`` -> -1577, ``5.2%`` -> 5.2, ``$ 5,363`` -> 5363.

    Returns None for anything that is not purely a number (labels, dashes, notes), so callers
    can tell value cells from label cells.
    """
    text = cell.strip()
    match = _CELL_RE.match(text)
    if match is None:
        return None
    value = float(match.group(1).replace(",", ""))
    negative = text.startswith("(") or text.lstrip("(").lstrip().startswith(("-", "\u2212"))
    return -value if negative else value


def is_dash(cell: str) -> bool:
    return cell.strip() in _DASHES
