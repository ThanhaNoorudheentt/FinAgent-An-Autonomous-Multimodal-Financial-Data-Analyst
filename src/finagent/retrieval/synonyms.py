"""Accounting vocabulary bridges: questions say "capex", statements say "purchases of PP&E".

Deliberately small, generic accounting terminology (not tuned to any benchmark). Extend it as
you find vocabulary gaps in your own error analysis.
"""

from __future__ import annotations

import re

# trigger regex -> terms appended to the query when the trigger matches
_EXPANSIONS: tuple[tuple[str, str], ...] = (
    (
        r"\bcapex\b|capital expenditures?|capital spending",
        "purchases of property plant and equipment",
    ),
    (r"\bppne\b|\bppe\b|\bpp&e\b|net property", "property plant and equipment net"),
    (r"\brevenues?\b|\bsales\b", "net sales total revenues"),
    (r"\bcogs\b|cost of goods sold|cost of revenues?", "cost of sales"),
    (r"\bsg&a\b|\bsga\b", "selling general and administrative expenses"),
    (r"net income|net profit|net earnings", "net income net earnings"),
    (r"operating margin|operating income|\bebit\b", "operating income"),
    (r"\bd&a\b|depreciation", "depreciation and amortization"),
    (r"dividends?", "dividends paid to shareholders"),
    (
        r"buybacks?|share repurchases?|repurchase of (?:common )?stock",
        "purchases of treasury stock",
    ),
    (
        r"quick ratio|current ratio|liquidity",
        "total current assets total current liabilities inventories cash and cash equivalents",
    ),
    (
        r"capital[- ]intensive",
        "property plant and equipment net purchases of property plant and equipment net sales",
    ),
    (r"gross margin|gross profit", "gross profit net sales cost of sales"),
    (r"\bdebt\b|leverage|borrowings", "long-term debt short-term borrowings total debt"),
    (r"working capital", "total current assets total current liabilities"),
    (
        r"free cash flow|\bfcf\b",
        "net cash provided by operating activities purchases of property plant and equipment",
    ),
)
_COMPILED = tuple((re.compile(trigger, re.IGNORECASE), terms) for trigger, terms in _EXPANSIONS)


def expand_query(query: str) -> str:
    """Return ``query`` plus accounting synonyms for any trigger it contains."""
    extra = [terms for pattern, terms in _COMPILED if pattern.search(query)]
    return f"{query} {' '.join(extra)}" if extra else query
