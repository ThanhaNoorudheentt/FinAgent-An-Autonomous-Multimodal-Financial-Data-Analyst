"""Scoring utilities: numeric answer matching and evidence recall."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from finagent.schemas import Evidence, FinanceBenchEvidence

_NUMBER_RE = re.compile(r"(\()?\s*\$?\s*([-\u2212\u2013])?\s*\$?\s*(\d[\d,]*(?:\.\d+)?|\.\d+)")
_PUNCT_RE = re.compile(r"[^\w\s]")


@dataclass(frozen=True)
class ParsedNumber:
    value: float
    is_percent: bool


def parse_number(text: str) -> ParsedNumber | None:
    """Extract the first number from ``text``.

    Handles ``$1,234.5``, ``(3.2)`` (negative), ``-5``, and ``14.1%``. Scale words such as
    "million" are deliberately ignored: the question fixes the unit, and unit mismatches are
    better caught by the LLM judge than by guessing here.
    """
    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    open_paren, sign, digits = match.group(1), match.group(2), match.group(3)
    value = float(digits.replace(",", ""))
    rest = text[match.end() :].lstrip()
    is_percent = rest.startswith("%") or rest.lower().startswith("percent")
    after_pct = rest.lstrip("%").lstrip()
    if sign or (open_paren and after_pct.startswith(")")):
        value = -value
    return ParsedNumber(value=value, is_percent=is_percent)


def _candidates(p: ParsedNumber) -> list[float]:
    return [p.value, p.value / 100.0] if p.is_percent else [p.value]


def normalize_text(text: str) -> str:
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def answers_match(
    predicted: str, gold: float | str, *, rel_tol: float = 1e-2, abs_tol: float = 1e-6
) -> bool:
    """True if ``predicted`` equals ``gold`` numerically (within tolerance) or textually.

    Percent signs make both ``14.1%`` and ``0.141`` acceptable for a gold of 0.141.
    Non-numeric golds (e.g. "yes"/"no") are compared as normalized text.
    """
    pred_num = parse_number(predicted)
    if isinstance(gold, (int, float)):
        gold_num: ParsedNumber | None = ParsedNumber(float(gold), False)
    else:
        gold_num = parse_number(gold)

    if pred_num is not None and gold_num is not None:
        return any(
            math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
            for a in _candidates(pred_num)
            for b in _candidates(gold_num)
        )
    if isinstance(gold, str):
        return normalize_text(predicted) == normalize_text(gold)
    return False


def evidence_recall(
    retrieved: Iterable[Evidence],
    gold: Iterable[FinanceBenchEvidence],
    *,
    page_offset: int = 0,
) -> float:
    """Fraction of gold (document, page) pairs that appear among the retrieved evidence.

    ``page_offset`` is added to gold page numbers. Check whether your parser numbers pages
    from 0 or 1 and set it accordingly (FinanceBench page numbers should be verified against
    the PDFs once you have ingestion running).
    """
    gold_pairs = {(g.doc_name, g.evidence_page_num + page_offset) for g in gold}
    if not gold_pairs:
        return 1.0
    got = {(e.doc_id, e.page) for e in retrieved if e.page is not None}
    return len(gold_pairs & got) / len(gold_pairs)
