"""Deterministic answer verification: is every number in the answer traceable to the filing?

This is a grounding check, not a correctness proof. A number counts as grounded when it appears
(allowing the rounding the answer itself shows, and unit scaling) in text the agent retrieved or in
a tool-computed result. Mental arithmetic that the calculator never saw is flagged, which pushes
the agent to use tools.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_NUMBER_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_UNIT_SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6)  # thousands / millions / billions of the same figure
_PERCENT_SCALE = 0.01  # "14.5%" may be backed by a computed fraction 0.145


@dataclass(frozen=True)
class Verification:
    checked: int
    ungrounded: tuple[float, ...]

    @property
    def grounded(self) -> bool:
        return not self.ungrounded


@dataclass(frozen=True)
class _Number:
    value: float
    decimals: int  # digits after the decimal point as written
    percent: bool  # followed by a percent sign
    separated: bool = False  # written with thousands separators, so a quantity and not a year


def _scan(text: str) -> list[_Number]:
    found: list[_Number] = []
    for match in _NUMBER_RE.finditer(text):
        clean = match.group().replace(",", "").rstrip(".")
        try:
            value = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        after = text[match.end() :].lstrip()
        percent = after.startswith("%") or after.lower().startswith("percent")
        found.append(_Number(value, decimals, percent, separated="," in match.group()))
    return found


def extract_numbers(text: str) -> list[float]:
    """All unsigned numbers in ``text`` (thousands separators removed)."""
    return [n.value for n in _scan(text)]


def _needs_check(number: _Number) -> bool:
    whole = number.value == int(number.value)
    is_year = whole and not number.separated and 1900 <= number.value <= 2100
    is_small_count = whole and number.value <= 10
    return not (is_year or is_small_count)


def _is_grounded(number: _Number, pool: list[float]) -> bool:
    """True if some source number equals ``number`` up to the rounding the answer itself shows.

    The tolerance is half a unit of the answer's last printed digit, so "1,900" must match
    exactly while "8.7" (billions) may stand for 8,738 (millions).
    """
    half_unit = 0.5 * 10.0 ** (-number.decimals)
    scales = (*_UNIT_SCALES, _PERCENT_SCALE) if number.percent else _UNIT_SCALES
    return any(
        abs(number.value * scale - source) <= half_unit * scale * 1.0001
        for source in pool
        for scale in scales
    )


def verify_answer(answer: str, sources: Iterable[str], computed: Iterable[float]) -> Verification:
    """Check each number in ``answer`` against numbers in ``sources`` and ``computed`` results.

    A grounding check, not a correctness proof: with few significant digits a number can match
    by coincidence. Mental arithmetic that the calculator never saw is flagged, which pushes
    the agent to use tools.
    """
    pool = [abs(v) for v in computed]
    for text in sources:
        pool.extend(extract_numbers(text))
    to_check = [n for n in _scan(answer) if _needs_check(n)]
    missing = tuple(n.value for n in to_check if not _is_grounded(n, pool))
    return Verification(checked=len(to_check), ungrounded=missing)


def confidence_score(
    model_confidence: float | None, verification: Verification, *, abstained: bool, tool_errors: int
) -> float:
    """A heuristic confidence indicator in [0, 1] (not a calibrated probability).

    Starts from the model's own estimate (0.5 if absent), halves it when any number in the answer
    is ungrounded, and trims it slightly per failed tool call. Abstentions score 0.
    """
    if abstained:
        return 0.0
    score = 0.5 if model_confidence is None else min(max(model_confidence, 0.0), 1.0)
    if not verification.grounded:
        score *= 0.5
    score *= 0.95**tool_errors
    return round(score, 3)
