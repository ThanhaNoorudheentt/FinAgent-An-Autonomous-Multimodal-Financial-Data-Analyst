"""Shared formatting for calculation results."""

from __future__ import annotations

from finagent.tools.calculator import format_result


def format_calculation_result(result: float | str) -> str:
    return result if isinstance(result, str) else format_result(result)
