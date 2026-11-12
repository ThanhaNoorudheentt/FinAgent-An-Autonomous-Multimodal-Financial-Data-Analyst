"""A safe arithmetic evaluator for LLM-written expressions.

Models are unreliable at arithmetic, so the agent writes an expression and this module computes
it. It parses the expression into an AST and interprets only a small whitelist of node types, so
no code is ever executed: no attribute access, imports, comprehensions or arbitrary names.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping
from typing import Any

MAX_EXPRESSION_CHARS = 500
MAX_ABS_RESULT = 1e30
MAX_EXPONENT = 100


class CalculatorError(ValueError):
    """The expression is invalid, unsafe, or not computable (e.g. division by zero)."""


def pct_change(old: float, new: float) -> float:
    """Percentage change from ``old`` to ``new`` (25 means +25%). Uses |old| as the base."""
    if old == 0:
        raise CalculatorError("pct_change: base value is zero")
    return (new - old) / abs(old) * 100.0


def margin_pct(part: float, whole: float) -> float:
    """``part`` as a percentage of ``whole`` (e.g. operating income / net sales)."""
    if whole == 0:
        raise CalculatorError("margin_pct: denominator is zero")
    return part / whole * 100.0


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise CalculatorError("ratio: denominator is zero")
    return numerator / denominator


def cagr_pct(begin: float, end: float, years: float) -> float:
    """Compound annual growth rate as a percentage."""
    if begin <= 0 or end <= 0 or years <= 0:
        raise CalculatorError("cagr_pct needs positive begin, end and years")
    growth: float = (end / begin) ** (1.0 / years)
    return (growth - 1.0) * 100.0


def average(*values: float) -> float:
    if not values:
        raise CalculatorError("average of nothing")
    return sum(values) / len(values)


def _sum(*values: float) -> float:
    return sum(values)


def _round(value: float, digits: float = 0) -> float:
    """round() that accepts the float digit count a model's JSON number becomes."""
    if digits != int(digits):
        raise CalculatorError("round: digits must be a whole number")
    return round(value, int(digits))


FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": _round,
    "min": min,
    "max": max,
    "sum": _sum,
    "sqrt": math.sqrt,
    "pct_change": pct_change,
    "margin_pct": margin_pct,
    "ratio": ratio,
    "cagr_pct": cagr_pct,
    "average": average,
}

_BINARY: dict[type[ast.operator], Callable[[Any, Any], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def evaluate(expression: str, variables: Mapping[str, float] | None = None) -> float | bool:
    """Evaluate ``expression``. Returns a float, or a bool for comparisons such as ``a > b``."""
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculatorError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(
            f"syntax error: {exc.msg}. Write plain numbers without thousands separators."
        ) from exc
    result: float | bool = _eval(tree.body, variables or {})
    if isinstance(result, float) and (math.isnan(result) or abs(result) > MAX_ABS_RESULT):
        raise CalculatorError("result is out of range")
    return result


def _eval(node: ast.AST, variables: Mapping[str, float]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise CalculatorError("only numeric constants are allowed")
        return float(node.value)
    if isinstance(node, ast.Tuple):
        raise CalculatorError(
            "unexpected comma: write numbers without thousands separators (1577, not 1,577)"
        )
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise CalculatorError(f"unknown name {node.id!r}")
        return float(variables[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        value = _eval(node.operand, variables)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _binary(type(node.op), _eval(node.left, variables), _eval(node.right, variables))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, variables)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _eval(comparator, variables)
            if type(op) not in _COMPARE:
                raise CalculatorError("unsupported comparison")
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        function = FUNCTIONS.get(node.func.id)
        if function is None:
            raise CalculatorError(f"unknown function {node.func.id!r}")
        args = [_eval(a, variables) for a in node.args]
        try:
            return float(function(*args))
        except (TypeError, ValueError) as exc:
            raise CalculatorError(f"{node.func.id}: {exc}") from exc
    raise CalculatorError(f"unsupported syntax: {type(node).__name__}")


def _binary(op: type[ast.operator], left: float, right: float) -> float:
    if op is ast.Pow and (abs(right) > MAX_EXPONENT or abs(left) > 1e6):
        raise CalculatorError("exponent or base too large")
    try:
        return float(_BINARY[op](left, right))
    except ZeroDivisionError as exc:
        raise CalculatorError("division by zero") from exc
    except OverflowError as exc:
        raise CalculatorError("result is out of range") from exc


def format_result(value: float | bool) -> str:
    """Human- and model-friendly rendering: ``94``, ``14.4638``, ``True``."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value)}"
    return f"{value:.6g}" if abs(value) < 1e6 else f"{value:,.2f}"
