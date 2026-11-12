"""Render an :class:`Answer` as an evidence-grounded analyst report (Markdown)."""

from __future__ import annotations

from finagent.calc_format import format_calculation_result
from finagent.schemas import Answer

_SNIPPET_CHARS = 220


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _SNIPPET_CHARS else flat[: _SNIPPET_CHARS - 1] + "\u2026"


def render_report(question: str, answer: Answer, *, doc_id: str) -> str:
    """Question, answer, confidence, calculations, sources and warnings in one document."""
    lines = [
        "# FinAgent analyst report",
        "",
        f"**Filing:** {doc_id}  ",
        f"**Question:** {question}",
        "",
        "## Answer",
        "",
        answer.answer,
        "",
    ]
    if answer.confidence is not None:
        lines += [
            f"**Confidence:** {answer.confidence:.0%} (heuristic indicator, not calibrated)",
            "",
        ]

    if answer.calculations:
        lines += ["## Calculations", "", "| # | What | Expression | Result |", "|---|---|---|---|"]
        for i, calc in enumerate(answer.calculations, start=1):
            what = calc.description.replace("|", "/") or "-"
            result = format_calculation_result(calc.result)
            lines.append(f"| {i} | {what} | `{calc.code}` | {result} |")
        lines.append("")

    if answer.evidence:
        lines += ["## Evidence", ""]
        ranked = sorted(answer.evidence, key=lambda e: (e.page is None, e.page or 0))
        for e in ranked[:12]:
            page = f"page {e.page}" if e.page is not None else "page ?"
            lines.append(f"- **{e.doc_id}, {page}** ({e.kind}): {_snippet(e.content)}")
        if len(ranked) > 12:
            lines.append(f"- ... and {len(ranked) - 12} more passages")
        lines.append("")

    if answer.notes:
        lines += ["## Verification notes", ""] + [f"- {n}" for n in answer.notes] + [""]
    return "\n".join(lines)
