"""FinQA scoring."""

from __future__ import annotations

from finagent.evaluation.metrics import answers_match
from finagent.schemas import Answer, FinQAExample


def score_finqa(example: FinQAExample, answer: Answer) -> bool:
    """Correct if the prediction matches the exact value (``exe_ans``) or the rounded gold string.

    FinQA's written answer is rounded (e.g. "14%" for 0.14464), so a model that rounds the same
    way is right even though it misses the exact value; a model that reports more precision
    (14.5%) matches the exact value. Accepting either avoids penalising both behaviours.
    """
    golds: list[float | str] = []
    if example.exe_ans not in (None, ""):
        golds.append(example.exe_ans)
    if example.answer.strip():
        golds.append(example.answer)
    return any(answers_match(answer.answer, g) for g in golds)
