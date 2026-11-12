"""FinQA with a calculator tool: the LLM writes the expression, a tool computes it.

Compare with :class:`DirectFinQASystem` (the LLM does its own arithmetic) to measure what the tool
adds. This is the program-of-thought idea (Chen et al., 2022; Gao et al., 2022).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from finagent.llm.base import LLMClient, generate_model
from finagent.schemas import Answer, Calculation, Evidence, FinQAExample
from finagent.tools.calculator import CalculatorError, evaluate, format_result

SYSTEM = (
    "You are a careful financial analyst. Use ONLY the provided report excerpt. "
    "You never compute in your head: you write one arithmetic expression for a calculator."
)

PROMPT = """{context}

QUESTION: {question}

Respond as JSON with fields:
- "reasoning": which numbers you use and why
- "expression": ONE arithmetic expression whose value is the final answer, using plain numbers
  from the excerpt (no thousands separators; negative for numbers in parentheses). Functions
  available: pct_change(old,new), margin_pct(part,whole), ratio(a,b), average(...), abs, round.
  A comparison such as a > b is allowed for yes/no questions.
- "answer_format": "percent" if the answer is a percentage (the expression should then give the
  percentage value, e.g. 14.46 for 14.46%), "yes_no" for yes/no questions, else "number"
- "fallback_answer": your best direct answer, used only if the expression cannot be evaluated"""


class _Reply(BaseModel):
    reasoning: str
    expression: str | None = None
    answer_format: Literal["number", "percent", "yes_no"] = "number"
    fallback_answer: str | None = None


class ProgramFinQASystem:
    """FinQA baseline 2: gold context plus a calculator tool."""

    name = "program-of-thought"

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def answer(self, example: FinQAExample) -> Answer:
        prompt = PROMPT.format(context=example.context_as_text(), question=example.question)
        reply = generate_model(self._client, prompt, _Reply, system=SYSTEM)
        evidence = [Evidence(doc_id=example.id, kind="table", content=example.context_as_text())]
        notes: list[str] = []
        calculations: list[Calculation] = []
        text = (reply.fallback_answer or "").strip()
        if reply.expression:
            try:
                value = evaluate(reply.expression)
            except CalculatorError as exc:
                notes.append(f"Calculator error: {exc}; used the model's direct answer.")
            else:
                text = _format(value, reply.answer_format)
                calculations.append(
                    Calculation(description=reply.reasoning, code=reply.expression, result=text)
                )
        return Answer(
            answer=text, evidence=evidence, calculations=calculations, notes=notes, llm_calls=1
        )


def _format(value: float | bool, answer_format: str) -> str:
    if isinstance(value, bool):
        return format_result(value)
    if answer_format == "percent":
        return f"{value:.2f}%"
    if answer_format == "yes_no":
        return "yes" if value else "no"
    return format_result(value)
