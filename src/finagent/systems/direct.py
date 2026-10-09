"""Baseline 0: give the LLM the gold FinQA context and let it reason and compute itself.

This is deliberately naive. It measures how far an LLM gets when it does its own arithmetic,
which is the number the tool-using agent (Stage 3+) must beat.
"""

from __future__ import annotations

from pydantic import BaseModel

from finagent.llm.base import LLMClient, generate_model
from finagent.schemas import Answer, Evidence, FinQAExample

SYSTEM = (
    "You are a careful financial analyst. Answer using ONLY the provided report excerpt. "
    "Show brief step-by-step reasoning, then give the final answer."
)

PROMPT = """{context}

QUESTION: {question}

Respond as JSON with two fields:
- "reasoning": the steps, including the arithmetic
- "answer": ONLY the final value (a number, with % if the question asks for a percentage,
  or "yes"/"no"). No units, no explanation."""


class _Reply(BaseModel):
    reasoning: str
    answer: str


class DirectFinQASystem:
    """FinQA baseline with gold context and no tools."""

    name = "direct-gold-context"

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def answer(self, example: FinQAExample) -> Answer:
        prompt = PROMPT.format(context=example.context_as_text(), question=example.question)
        reply = generate_model(self._client, prompt, _Reply, system=SYSTEM)
        return Answer(
            answer=reply.answer.strip(),
            evidence=[Evidence(doc_id=example.id, kind="table", content=example.context_as_text())],
        )
