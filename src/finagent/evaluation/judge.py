"""LLM-as-judge for free-form answers (FinanceBench answers are sentences, not numbers)."""

from __future__ import annotations

from pydantic import BaseModel

from finagent.llm.base import LLMClient, generate_model

JUDGE_SYSTEM = (
    "You are a strict but fair grader of answers to financial questions. "
    "Compare the PREDICTED answer with the GOLD answer."
)

JUDGE_PROMPT = """QUESTION:
{question}

GOLD ANSWER:
{gold}

PREDICTED ANSWER:
{predicted}

Grade the predicted answer:
- correct = true if it states the same key facts and numbers as the gold answer. Allow
  equivalent formatting, units (e.g. $1,577 million vs 1.577 billion) and sensible rounding.
- correct = false if any key number or conclusion differs, is missing, or the answer refuses
  or hedges without committing.
Give a one-sentence reason."""


class Verdict(BaseModel):
    correct: bool
    reason: str


def judge_answer(client: LLMClient, question: str, gold: str, predicted: str) -> Verdict:
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, predicted=predicted)
    return generate_model(client, prompt, Verdict, system=JUDGE_SYSTEM)
