from __future__ import annotations

from finagent.evaluation.judge import judge_answer
from finagent.llm.fake import FakeClient


def test_judge_parses_verdict_and_prompt_contains_inputs() -> None:
    client = FakeClient('{"correct": true, "reason": "same number"}')
    verdict = judge_answer(client, "capex?", "$1,577 million", "1.577 billion dollars")
    assert verdict.correct is True
    assert "$1,577 million" in client.calls[0]
    assert "1.577 billion dollars" in client.calls[0]
