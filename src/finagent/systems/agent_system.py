"""Adapter that runs :class:`FinAgent` on FinanceBench examples."""

from __future__ import annotations

from collections.abc import Callable

from finagent.agent.agent import FinAgent
from finagent.filing import FilingContext
from finagent.schemas import Answer, FinanceBenchExample


class AgentFinanceBenchSystem:
    name = "agent"

    def __init__(self, agent: FinAgent, filings: Callable[[str], FilingContext]) -> None:
        self._agent = agent
        self._filings = filings

    def answer(self, example: FinanceBenchExample) -> Answer:
        return self._agent.answer(example.question, self._filings(example.doc_name))
