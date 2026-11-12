"""Structured outputs the agent asks the LLM for."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Statement = Literal["balance_sheet", "income_statement", "cash_flow"]
Action = Literal["search_filing", "find_line_items", "calculate", "sql", "final_answer"]


class QueryPlan(BaseModel):
    """The agent's plan: what to look for, in the filing's own vocabulary."""

    approach: str
    search_queries: list[str] = Field(default_factory=list)
    line_items: list[str] = Field(default_factory=list)
    statements: list[Statement] = Field(default_factory=list)


class AgentStep(BaseModel):
    """One decision. Fields not used by the chosen action are null."""

    thought: str
    action: Action
    query: str | None = None  # search_filing, find_line_items
    statement: Statement | None = None  # find_line_items (optional filter)
    expression: str | None = None  # calculate
    sql: str | None = None  # sql
    answer: str | None = None  # final_answer
    confidence: float | None = None  # final_answer, 0..1
