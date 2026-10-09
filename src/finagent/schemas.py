"""Data contracts shared across the whole system.

Every answer FinAgent produces carries its evidence and calculations in structured form,
so that "evidence-grounded" is something we can test rather than just claim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EvidenceKind = Literal["text", "table", "chart"]


class Evidence(BaseModel):
    """A piece of source material supporting an answer."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    page: int | None = None
    kind: EvidenceKind = "text"
    content: str


class Calculation(BaseModel):
    """A computation that was executed by a tool (never by the LLM's own arithmetic)."""

    description: str
    code: str
    result: float | str
    inputs: list[Evidence] = Field(default_factory=list)


class Answer(BaseModel):
    """The final, structured output of any QA system in this project."""

    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


# --------------------------------------------------------------------------------------
# Benchmark example schemas
# --------------------------------------------------------------------------------------


class FinQAExample(BaseModel):
    """One FinQA question with its gold context (text around a table, plus the table)."""

    id: str
    question: str
    answer: str
    exe_ans: float | str | None = None
    program: str | None = None
    pre_text: list[str]
    post_text: list[str]
    table: list[list[str]]

    def context_as_text(self) -> str:
        """Render pre-text, table, and post-text as a single prompt-ready string."""
        rows = [" | ".join(row) for row in self.table]
        return "\n".join(
            [
                "TEXT BEFORE TABLE:",
                *self.pre_text,
                "",
                "TABLE:",
                *rows,
                "",
                "TEXT AFTER TABLE:",
                *self.post_text,
            ]
        )


class FinanceBenchEvidence(BaseModel):
    """A gold evidence span from a FinanceBench filing."""

    evidence_text: str
    doc_name: str
    evidence_page_num: int


class FinanceBenchExample(BaseModel):
    """One FinanceBench open-book question about a public-company filing."""

    financebench_id: str
    company: str
    doc_name: str
    question_type: str
    question_reasoning: str | None = None
    question: str
    answer: str
    justification: str | None = None
    evidence: list[FinanceBenchEvidence]
