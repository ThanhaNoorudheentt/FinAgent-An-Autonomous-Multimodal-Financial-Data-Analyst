"""Baseline 1: conventional text RAG over a single filing (FinanceBench "single document" setup)."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from finagent.llm.base import LLMClient, generate_model
from finagent.retrieval.retrievers import Retriever
from finagent.schemas import Answer, Evidence, FinanceBenchExample

SYSTEM = (
    "You are a careful financial analyst answering questions about a company's SEC filing. "
    "Use ONLY the excerpts provided. Never guess: if the excerpts do not contain what is needed, "
    "say so."
)

PROMPT = """EXCERPTS FROM {doc_name}:
{excerpts}

QUESTION: {question}

Instructions:
- Do any arithmetic step by step in "reasoning" (state the formula and the numbers used).
- Pay attention to units (thousands/millions) and the fiscal year the question asks about.
- "answer" must be a concise final answer including units. If the excerpts are insufficient,
  set "answer" to "Insufficient information in the retrieved excerpts."

Respond as JSON with fields "reasoning" and "answer"."""

RetrieverFactory = Callable[[str], Retriever]  # doc_name -> retriever for that filing


class _Reply(BaseModel):
    reasoning: str
    answer: str


class RagFinanceBenchSystem:
    """Retrieve top-k chunks from the question's filing, then answer with an LLM."""

    def __init__(
        self, client: LLMClient, retriever_factory: RetrieverFactory, *, k: int = 5
    ) -> None:
        self._client = client
        self._factory = retriever_factory
        self._k = k
        self.name = f"rag-k{k}"

    def answer(self, example: FinanceBenchExample) -> Answer:
        hits = self._factory(example.doc_name).retrieve(example.question, self._k)
        excerpts = "\n\n".join(
            f"[{i}] (page {h.chunk.page})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
        )
        prompt = PROMPT.format(
            doc_name=example.doc_name, excerpts=excerpts or "(none)", question=example.question
        )
        reply = generate_model(self._client, prompt, _Reply, system=SYSTEM)
        return Answer(
            answer=reply.answer.strip(),
            evidence=[
                Evidence(
                    doc_id=h.chunk.doc_id, page=h.chunk.page, kind="text", content=h.chunk.text
                )
                for h in hits
            ],
        )
