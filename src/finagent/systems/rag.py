"""Baseline 1: conventional text RAG over a single filing (FinanceBench "single document" setup)."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from finagent.llm.base import Image, LLMClient, generate_model
from finagent.retrieval.retrievers import Retriever
from finagent.schemas import Answer, Evidence, FinanceBenchExample

SYSTEM = (
    "You are a careful financial analyst answering questions about a company's SEC filing. "
    "Use ONLY the excerpts provided. Never guess: if the excerpts do not contain what is needed, "
    "say so."
)

PROMPT = """EXCERPTS FROM {doc_name}:
{excerpts}
{image_note}
QUESTION: {question}

Instructions:
- Do any arithmetic step by step in "reasoning" (state the formula and the numbers used).
- Pay attention to units (thousands/millions) and the fiscal year the question asks about.
- "answer" must be a concise final answer including units. If the excerpts are insufficient,
  set "answer" to "Insufficient information in the retrieved excerpts."

Respond as JSON with fields "reasoning" and "answer"."""

RetrieverFactory = Callable[[str], Retriever]  # doc_name -> retriever for that filing
PageImageProvider = Callable[[str, list[int]], list[Image]]  # (doc_name, pages) -> page images


class _Reply(BaseModel):
    reasoning: str
    answer: str


class RagFinanceBenchSystem:
    """Retrieve top-k chunks from a filing, then answer with an LLM.

    With ``page_images`` the top ``n_images`` distinct retrieved pages are also attached as images,
    which lets the model read charts and complex tables that text extraction garbles
    (the "multimodal RAG" variant).
    """

    def __init__(
        self,
        client: LLMClient,
        retriever_factory: RetrieverFactory,
        *,
        k: int = 5,
        page_images: PageImageProvider | None = None,
        n_images: int = 2,
    ) -> None:
        self._client = client
        self._factory = retriever_factory
        self._k = k
        self._page_images = page_images
        self._n_images = n_images
        self.name = f"{'mmrag' if page_images else 'rag'}-k{k}"

    def answer(self, example: FinanceBenchExample) -> Answer:
        return self.ask(example.question, example.doc_name)

    def ask(self, question: str, doc_name: str) -> Answer:
        hits = self._factory(doc_name).retrieve(question, self._k)
        excerpts = "\n\n".join(
            f"[{i}] (page {h.chunk.page})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
        )
        images: list[Image] | None = None
        image_note = ""
        if self._page_images is not None and hits:
            pages = list(dict.fromkeys(h.chunk.page for h in hits))[: self._n_images]
            images = self._page_images(doc_name, pages)
            image_note = (
                f"\n(Images of pages {', '.join(map(str, pages))} are attached, in that order. "
                "Use them for charts and tables; prefer them where the text looks garbled.)\n"
            )
        prompt = PROMPT.format(
            doc_name=doc_name,
            excerpts=excerpts or "(none)",
            question=question,
            image_note=image_note,
        )
        reply = generate_model(self._client, prompt, _Reply, system=SYSTEM, images=images)
        return Answer(
            answer=reply.answer.strip(),
            evidence=[
                Evidence(
                    doc_id=h.chunk.doc_id,
                    page=h.chunk.page,
                    kind=h.chunk.kind,
                    content=h.chunk.text,
                )
                for h in hits
            ],
            llm_calls=1,
        )
