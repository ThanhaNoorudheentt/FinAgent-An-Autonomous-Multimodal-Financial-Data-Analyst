"""Builds a :class:`FilingContext` (chunks, retriever, line items) for a PDF, with caching."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from finagent.config import Settings
from finagent.filing import FilingContext
from finagent.indexing.builder import load_or_build_index
from finagent.ingestion.chunking import chunk_pages
from finagent.ingestion.line_items import LineItemTable, parse_line_items
from finagent.ingestion.pdf import ParseMode, extract_pages
from finagent.ingestion.render import render_page
from finagent.llm.base import EmbeddingClient, Image, LLMClient
from finagent.llm.local_embeddings import SentenceTransformerClient
from finagent.retrieval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retriever,
    StatementAwareRetriever,
)

RetrieverKind = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class PipelineConfig:
    retriever: RetrieverKind = "hybrid"
    parse_mode: ParseMode = "layout"
    statement_aware: bool = True
    expand_queries: bool = True


# Ingestion/retrieval variants, from the plain baseline to the full pipeline.
VARIANTS: dict[str, PipelineConfig] = {
    "plain": PipelineConfig(parse_mode="plain", statement_aware=False, expand_queries=False),
    "layout": PipelineConfig(parse_mode="layout", statement_aware=False, expand_queries=False),
    "layout+syn": PipelineConfig(parse_mode="layout", statement_aware=False, expand_queries=True),
    "full": PipelineConfig(),
}

# The three systems compared in the project: conventional, multimodal and agentic RAG.
SYSTEM_VARIANT = {"rag": "plain", "mmrag": "full", "agent": "full"}


def make_config(variant: str, retriever: RetrieverKind) -> PipelineConfig:
    return dataclasses.replace(VARIANTS[variant], retriever=retriever)


class FilingBuilder:
    """Parses, chunks, indexes and caches filings; call it with a doc name to get its context."""

    def __init__(
        self, settings: Settings, client: LLMClient | None, config: PipelineConfig
    ) -> None:
        self._settings = settings
        self._client = client
        self._config = config
        self._cache: dict[str, FilingContext] = {}
        self._local_embedding_client: EmbeddingClient | None = None

    def pdf_path(self, doc_name: str) -> Path:
        return self._settings.pdf_dir / f"{doc_name}.pdf"

    def __call__(self, doc_name: str) -> FilingContext:
        if doc_name not in self._cache:
            self._cache[doc_name] = self.from_pdf(self.pdf_path(doc_name), doc_name)
        return self._cache[doc_name]

    def retriever_for(self, doc_name: str) -> Retriever:
        return self(doc_name).retriever

    def page_images(self, doc_name: str, pages: list[int]) -> list[Image]:
        pdf = self.pdf_path(doc_name)
        return [render_page(pdf, p) for p in pages]

    def from_pdf(self, pdf_path: Path, doc_id: str) -> FilingContext:
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"{pdf_path} not found. Run `finagent download-pdfs` or place the PDF there."
            )
        cfg, s = self._config, self._settings
        pages = extract_pages(pdf_path, doc_id, mode=cfg.parse_mode)
        chunks = chunk_pages(pages, max_chars=s.chunk_max_chars, overlap=s.chunk_overlap_chars)

        retriever: Retriever = BM25Retriever(chunks, expand=cfg.expand_queries)
        if cfg.retriever != "bm25":
            embedding_client: EmbeddingClient | None
            embed_model_key: str
            if s.embedding_backend == "sentence-transformers":
                if self._local_embedding_client is None:
                    self._local_embedding_client = SentenceTransformerClient(s.local_embed_model)
                embedding_client = self._local_embedding_client
                # Include backend and model in the cache key so Gemini vectors are never reused.
                embed_model_key = f"sentence-transformers:{s.local_embed_model}"
            else:
                embedding_client = self._client
                embed_model_key = f"gemini:{s.embed_model}"
            if embedding_client is None:
                raise ValueError(
                    f"The {cfg.retriever!r} retriever needs an embedding client. "
                    "Configure Gemini or set FINAGENT_EMBEDDING_BACKEND=sentence-transformers."
                )
            index = load_or_build_index(
                pdf_path,
                doc_id,
                embedding_client,
                index_dir=s.index_dir,
                embed_model=embed_model_key,
                max_chars=s.chunk_max_chars,
                overlap=s.chunk_overlap_chars,
                parse_mode=cfg.parse_mode,
                chunks=chunks,
            )
            dense = DenseRetriever(index, embedding_client)
            retriever = dense if cfg.retriever == "dense" else HybridRetriever([retriever, dense])
        if cfg.statement_aware:
            retriever = StatementAwareRetriever(retriever, chunks)
        return FilingContext(
            doc_id=doc_id,
            pdf_path=pdf_path,
            chunks=chunks,
            retriever=retriever,
            line_items=LineItemTable(parse_line_items(pages)),
        )
