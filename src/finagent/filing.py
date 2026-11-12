"""Everything the system knows about one filing, built once and shared by all components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from finagent.ingestion.line_items import LineItemTable
from finagent.retrieval.retrievers import Retriever
from finagent.schemas import Chunk


@dataclass(frozen=True)
class FilingContext:
    doc_id: str
    pdf_path: Path
    chunks: list[Chunk]
    retriever: Retriever
    line_items: LineItemTable
