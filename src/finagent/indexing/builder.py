"""Turn a PDF into chunks and (optionally) a persisted embedding index."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from finagent.indexing.vector_index import VectorIndex
from finagent.ingestion.chunking import chunk_pages
from finagent.ingestion.pdf import extract_pages
from finagent.llm.base import LLMClient
from finagent.llm.cache import make_key
from finagent.schemas import Chunk

logger = logging.getLogger(__name__)


def prepare_chunks(pdf_path: Path, doc_id: str, *, max_chars: int, overlap: int) -> list[Chunk]:
    """Parse and chunk a PDF. Pure CPU work: no API calls."""
    return chunk_pages(extract_pages(pdf_path, doc_id), max_chars=max_chars, overlap=overlap)


def load_or_build_index(
    pdf_path: Path,
    doc_id: str,
    client: LLMClient,
    *,
    index_dir: Path,
    embed_model: str,
    max_chars: int,
    overlap: int,
) -> VectorIndex:
    """Return the embedding index for a PDF, embedding it only if no matching index exists.

    The index folder name hashes the PDF bytes, embedding model, and chunking parameters,
    so changing any of them rebuilds automatically and stale indexes are never reused.
    """
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    key = make_key(pdf=digest, model=embed_model, max_chars=max_chars, overlap=overlap)
    directory = index_dir / f"{doc_id}-{key[:12]}"
    if (directory / "vectors.npy").exists():
        return VectorIndex.load(directory)

    chunks = prepare_chunks(pdf_path, doc_id, max_chars=max_chars, overlap=overlap)
    logger.info("Embedding %d chunks for %s", len(chunks), doc_id)
    index = VectorIndex(chunks, client.embed([c.text for c in chunks]))
    index.save(directory)
    return index
