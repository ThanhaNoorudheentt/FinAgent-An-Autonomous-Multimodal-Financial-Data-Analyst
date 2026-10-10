"""PDF text extraction (Stage 1: plain text; tables and charts arrive in Stage 2)."""

from __future__ import annotations

from pathlib import Path

from finagent.schemas import Page


def extract_pages(path: Path, doc_id: str | None = None) -> list[Page]:
    """Extract one :class:`Page` per PDF page, with 0-based page numbers."""
    import pymupdf

    doc_id = doc_id or path.stem
    with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
        return [
            Page(doc_id=doc_id, page=i, text=page.get_text("text")) for i, page in enumerate(doc)
        ]
