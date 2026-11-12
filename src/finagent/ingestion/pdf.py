"""PDF text extraction in two modes.

``plain``   one text stream per page (the conventional-RAG baseline).
``layout``  table-like pages are rebuilt row by row and tagged when they are a primary financial
            statement; prose pages keep plain text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from finagent.ingestion.layout import MIN_ROWS_FOR_TABLE, Word, numeric_row_count, words_to_rows
from finagent.ingestion.structure import classify_page
from finagent.schemas import Page

ParseMode = Literal["plain", "layout"]
PARSE_VERSION = "v2"  # bump when extraction output changes, so cached indexes rebuild


def extract_pages(
    path: Path, doc_id: str | None = None, *, mode: ParseMode = "plain"
) -> list[Page]:
    """Extract one :class:`Page` per PDF page, with 0-based page numbers."""
    import pymupdf

    doc_id = doc_id or path.stem
    pages: list[Page] = []
    with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
        for index, page in enumerate(doc):
            plain = page.get_text("text")
            if mode == "plain":
                pages.append(Page(doc_id=doc_id, page=index, text=plain))
                continue
            words = [Word(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]
            rows = words_to_rows(words)
            numeric = numeric_row_count(rows)
            if numeric >= MIN_ROWS_FOR_TABLE:
                pages.append(
                    Page(
                        doc_id=doc_id,
                        page=index,
                        text="\n".join(rows),
                        tabular=True,
                        tags=classify_page(rows, numeric),
                    )
                )
            else:
                pages.append(Page(doc_id=doc_id, page=index, text=plain))
    return pages
