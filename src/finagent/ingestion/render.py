"""Render PDF pages to images for multimodal prompting (charts, scanned or complex tables)."""

from __future__ import annotations

from pathlib import Path

from finagent.llm.base import Image


def render_page(pdf_path: Path, page: int, *, dpi: int = 110) -> Image:
    """Render one 0-based page as a PNG. 110 dpi keeps statement text legible at modest cost."""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:  # type: ignore[no-untyped-call]
        pixmap = doc[page].get_pixmap(dpi=dpi)
        return Image(data=pixmap.tobytes("png"), mime_type="image/png")
