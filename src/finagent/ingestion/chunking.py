"""Page-bounded, line-aware chunking with overlap."""

from __future__ import annotations

from collections.abc import Iterable

from finagent.schemas import Chunk, Page


def chunk_page(page: Page, *, max_chars: int = 1500, overlap: int = 200) -> list[Chunk]:
    """Split a page into chunks of at most ``max_chars`` characters (plus one newline per line).

    Chunks break on line boundaries so numbers and labels are not cut in half, consecutive
    chunks share up to ``overlap`` characters of trailing lines, and no chunk crosses a page,
    so every retrieved passage maps to exactly one citable page.
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    pieces: list[str] = []
    for raw in page.text.splitlines():
        line = raw.strip()
        while len(line) > max_chars:  # pathological line: hard split with overlap
            pieces.append(line[:max_chars])
            line = line[max_chars - overlap :]
        if line:
            pieces.append(line)

    texts: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        if current and size + len(piece) + 1 > max_chars:
            texts.append("\n".join(current))
            current, size = _overlap_tail(current, overlap)
        current.append(piece)
        size += len(piece) + 1
    if current:
        texts.append("\n".join(current))

    return [
        Chunk(
            chunk_id=f"{page.doc_id}:p{page.page}:c{i}",
            doc_id=page.doc_id,
            page=page.page,
            text=text,
        )
        for i, text in enumerate(texts)
    ]


def _overlap_tail(lines: list[str], overlap: int) -> tuple[list[str], int]:
    keep: list[str] = []
    size = 0
    for line in reversed(lines):
        if size + len(line) + 1 > overlap:
            break
        keep.insert(0, line)
        size += len(line) + 1
    return keep, size


def chunk_pages(pages: Iterable[Page], *, max_chars: int = 1500, overlap: int = 200) -> list[Chunk]:
    return [c for page in pages for c in chunk_page(page, max_chars=max_chars, overlap=overlap)]
