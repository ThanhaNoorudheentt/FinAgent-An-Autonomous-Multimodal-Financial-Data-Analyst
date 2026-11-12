"""Page-bounded, line-aware chunking with overlap."""

from __future__ import annotations

from collections.abc import Iterable

from finagent.ingestion.layout import CELL_SEPARATOR
from finagent.ingestion.numbers import YEAR_RE
from finagent.schemas import Chunk, Page

_MAX_HEADER_LINES = 8


def chunk_page(page: Page, *, max_chars: int = 1500, overlap: int = 200) -> list[Chunk]:
    """Split a page into chunks of at most ``max_chars`` characters (plus one newline per line).

    Chunks break on line boundaries so numbers and labels are not cut in half, consecutive
    chunks share up to ``overlap`` characters of trailing lines, and no chunk crosses a page,
    so every retrieved passage maps to exactly one citable page.

    For table pages every chunk after the first repeats the table header (title, units and the
    year row), because a row such as ``Net income | 5,363 | 4,869`` is meaningless without
    knowing which year each column is.
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    lines = _split_lines(page.text, max_chars, overlap)
    groups = _pack(lines, max_chars, overlap)
    header = _table_header(lines) if page.tabular else []

    chunks: list[Chunk] = []
    for i, group in enumerate(groups):
        body = group if i == 0 or not header else [*header, *[g for g in group if g not in header]]
        chunks.append(
            Chunk(
                chunk_id=f"{page.doc_id}:p{page.page}:c{i}",
                doc_id=page.doc_id,
                page=page.page,
                text="\n".join(body),
                kind="table" if page.tabular else "text",
                tags=page.tags,
            )
        )
    return chunks


def _split_lines(text: str, max_chars: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        while len(line) > max_chars:  # pathological line: hard split with overlap
            pieces.append(line[:max_chars])
            line = line[max_chars - overlap :]
        if line:
            pieces.append(line)
    return pieces


def _pack(pieces: list[str], max_chars: int, overlap: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        if current and size + len(piece) + 1 > max_chars:
            groups.append(current)
            current, size = _overlap_tail(current, overlap)
        current.append(piece)
        size += len(piece) + 1
    if current:
        groups.append(current)
    return groups


def _table_header(lines: list[str]) -> list[str]:
    """Lines up to and including the first row that holds two or more year cells."""
    for i, line in enumerate(lines[:_MAX_HEADER_LINES]):
        cells = [c.strip() for c in line.split(CELL_SEPARATOR)]
        if sum(1 for c in cells if YEAR_RE.match(c)) >= 2:
            return lines[: i + 1]
    return []


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
