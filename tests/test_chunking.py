from __future__ import annotations

from itertools import pairwise

import pytest

from finagent.ingestion.chunking import chunk_page, chunk_pages
from finagent.schemas import Page


def make_page(text: str, page: int = 3) -> Page:
    return Page(doc_id="DOC", page=page, text=text)


def test_short_page_is_one_chunk_with_exact_attribution() -> None:
    [chunk] = chunk_page(make_page("Revenue\n1,577\n"))
    assert chunk.text == "Revenue\n1,577"
    assert (chunk.doc_id, chunk.page, chunk.chunk_id) == ("DOC", 3, "DOC:p3:c0")


def test_empty_page_yields_no_chunks() -> None:
    assert chunk_page(make_page("  \n\n  ")) == []


def test_chunks_respect_max_size_and_never_split_lines() -> None:
    lines = [f"line {i:03d} with some padding text" for i in range(100)]
    chunks = chunk_page(make_page("\n".join(lines)), max_chars=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    rebuilt = {ln for c in chunks for ln in c.text.split("\n")}
    assert rebuilt == set(lines)  # no line lost or cut


def test_consecutive_chunks_overlap() -> None:
    lines = [f"row{i:03d}" for i in range(200)]
    chunks = chunk_page(make_page("\n".join(lines)), max_chars=200, overlap=50)
    for prev, nxt in pairwise(chunks):
        assert set(prev.text.split("\n")) & set(nxt.text.split("\n"))


def test_oversize_line_is_hard_split() -> None:
    chunks = chunk_page(make_page("x" * 1000), max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)


def test_overlap_must_be_smaller_than_max() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_page(make_page("a"), max_chars=100, overlap=100)


def test_chunk_pages_keeps_pages_separate() -> None:
    chunks = chunk_pages([make_page("a", 0), make_page("b", 1)])
    assert [(c.page, c.text) for c in chunks] == [(0, "a"), (1, "b")]
