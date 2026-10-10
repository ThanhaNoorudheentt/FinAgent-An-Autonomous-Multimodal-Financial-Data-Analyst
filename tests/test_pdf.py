from __future__ import annotations

from pathlib import Path

import pymupdf

from finagent.ingestion.pdf import extract_pages


def make_pdf(path: Path, texts: list[str]) -> None:
    doc = pymupdf.open()
    for text in texts:
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_extract_pages_is_zero_based_and_keeps_text(tmp_path: Path) -> None:
    pdf = tmp_path / "ACME_2023_10K.pdf"
    make_pdf(pdf, ["Net sales 1,577", "Total assets 9,000"])
    pages = extract_pages(pdf)
    assert [p.page for p in pages] == [0, 1]
    assert "Net sales 1,577" in pages[0].text
    assert "Total assets" in pages[1].text
    assert pages[0].doc_id == "ACME_2023_10K"  # defaults to the file stem


def test_explicit_doc_id(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    make_pdf(pdf, ["hello"])
    assert extract_pages(pdf, "custom")[0].doc_id == "custom"
