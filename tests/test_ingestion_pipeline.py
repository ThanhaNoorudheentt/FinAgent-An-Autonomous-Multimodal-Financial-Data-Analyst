from __future__ import annotations

from pathlib import Path

import pytest

from finagent.config import Settings
from finagent.indexing.builder import load_or_build_index
from finagent.ingestion.chunking import chunk_page
from finagent.ingestion.line_items import detect_unit, format_item, parse_line_items
from finagent.ingestion.pdf import extract_pages
from finagent.llm.fake import FakeClient
from finagent.pipeline import FilingBuilder, PipelineConfig, make_config
from finagent.retrieval.retrievers import StatementAwareRetriever
from finagent.schemas import Chunk, Page, ScoredChunk
from tests.helpers import DOC, make_filing, make_statement_pdf

CAPEX_ROW = "Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373) | (1,420)"


def test_layout_mode_rebuilds_rows_and_tags_the_statement(tmp_path: Path) -> None:
    pdf = tmp_path / f"{DOC}.pdf"
    make_statement_pdf(pdf)
    prose, table = extract_pages(pdf, mode="layout")
    assert not prose.tabular
    assert table.tabular
    assert table.tags == ("cash_flow",)
    assert CAPEX_ROW in table.text.splitlines()


def test_plain_mode_does_not_tag_or_rebuild(tmp_path: Path) -> None:
    pdf = tmp_path / f"{DOC}.pdf"
    make_statement_pdf(pdf)
    pages = extract_pages(pdf, mode="plain")
    assert not any(p.tabular or p.tags for p in pages)


def test_table_chunks_repeat_the_year_header_and_carry_tags() -> None:
    rows = [f"Row {i:02d} | {i}00 | {i}01" for i in range(30)]
    text = "\n".join(["Consolidated Balance Sheet", "(Millions) | 2018 | 2017", *rows])
    page = Page(doc_id="D", page=3, text=text, tabular=True, tags=("balance_sheet",))
    chunks = chunk_page(page, max_chars=300, overlap=60)
    assert len(chunks) > 2
    assert all(c.kind == "table" and c.tags == ("balance_sheet",) for c in chunks)
    for later in chunks[1:]:
        assert later.text.startswith("Consolidated Balance Sheet\n(Millions) | 2018 | 2017")


def test_parse_line_items_pairs_values_with_year_header_and_unit() -> None:
    text = "\n".join(
        [
            "Consolidated Statement of Cash Flows",
            "(Millions) | 2018 | 2017 | 2016",
            CAPEX_ROW,
            "Proceeds from sale of businesses, net of cash sold",
            "846 | 1,065 | 142",
            "Cash Flows from Financing Activities",
            "Dividends | (3,193) | (2,803)",
        ]
    )
    page = Page(doc_id="D", page=5, text=text, tabular=True, tags=("cash_flow",))
    items = parse_line_items([page])
    by_label = {i.label: i for i in items}
    capex = by_label["Purchases of property, plant and equipment (PP&E)"]
    assert capex.values == (-1577.0, -1373.0, -1420.0)
    assert capex.periods == ("2018", "2017", "2016")
    assert capex.unit == "millions"
    assert by_label["Proceeds from sale of businesses, net of cash sold"].values == (846, 1065, 142)
    assert by_label["Dividends"].values == (-3193.0, -2803.0)
    assert "Cash Flows from Financing Activities" not in by_label  # heading, not a data row
    assert format_item(capex).startswith("p5 [cash_flow] Purchases of property")


def test_non_tabular_pages_yield_no_line_items() -> None:
    assert parse_line_items([Page(doc_id="D", page=0, text="Net income | 5 | 6")]) == []


def test_detect_unit() -> None:
    assert detect_unit("(Millions) 2018") == "millions"
    assert detect_unit("Amounts in thousands, except per share") == "thousands"
    assert detect_unit("no unit here") == ""


def test_line_item_search_uses_synonyms_and_statement_filter(tmp_path: Path) -> None:
    filing = make_filing(tmp_path)
    table = filing.line_items
    top = table.search("What was capex?", k=1)[0]
    assert top.label.startswith("Purchases of property")
    assert table.search("capex", statement="balance_sheet") == []
    conn = table.sqlite()
    assert conn.execute("SELECT COUNT(*) FROM line_items").fetchone() == (30,)  # 10 rows x 3 years
    assert table.sqlite() is conn  # built once


def test_statement_aware_retriever_prioritises_the_named_statement() -> None:
    stmt = Chunk(
        chunk_id="D:p9:c0",
        doc_id="D",
        page=9,
        text="purchases cash",
        kind="table",
        tags=("cash_flow",),
    )
    other = Chunk(chunk_id="D:p1:c0", doc_id="D", page=1, text="capital spending prose")

    class Base:
        def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
            return [ScoredChunk(chunk=other, score=1.0)][:k]

    retriever = StatementAwareRetriever(Base(), [stmt, other])
    named = retriever.retrieve("capex shown in the cash flow statement", 2)
    assert [h.chunk.page for h in named] == [9, 1]
    plain = retriever.retrieve("capex in general", 2)
    assert [h.chunk.page for h in plain] == [1]  # no statement named: base order unchanged


def test_filing_builder_end_to_end_offline(tmp_path: Path) -> None:
    filing = make_filing(tmp_path)
    assert filing.doc_id == DOC
    hits = filing.retriever.retrieve("What is capex per the cash flow statement?", 3)
    assert hits[0].chunk.kind == "table"
    assert hits[0].chunk.page == 1


def test_filing_builder_errors_and_images(tmp_path: Path) -> None:
    settings = Settings(pdf_dir=tmp_path, index_dir=tmp_path / "i", _env_file=None)  # type: ignore[call-arg]
    with pytest.raises(FileNotFoundError, match="download-pdfs"):
        FilingBuilder(settings, None, PipelineConfig(retriever="bm25"))("NOPE")
    make_statement_pdf(tmp_path / f"{DOC}.pdf")
    with pytest.raises(ValueError, match="needs an embedding client"):
        FilingBuilder(settings, None, PipelineConfig(retriever="dense"))(DOC)
    images = FilingBuilder(settings, None, PipelineConfig(retriever="bm25")).page_images(DOC, [1])
    assert images[0].mime_type == "image/png"
    assert images[0].data.startswith(b"\x89PNG")


def test_make_config_variants() -> None:
    plain = make_config("plain", "bm25")
    assert (plain.parse_mode, plain.statement_aware, plain.expand_queries) == (
        "plain",
        False,
        False,
    )
    full = make_config("full", "hybrid")
    assert (full.parse_mode, full.statement_aware, full.retriever) == ("layout", True, "hybrid")


def test_dense_retriever_pipeline_with_fake_embeddings(tmp_path: Path) -> None:
    make_statement_pdf(tmp_path / f"{DOC}.pdf")
    settings = Settings(pdf_dir=tmp_path, index_dir=tmp_path / "i", _env_file=None)  # type: ignore[call-arg]
    filing = FilingBuilder(settings, FakeClient(), PipelineConfig(retriever="hybrid"))(DOC)
    assert filing.retriever.retrieve("capex cash flow statement", 2)


def test_changing_parse_mode_rebuilds_the_index(tmp_path: Path) -> None:
    pdf = tmp_path / f"{DOC}.pdf"
    make_statement_pdf(pdf)
    client = FakeClient()
    kwargs = {"index_dir": tmp_path / "i", "embed_model": "m", "max_chars": 1500, "overlap": 200}
    load_or_build_index(pdf, DOC, client, parse_mode="plain", **kwargs)  # type: ignore[arg-type]
    load_or_build_index(pdf, DOC, client, parse_mode="layout", **kwargs)  # type: ignore[arg-type]
    assert len(list((tmp_path / "i").iterdir())) == 2
