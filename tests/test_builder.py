from __future__ import annotations

from pathlib import Path

from finagent.indexing.builder import load_or_build_index, prepare_chunks
from finagent.llm.fake import FakeClient
from tests.test_pdf import make_pdf


class CountingClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.embedded_texts = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts += len(texts)
        return super().embed(texts)


KWARGS = {"embed_model": "m1", "max_chars": 1500, "overlap": 200}


def test_prepare_chunks_cites_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "D.pdf"
    make_pdf(pdf, ["page zero text", "page one text"])
    chunks = prepare_chunks(pdf, "D", max_chars=1500, overlap=200)
    assert [(c.page, "text" in c.text) for c in chunks] == [(0, True), (1, True)]


def test_index_is_built_once_then_loaded_from_disk(tmp_path: Path) -> None:
    pdf = tmp_path / "D.pdf"
    make_pdf(pdf, ["alpha", "beta"])
    client = CountingClient()
    first = load_or_build_index(pdf, "D", client, index_dir=tmp_path / "idx", **KWARGS)
    assert client.embedded_texts == 2
    second = load_or_build_index(pdf, "D", client, index_dir=tmp_path / "idx", **KWARGS)
    assert client.embedded_texts == 2  # no re-embedding
    assert second.chunks == first.chunks


def test_changing_embed_model_or_chunking_rebuilds(tmp_path: Path) -> None:
    pdf = tmp_path / "D.pdf"
    make_pdf(pdf, ["alpha"])
    client = CountingClient()
    load_or_build_index(pdf, "D", client, index_dir=tmp_path / "i", **KWARGS)
    load_or_build_index(
        pdf, "D", client, index_dir=tmp_path / "i", **{**KWARGS, "embed_model": "m2"}
    )
    load_or_build_index(pdf, "D", client, index_dir=tmp_path / "i", **{**KWARGS, "max_chars": 900})
    assert client.embedded_texts == 3
