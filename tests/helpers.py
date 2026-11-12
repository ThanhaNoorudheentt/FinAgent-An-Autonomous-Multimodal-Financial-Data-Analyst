"""Shared test fixtures: a synthetic one-page cash-flow statement PDF and scripted fake LLMs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymupdf

from finagent.config import Settings
from finagent.filing import FilingContext
from finagent.pipeline import FilingBuilder, PipelineConfig

DOC = "ACME_2018_10K"
ROWS: list[tuple[str, list[str]]] = [
    ("Net income", ["5,363", "4,869", "5,058"]),
    ("Depreciation and amortization", ["1,488", "1,544", "1,474"]),
    ("Accounts receivable", ["(305)", "(245)", "(313)"]),
    ("Inventories", ["(509)", "(387)", "57"]),
    ("Net cash provided by operating activities", ["6,439", "6,240", "6,662"]),
    ("Purchases of property, plant and equipment (PP&E)", ["(1,577)", "(1,373)", "(1,420)"]),
    ("Proceeds from sale of PP&E", ["262", "49", "58"]),
    ("Net cash used in investing activities", ["(1,828)", "(2,152)", "(1,410)"]),
    ("Dividends paid to shareholders", ["(3,193)", "(2,803)", "(2,678)"]),
    ("Net increase in cash and cash equivalents", ["(200)", "655", "600"]),
]


def make_statement_pdf(path: Path, *, extra_prose_page: bool = True) -> None:
    """Page 0: prose. Page 1: a cash-flow statement laid out in columns."""
    doc = pymupdf.open()
    if extra_prose_page:
        doc.new_page().insert_text((72, 72), "Management discussion of liquidity and results.")
    page = doc.new_page()
    page.insert_text((72, 50), "Consolidated Statement of Cash Flows", fontsize=11)
    page.insert_text((72, 70), "(Millions)", fontsize=9)
    for x, year in ((340, "2018"), (410, "2017"), (480, "2016")):
        page.insert_text((x, 70), year, fontsize=9)
    for i, (label, values) in enumerate(ROWS):
        y = 95 + 14 * i
        page.insert_text((72, y), label, fontsize=9)
        for x, value in zip((340, 410, 480), values, strict=True):
            page.insert_text((x, y), value, fontsize=9)
    doc.save(path)
    doc.close()


def make_filing(tmp_path: Path, *, retriever: str = "bm25") -> FilingContext:
    pdf = tmp_path / f"{DOC}.pdf"
    make_statement_pdf(pdf)
    settings = Settings(pdf_dir=tmp_path, index_dir=tmp_path / "idx", _env_file=None)  # type: ignore[call-arg]
    config = PipelineConfig(retriever=retriever)  # type: ignore[arg-type]
    return FilingBuilder(settings, None, config)(DOC)


class Script:
    """Scripted LLM: answers the planning prompt, then pops agent steps in order."""

    def __init__(
        self,
        steps: list[dict[str, Any] | str],
        plan: dict[str, Any] | str | None = None,
        judge: Callable[[str], str] | None = None,
    ) -> None:
        self.steps = list(steps)
        self.plan = (
            plan
            if plan is not None
            else {
                "approach": "find capex",
                "search_queries": ["purchases of property plant and equipment"],
                "line_items": ["Purchases of property, plant and equipment"],
                "statements": ["cash_flow"],
            }
        )
        self.judge = judge

    def __call__(self, prompt: str) -> str:
        if "GOLD ANSWER" in prompt and self.judge is not None:
            return self.judge(prompt)
        if "Plan how to answer" in prompt:
            return self.plan if isinstance(self.plan, str) else json.dumps(self.plan)
        step = self.steps.pop(0)
        return step if isinstance(step, str) else json.dumps(step)
