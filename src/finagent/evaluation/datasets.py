"""Loaders that turn raw benchmark files into validated pydantic examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finagent.schemas import FinanceBenchExample, FinQAExample


def load_finqa(path: Path, *, limit: int | None = None) -> list[FinQAExample]:
    """Load FinQA (train/dev/test .json from the official repo)."""
    raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    examples: list[FinQAExample] = []
    for item in raw[:limit]:
        qa = item.get("qa")
        if qa is None:
            raise ValueError(f"Entry {item.get('id')!r} has no 'qa' field; is this FinQA?")
        examples.append(
            FinQAExample(
                id=item["id"],
                question=qa["question"],
                answer=str(qa["answer"]),
                exe_ans=qa.get("exe_ans"),
                program=qa.get("program"),
                pre_text=item["pre_text"],
                post_text=item["post_text"],
                table=[[str(cell) for cell in row] for row in item["table"]],
            )
        )
    return examples


def load_financebench(path: Path, *, limit: int | None = None) -> list[FinanceBenchExample]:
    """Load the open-source FinanceBench sample (.jsonl, 150 questions)."""
    examples: list[FinanceBenchExample] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            examples.append(FinanceBenchExample.model_validate(json.loads(line)))
            if limit is not None and len(examples) >= limit:
                break
    return examples
