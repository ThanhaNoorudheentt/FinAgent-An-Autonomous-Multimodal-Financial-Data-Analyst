from __future__ import annotations

import json
from pathlib import Path

import pytest

from finagent.evaluation.datasets import load_financebench, load_finqa


def test_load_finqa(tmp_path: Path) -> None:
    raw = [
        {
            "id": "X/1",
            "pre_text": ["a"],
            "post_text": ["b"],
            "table": [["", "v"], ["r", 5]],
            "qa": {"question": "q?", "answer": "5", "exe_ans": 5.0, "program": "add(2, 3)"},
        }
    ]
    path = tmp_path / "f.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    [ex] = load_finqa(path)
    assert ex.id == "X/1"
    assert ex.table[1] == ["r", "5"]  # cells coerced to str
    assert "TABLE:" in ex.context_as_text()


def test_load_finqa_rejects_wrong_format(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps([{"id": "X"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="no 'qa'"):
        load_finqa(path)


def test_load_financebench_respects_limit(tmp_path: Path) -> None:
    row = {
        "financebench_id": "fb_1",
        "company": "3M",
        "doc_name": "3M_2018_10K",
        "question_type": "metrics-generated",
        "question": "q",
        "answer": "$1577.00",
        "extra_field": "ignored",
        "evidence": [{"evidence_text": "t", "doc_name": "3M_2018_10K", "evidence_page_num": 59}],
    }
    path = tmp_path / "fb.jsonl"
    path.write_text("\n".join(json.dumps({**row, "financebench_id": f"fb_{i}"}) for i in range(3)))
    assert len(load_financebench(path)) == 3
    assert len(load_financebench(path, limit=2)) == 2
