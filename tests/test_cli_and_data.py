from __future__ import annotations

from pathlib import Path

import pytest

from finagent.cli import _build_parser, _score_finqa
from finagent.data import DATASETS, download
from finagent.schemas import Answer, FinQAExample


def test_unknown_dataset_name(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        download("nope", tmp_path)


def test_existing_file_is_not_redownloaded(tmp_path: Path) -> None:
    name = next(iter(DATASETS))
    dest = tmp_path / DATASETS[name][1]
    dest.write_text("cached")
    assert download(name, tmp_path) == dest
    assert dest.read_text() == "cached"


def test_score_uses_exe_ans_then_answer(finqa_example: FinQAExample) -> None:
    assert _score_finqa(finqa_example, Answer(answer="94"))
    assert not _score_finqa(finqa_example, Answer(answer="95"))
    no_exe = finqa_example.model_copy(update={"exe_ans": None, "answer": "14.1%"})
    assert _score_finqa(no_exe, Answer(answer="14.1%"))


def test_parser_defaults() -> None:
    args = _build_parser().parse_args(["eval-finqa"])
    assert args.limit == 20
    assert args.split == "test"


def test_score_accepts_rounded_or_exact(finqa_example: FinQAExample) -> None:
    ex = finqa_example.model_copy(update={"exe_ans": 0.14464, "answer": "14%"})
    assert _score_finqa(ex, Answer(answer="14%"))  # rounded like the gold string
    assert _score_finqa(ex, Answer(answer="14.5%"))  # more precise, matches exe_ans
    assert not _score_finqa(ex, Answer(answer="20%"))
