from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from finagent import cli
from finagent.llm.base import QuotaExhaustedError
from finagent.llm.fake import FakeClient
from tests.helpers import DOC, Script, make_statement_pdf

QUESTION = "What was FY2018 capital expenditure, per the cash flow statement?"
FIND = {"thought": "look up", "action": "find_line_items", "query": "capex"}
CALC = {"thought": "sign", "action": "calculate", "expression": "abs(-1577)"}
FINAL = {"thought": "done", "action": "final_answer", "answer": "$1,577 million", "confidence": 0.9}
VERDICT = '{"correct": true, "reason": "same number"}'
RAG_REPLY = json.dumps({"reasoning": "r", "answer": "$1,577 million"})


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated workspace with one filing PDF and a one-question FinanceBench file."""
    (tmp_path / "pdfs").mkdir()
    make_statement_pdf(tmp_path / "pdfs" / f"{DOC}.pdf")
    row = {
        "financebench_id": "fb_1",
        "company": "ACME",
        "doc_name": DOC,
        "question_type": "metrics-generated",
        "question": QUESTION,
        "answer": "$1,577 million",
        "evidence": [{"evidence_text": "x", "doc_name": DOC, "evidence_page_num": 1}],
    }
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "financebench.jsonl").write_text(json.dumps(row) + "\n")
    for name, sub in (
        ("DATA_DIR", "data"),
        ("PDF_DIR", "pdfs"),
        ("RUNS_DIR", "runs"),
        ("INDEX_DIR", "idx"),
        ("CACHE_DIR", "cache"),
    ):
        monkeypatch.setenv(f"FINAGENT_{name}", str(tmp_path / sub))
    monkeypatch.chdir(tmp_path)  # no stray .env is picked up
    return tmp_path


def use_fake(monkeypatch: pytest.MonkeyPatch, responder: Any) -> FakeClient:
    client = FakeClient(responder)
    monkeypatch.setattr(cli, "GeminiClient", lambda *_a, **_k: client)
    return client


def test_ask_agent_prints_and_saves_a_report(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake(monkeypatch, Script([FIND, CALC, FINAL]))
    out = env / "report.md"
    code = cli.main(
        [
            "ask",
            "--pdf",
            str(env / "pdfs" / f"{DOC}.pdf"),
            "--question",
            QUESTION,
            "--report-out",
            str(out),
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "# FinAgent analyst report" in printed
    assert "$1,577 million" in printed
    assert "`abs(-1577)`" in printed
    assert out.read_text().strip() == printed.strip()


@pytest.mark.parametrize("system", ["rag", "mmrag"])
def test_ask_rag_systems(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], system: str
) -> None:
    client = use_fake(monkeypatch, lambda _p: RAG_REPLY)
    code = cli.main(
        [
            "ask",
            "--pdf",
            str(env / "pdfs" / f"{DOC}.pdf"),
            "--question",
            QUESTION,
            "--system",
            system,
        ]
    )
    assert code == 0
    assert "$1,577 million" in capsys.readouterr().out
    assert ("attached" in client.calls[0]) == (system == "mmrag")


@pytest.mark.parametrize(("system", "calls"), [("agent", 4.0), ("rag", 1.0), ("mmrag", 1.0)])
def test_eval_financebench_all_three_systems(
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    system: str,
    calls: float,
) -> None:
    script = Script([FIND, CALC, FINAL], judge=lambda _p: VERDICT)
    use_fake(
        monkeypatch,
        script if system == "agent" else (lambda p: VERDICT if "GOLD ANSWER" in p else RAG_REPLY),
    )
    code = cli.main(["eval-financebench", "--system", system, "--retriever", "bm25"])
    summary = capsys.readouterr().out
    assert code == 0
    assert "n=1 accuracy=1.000 errors=0" in summary
    assert "evidence_recall=1.000" in summary
    assert f"llm_calls={calls:.3f}" in summary
    assert list((env / "runs").glob(f"financebench-{system}-bm25*.jsonl"))


def test_eval_retrieval_full_pipeline_ranks_the_statement_page_first(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = env / "rows.jsonl"
    code = cli.main(["eval-retrieval", "--variant", "full", "--ks", "1,3", "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 0
    assert "full/bm25: n=1 skipped=0 recall@1=1.000" in printed
    assert json.loads(out.read_text())["first_gold_rank"] == 1


def test_compare_prints_one_line_per_run(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake(monkeypatch, lambda p: VERDICT if "GOLD ANSWER" in p else RAG_REPLY)
    cli.main(["eval-financebench", "--system", "rag", "--retriever", "bm25"])
    capsys.readouterr()
    run = next((env / "runs").glob("financebench-rag-bm25*.jsonl"))
    assert cli.main(["compare", str(run)]) == 0
    line = capsys.readouterr().out.splitlines()[-1]
    assert run.stem[:44] in line
    assert "1.000" in line


def test_quota_exhaustion_exits_cleanly_with_code_3(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(_prompt: str) -> str:
        raise QuotaExhaustedError("Daily API quota exhausted for test-model.")

    use_fake(monkeypatch, boom)
    code = cli.main(["eval-financebench", "--system", "agent", "--retriever", "bm25"])
    assert code == 3
    printed = capsys.readouterr().out
    assert "Stopped: Daily API quota exhausted" in printed
    assert "re-run the same command" in printed


def test_eval_finqa_program_system(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    finqa = [
        {
            "id": "X/1",
            "pre_text": ["a"],
            "post_text": ["b"],
            "table": [["", "v"], ["r", "5"]],
            "qa": {"question": "change?", "answer": "1.6%", "exe_ans": 0.01639, "program": "p"},
        }
    ]
    (env / "data" / "finqa_test.json").write_text(json.dumps(finqa))
    reply = {"reasoning": "r", "expression": "pct_change(5735, 5829)", "answer_format": "percent"}
    use_fake(monkeypatch, lambda _p: json.dumps(reply))
    assert cli.main(["eval-finqa", "--system", "program", "--limit", "1"]) == 0
    assert "n=1 accuracy=1.000" in capsys.readouterr().out


def test_parser_exposes_all_commands() -> None:
    parser = cli._build_parser()
    for command in (
        "download",
        "download-pdfs",
        "eval-finqa",
        "eval-financebench",
        "eval-retrieval",
        "ask",
        "compare",
    ):
        assert command in parser._subparsers._group_actions[0].choices  # type: ignore[union-attr]
