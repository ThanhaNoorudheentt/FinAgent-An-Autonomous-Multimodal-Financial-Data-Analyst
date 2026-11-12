"""Command-line entry point.

Data:        download, download-pdfs
Evaluation:  eval-finqa, eval-financebench, eval-retrieval, compare
Use:         ask   (answer a question about any filing PDF and print an analyst report)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from finagent.agent.agent import AgentConfig, FinAgent
from finagent.config import Settings, get_settings
from finagent.data import DATASETS, download, download_financebench_pdfs
from finagent.evaluation.datasets import load_financebench, load_finqa
from finagent.evaluation.financebench import evidence_metrics, make_judge_scorer
from finagent.evaluation.finqa import score_finqa
from finagent.evaluation.harness import EvalSummary, run_evaluation, summarize
from finagent.evaluation.retrieval import RetrievalRow, breakdown, evaluate_retrieval
from finagent.evaluation.splits import split_examples
from finagent.ingestion.render import render_page
from finagent.llm.base import Image, LLMClient, QuotaExhaustedError
from finagent.llm.gemini import GeminiClient
from finagent.pipeline import SYSTEM_VARIANT, VARIANTS, FilingBuilder, make_config
from finagent.reporting.report import render_report
from finagent.schemas import FinanceBenchExample, FinQAExample
from finagent.systems.agent_system import AgentFinanceBenchSystem
from finagent.systems.direct import DirectFinQASystem
from finagent.systems.program import ProgramFinQASystem
from finagent.systems.rag import RagFinanceBenchSystem

SPLITS = ["all", "dev", "test"]
RETRIEVERS = ["bm25", "dense", "hybrid"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finagent")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download benchmark data")
    dl.add_argument("names", nargs="*", default=["finqa-test", "financebench"], choices=DATASETS)
    dl.add_argument("--force", action="store_true")

    pdfs = sub.add_parser("download-pdfs", help="download FinanceBench filings")
    pdfs.add_argument("--limit", type=int, default=150, help="cover the first N questions")

    fq = sub.add_parser("eval-finqa", help="FinQA with gold context: direct LLM vs calculator tool")
    fq.add_argument("--system", default="direct", choices=["direct", "program"])
    fq.add_argument("--split", default="test", choices=["train", "dev", "test"])
    fq.add_argument("--limit", type=int, default=20)
    fq.add_argument("--run-name", default=None)

    fb = sub.add_parser("eval-financebench", help="FinanceBench: rag | mmrag | agent")
    fb.add_argument("--system", default="rag", choices=sorted(SYSTEM_VARIANT))
    fb.add_argument("--retriever", default="hybrid", choices=RETRIEVERS)
    fb.add_argument("--k", type=int, default=None, help="chunks to retrieve (default: config)")
    fb.add_argument("--max-steps", type=int, default=AgentConfig().max_steps)
    fb.add_argument("--limit", type=int, default=None)
    fb.add_argument("--split", default="all", choices=SPLITS, help="dev for tuning, test for final")
    fb.add_argument("--run-name", default=None)
    fb.add_argument("--judge-model", default=None, help="model for grading (default: same)")

    rt = sub.add_parser("eval-retrieval", help="retrieval recall only (no generation calls)")
    rt.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    rt.add_argument("--retriever", default="bm25", choices=RETRIEVERS)
    rt.add_argument("--limit", type=int, default=150)
    rt.add_argument("--split", default="all", choices=SPLITS)
    rt.add_argument(
        "--ks", type=lambda v: [int(x) for x in v.split(",")], default=[1, 3, 5, 10, 20]
    )
    rt.add_argument("--breakdown-k", type=int, default=10)
    rt.add_argument("--out", type=Path, default=None, help="write per-question rows (JSONL)")

    ask = sub.add_parser("ask", help="answer a question about a filing PDF and print a report")
    ask.add_argument("--pdf", type=Path, required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--system", default="agent", choices=sorted(SYSTEM_VARIANT))
    ask.add_argument("--retriever", default="bm25", choices=RETRIEVERS)
    ask.add_argument("--report-out", type=Path, default=None, help="also save the report here")

    cmp_ = sub.add_parser("compare", help="side-by-side summary of evaluation run files")
    cmp_.add_argument("runs", nargs="+", type=Path)
    return parser


# ---------------------------------------------------------------------------------- helpers
def _print_summary(summary: EvalSummary, out: Path) -> None:
    extras = " ".join(f"{k}={v:.3f}" for k, v in summary.mean_metrics.items())
    print(
        f"n={summary.n} accuracy={summary.accuracy:.3f} errors={summary.n_errors} "
        f"mean_latency={summary.mean_latency_s:.2f}s {extras} -> {out}"
    )


def _print_breakdowns(rows: list[RetrievalRow], k: int) -> None:
    for title, key in (
        ("question_type", lambda r: r.question_type),
        ("question_reasoning", lambda r: r.question_reasoning),
    ):
        print(f"\nrecall@{k} by {title} (groups with n>=5):")
        for group, (count, recall) in breakdown(rows, key, k).items():
            if count >= 5:
                print(f"  {recall:.3f}  n={count:<3} {group[:80]}")


def _judge_client(args: argparse.Namespace, settings: Settings, answerer: LLMClient) -> LLMClient:
    if args.judge_model is None:
        return answerer
    return GeminiClient(settings.model_copy(update={"llm_model": args.judge_model}))


# ---------------------------------------------------------------------------------- commands
def _cmd_download_pdfs(args: argparse.Namespace, settings: Settings) -> int:
    path = download("financebench", settings.data_dir)
    examples = load_financebench(path, limit=args.limit)
    paths = download_financebench_pdfs({e.doc_name for e in examples}, settings.pdf_dir)
    print(f"{len(paths)} filings in {settings.pdf_dir}")
    return 0


def _cmd_eval_finqa(args: argparse.Namespace, settings: Settings) -> int:
    path = download(f"finqa-{args.split}", settings.data_dir)
    examples: list[FinQAExample] = load_finqa(path, limit=args.limit)
    client = GeminiClient(settings)
    system = DirectFinQASystem(client) if args.system == "direct" else ProgramFinQASystem(client)
    out = (
        settings.runs_dir / f"finqa-{args.split}-{args.run_name or args.system + '-baseline'}.jsonl"
    )
    summary = run_evaluation(
        examples,
        answer_fn=system.answer,
        score_fn=score_finqa,
        id_fn=lambda ex: ex.id,
        out_path=out,
    )
    _print_summary(summary, out)
    return 0


def _cmd_eval_financebench(args: argparse.Namespace, settings: Settings) -> int:
    path = download("financebench", settings.data_dir)
    bench: list[FinanceBenchExample] = split_examples(
        load_financebench(path), args.split, key=lambda e: e.financebench_id
    )[: args.limit]
    client = GeminiClient(settings)
    config = make_config(SYSTEM_VARIANT[args.system], args.retriever)
    builder = FilingBuilder(settings, client, config)
    k = args.k or settings.retrieval_k
    if args.system == "agent":
        system: Any = AgentFinanceBenchSystem(
            FinAgent(client, AgentConfig(max_steps=args.max_steps)), builder
        )
        default_name = f"agent-{args.retriever}"
    else:
        images = builder.page_images if args.system == "mmrag" else None
        system = RagFinanceBenchSystem(client, builder.retriever_for, k=k, page_images=images)
        default_name = f"{args.system}-{args.retriever}-k{k}"
    suffix = "" if args.split == "all" else f"-{args.split}"
    out = settings.runs_dir / f"financebench-{args.run_name or default_name}{suffix}.jsonl"
    summary = run_evaluation(
        bench,
        answer_fn=system.answer,
        score_fn=make_judge_scorer(
            _judge_client(args, settings, client), log_path=out.with_suffix(".judge.jsonl")
        ),
        id_fn=lambda ex: ex.financebench_id,
        out_path=out,
        metrics_fn=evidence_metrics,
    )
    _print_summary(summary, out)
    return 0


def _cmd_eval_retrieval(args: argparse.Namespace, settings: Settings) -> int:
    path = download("financebench", settings.data_dir)
    bench = split_examples(load_financebench(path), args.split, key=lambda e: e.financebench_id)
    client = GeminiClient(settings) if args.retriever != "bm25" else None
    builder = FilingBuilder(settings, client, make_config(args.variant, args.retriever))
    report = evaluate_retrieval(bench[: args.limit], builder.retriever_for, ks=args.ks)
    recalls = "  ".join(f"recall@{k}={v:.3f}" for k, v in report.recall_at_k.items())
    print(
        f"{args.variant}/{args.retriever}: n={report.n} skipped={report.n_skipped} "
        f"{recalls} mrr={report.mrr:.3f}"
    )
    _print_breakdowns(report.rows, args.breakdown_k if args.breakdown_k in args.ks else args.ks[0])
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for row in report.rows:
                fh.write(json.dumps(dataclasses.asdict(row), ensure_ascii=False) + "\n")
        print(f"per-question rows -> {args.out}")
    return 0


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    client = GeminiClient(settings)
    builder = FilingBuilder(
        settings, client, make_config(SYSTEM_VARIANT[args.system], args.retriever)
    )
    doc_id = args.pdf.stem
    filing = builder.from_pdf(args.pdf, doc_id)
    if args.system == "agent":
        answer = FinAgent(client).answer(args.question, filing)
    else:

        def images(_doc: str, pages: list[int]) -> list[Image]:
            return [render_page(args.pdf, p) for p in pages]

        rag = RagFinanceBenchSystem(
            client,
            lambda _doc: filing.retriever,
            k=settings.retrieval_k,
            page_images=images if args.system == "mmrag" else None,
        )
        answer = rag.ask(args.question, doc_id)
    report = render_report(args.question, answer, doc_id=doc_id)
    print(report)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report, encoding="utf-8")
    return 0


def _cmd_compare(args: argparse.Namespace, _settings: Settings) -> int:
    print(f"{'run':44} {'n':>4} {'acc':>6} {'err':>4} {'ev.rec':>7} {'calls':>6} {'lat(s)':>7}")
    for path in args.runs:
        s = summarize(path)
        m = s.mean_metrics
        recall = f"{m['evidence_recall']:.3f}" if "evidence_recall" in m else "-"
        calls = f"{m['llm_calls']:.1f}" if "llm_calls" in m else "-"
        print(
            f"{path.stem[:44]:44} {s.n:>4} {s.accuracy:>6.3f} {s.n_errors:>4} "
            f"{recall:>7} {calls:>6} {s.mean_latency_s:>7.1f}"
        )
    return 0


def _cmd_download(args: argparse.Namespace, settings: Settings) -> int:
    for name in args.names:
        print(download(name, settings.data_dir, force=args.force))
    return 0


_COMMANDS = {
    "download": _cmd_download,
    "download-pdfs": _cmd_download_pdfs,
    "eval-finqa": _cmd_eval_finqa,
    "eval-financebench": _cmd_eval_financebench,
    "eval-retrieval": _cmd_eval_retrieval,
    "ask": _cmd_ask,
    "compare": _cmd_compare,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return _COMMANDS[args.command](args, get_settings())
    except QuotaExhaustedError as exc:
        print(f"\nStopped: {exc}\nProgress is saved; re-run the same command to resume.")
        return 3
