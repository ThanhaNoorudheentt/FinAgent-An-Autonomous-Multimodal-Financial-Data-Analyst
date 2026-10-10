"""Command-line entry point.

``finagent download``            fetch benchmark data
``finagent download-pdfs``       fetch FinanceBench filings for the first N questions
``finagent eval-finqa``          Stage 0 baseline (gold context, no retrieval)
``finagent eval-financebench``   Stage 1 baseline (RAG over the filing PDF)
``finagent eval-retrieval``      retrieval recall only: no generation or judge calls
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from functools import cache
from pathlib import Path

from finagent.config import Settings, get_settings
from finagent.data import DATASETS, download, download_financebench_pdfs
from finagent.evaluation.datasets import load_financebench, load_finqa
from finagent.evaluation.financebench import evidence_metrics, make_judge_scorer
from finagent.evaluation.harness import EvalSummary, run_evaluation
from finagent.evaluation.metrics import answers_match
from finagent.evaluation.retrieval import evaluate_retrieval
from finagent.indexing.builder import load_or_build_index, prepare_chunks
from finagent.llm.base import LLMClient, QuotaExhaustedError
from finagent.llm.gemini import GeminiClient
from finagent.retrieval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retriever,
)
from finagent.schemas import Answer, FinQAExample
from finagent.systems.direct import DirectFinQASystem
from finagent.systems.rag import RagFinanceBenchSystem, RetrieverFactory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finagent")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download benchmark data")
    dl.add_argument("names", nargs="*", default=["finqa-test", "financebench"], choices=DATASETS)
    dl.add_argument("--force", action="store_true")

    pdfs = sub.add_parser("download-pdfs", help="download FinanceBench filings")
    pdfs.add_argument("--limit", type=int, default=20, help="cover the first N questions")

    fq = sub.add_parser("eval-finqa", help="evaluate the direct baseline on FinQA")
    fq.add_argument("--split", default="test", choices=["train", "dev", "test"])
    fq.add_argument("--limit", type=int, default=20, help="number of questions (default 20)")
    fq.add_argument("--run-name", default="direct-baseline")

    fb = sub.add_parser("eval-financebench", help="evaluate RAG on FinanceBench")
    fb.add_argument("--retriever", default="hybrid", choices=["bm25", "dense", "hybrid"])
    fb.add_argument("--k", type=int, default=None, help="chunks to retrieve (default: config)")
    fb.add_argument("--limit", type=int, default=20)
    fb.add_argument("--run-name", default=None, help="default: rag-<retriever>-k<k>")
    fb.add_argument("--judge-model", default=None, help="model for grading (default: same)")

    rt = sub.add_parser("eval-retrieval", help="measure retrieval recall (no generation calls)")
    rt.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "hybrid"])
    rt.add_argument("--limit", type=int, default=150)
    rt.add_argument(
        "--ks", type=lambda v: [int(x) for x in v.split(",")], default=[1, 3, 5, 10, 20]
    )
    return parser


def _score_finqa(example: FinQAExample, answer: Answer) -> bool:
    """Correct if the prediction matches the exact value (``exe_ans``) or the rounded gold string.

    FinQA's written answer is rounded (e.g. "14%" for 0.14464), so a model that rounds the same
    way is right even though it misses the exact value; a model that reports more precision
    (14.5%) matches the exact value. Accepting either avoids penalising both behaviours.
    """
    golds: list[float | str] = []
    if example.exe_ans not in (None, ""):
        golds.append(example.exe_ans)
    if example.answer.strip():
        golds.append(example.answer)
    return any(answers_match(answer.answer, g) for g in golds)


def _make_retriever_factory(
    kind: str, settings: Settings, client: LLMClient | None
) -> RetrieverFactory:
    """Build a per-filing retriever factory; parsing/indexing happens once per filing."""

    @cache
    def factory(doc_name: str) -> Retriever:
        pdf = settings.pdf_dir / f"{doc_name}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(
                f"{pdf} not found. Run `finagent download-pdfs` or place the PDF there."
            )
        size, overlap = settings.chunk_max_chars, settings.chunk_overlap_chars
        bm25 = BM25Retriever(prepare_chunks(pdf, doc_name, max_chars=size, overlap=overlap))
        if kind == "bm25":
            return bm25
        if client is None:
            raise ValueError(f"The {kind!r} retriever needs an LLM client for embeddings.")
        index = load_or_build_index(
            pdf,
            doc_name,
            client,
            index_dir=settings.index_dir,
            embed_model=settings.embed_model,
            max_chars=size,
            overlap=overlap,
        )
        dense = DenseRetriever(index, client)
        return dense if kind == "dense" else HybridRetriever([bm25, dense])

    return factory


def _print_summary(summary: EvalSummary, out: Path) -> None:
    extras = " ".join(f"{k}={v:.3f}" for k, v in summary.mean_metrics.items())
    print(
        f"n={summary.n} accuracy={summary.accuracy:.3f} errors={summary.n_errors} "
        f"mean_latency={summary.mean_latency_s:.2f}s {extras} -> {out}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return _dispatch(args, get_settings())
    except QuotaExhaustedError as exc:
        print(f"\nStopped: {exc}\nProgress is saved; re-run the same command to resume.")
        return 3


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "download":
        for name in args.names:
            print(download(name, settings.data_dir, force=args.force))
        return 0

    if args.command == "download-pdfs":
        path = download("financebench", settings.data_dir)
        examples = load_financebench(path, limit=args.limit)
        paths = download_financebench_pdfs({e.doc_name for e in examples}, settings.pdf_dir)
        print(f"{len(paths)} filings in {settings.pdf_dir}")
        return 0

    if args.command == "eval-finqa":
        path = download(f"finqa-{args.split}", settings.data_dir)
        finqa = load_finqa(path, limit=args.limit)
        direct = DirectFinQASystem(GeminiClient(settings))
        out = settings.runs_dir / f"finqa-{args.split}-{args.run_name}.jsonl"
        summary = run_evaluation(
            finqa,
            answer_fn=direct.answer,
            score_fn=_score_finqa,
            id_fn=lambda ex: ex.id,
            out_path=out,
        )
        _print_summary(summary, out)
        return 0

    if args.command == "eval-retrieval":
        path = download("financebench", settings.data_dir)
        bench = load_financebench(path, limit=args.limit)
        embed_client = GeminiClient(settings) if args.retriever != "bm25" else None
        report = evaluate_retrieval(
            bench, _make_retriever_factory(args.retriever, settings, embed_client), ks=args.ks
        )
        recalls = "  ".join(f"recall@{k}={v:.3f}" for k, v in report.recall_at_k.items())
        print(
            f"{args.retriever}: n={report.n} skipped={report.n_skipped} "
            f"{recalls} mrr={report.mrr:.3f}"
        )
        return 0

    if args.command == "eval-financebench":
        path = download("financebench", settings.data_dir)
        bench = load_financebench(path, limit=args.limit)
        client = GeminiClient(settings)
        judge_settings = (
            settings.model_copy(update={"llm_model": args.judge_model})
            if args.judge_model
            else settings
        )
        judge = client if args.judge_model is None else GeminiClient(judge_settings)
        k = args.k or settings.retrieval_k
        run_name = args.run_name or f"rag-{args.retriever}-k{k}"
        out = settings.runs_dir / f"financebench-{run_name}.jsonl"
        factory = _make_retriever_factory(args.retriever, settings, client)
        rag = RagFinanceBenchSystem(client, factory, k=k)
        summary = run_evaluation(
            bench,
            answer_fn=rag.answer,
            score_fn=make_judge_scorer(judge, log_path=out.with_suffix(".judge.jsonl")),
            id_fn=lambda ex: ex.financebench_id,
            out_path=out,
            metrics_fn=evidence_metrics,
        )
        _print_summary(summary, out)
        return 0
    return 2
