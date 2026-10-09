"""Command-line entry point: ``finagent download`` and ``finagent eval-finqa``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from finagent.config import get_settings
from finagent.data import DATASETS, download
from finagent.evaluation.datasets import load_finqa
from finagent.evaluation.harness import run_evaluation
from finagent.evaluation.metrics import answers_match
from finagent.llm.gemini import GeminiClient
from finagent.schemas import Answer, FinQAExample
from finagent.systems.direct import DirectFinQASystem


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finagent")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download benchmark data")
    dl.add_argument("names", nargs="*", default=["finqa-test", "financebench"], choices=DATASETS)
    dl.add_argument("--force", action="store_true")

    ev = sub.add_parser("eval-finqa", help="evaluate the direct baseline on FinQA")
    ev.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ev.add_argument("--limit", type=int, default=20, help="number of questions (default 20)")
    ev.add_argument("--run-name", default="direct-baseline")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()

    if args.command == "download":
        for name in args.names:
            print(download(name, settings.data_dir, force=args.force))
        return 0

    if args.command == "eval-finqa":
        path = download(f"finqa-{args.split}", settings.data_dir)
        examples = load_finqa(path, limit=args.limit)
        system = DirectFinQASystem(GeminiClient(settings))
        out = settings.runs_dir / f"finqa-{args.split}-{args.run_name}.jsonl"
        summary = run_evaluation(
            examples,
            answer_fn=system.answer,
            score_fn=_score_finqa,
            id_fn=lambda ex: ex.id,
            out_path=out,
        )
        print(
            f"n={summary.n} accuracy={summary.accuracy:.3f} "
            f"errors={summary.n_errors} mean_latency={summary.mean_latency_s:.2f}s -> {out}"
        )
        return 0
    return 2
