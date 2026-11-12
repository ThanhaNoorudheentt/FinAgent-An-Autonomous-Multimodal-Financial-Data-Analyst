"""FinAgent: plan, gather evidence, compute with tools, verify, answer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import ValidationError

from finagent.agent import prompts
from finagent.agent.models import AgentStep, QueryPlan
from finagent.agent.verifier import confidence_score, extract_numbers, verify_answer
from finagent.filing import FilingContext
from finagent.ingestion.line_items import LineItem, format_item
from finagent.llm.base import LLMClient, generate_model
from finagent.retrieval.retrievers import rrf_fuse
from finagent.schemas import Answer, Calculation, Chunk, Evidence
from finagent.tools.calculator import CalculatorError, evaluate, format_result
from finagent.tools.sql import SCHEMA_HELP, SqlToolError, run_select

logger = logging.getLogger(__name__)

INSUFFICIENT = "Insufficient information in the filing."


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 6  # tool steps before the agent is forced to answer
    context_chunks: int = 6  # chunks retrieved before the first step
    context_line_items: int = 8
    search_results: int = 4
    max_observation_chars: int = 3500
    history_turns: int = 6


@dataclass
class _Turn:
    step: AgentStep
    observation: str


@dataclass
class _Run:
    """Mutable state for one question."""

    calls: int = 0
    tool_errors: int = 0
    history: list[_Turn] = field(default_factory=list)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    calculations: list[Calculation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class FinAgent:
    """Answers a question about one filing. LLM quota errors propagate to the caller."""

    def __init__(self, client: LLMClient, config: AgentConfig | None = None) -> None:
        self._client = client
        self._cfg = config or AgentConfig()

    # ------------------------------------------------------------------ public
    def answer(self, question: str, filing: FilingContext) -> Answer:
        run = _Run()
        plan = self._plan(question, filing, run)
        context = self._initial_context(question, plan, filing, run)

        final: AgentStep | None = None
        for _ in range(self._cfg.max_steps):
            step = self._next_step(question, plan, context, run, force=False)
            if step is None:
                continue
            if step.action == "final_answer":
                final = step
                break
            run.history.append(_Turn(step, self._execute(step, filing, run)))

        if final is None:
            forced = self._next_step(question, plan, context, run, force=True)
            if forced is not None and forced.action == "final_answer":
                final = forced
                run.notes.append("Step budget exhausted; answer was forced.")
        if final is None:
            final = AgentStep(
                thought="", action="final_answer", answer=INSUFFICIENT, confidence=0.0
            )
            run.notes.append("No valid final answer was produced.")
        return self._finish(final, context, run)

    # ------------------------------------------------------------------ steps
    def _plan(self, question: str, filing: FilingContext, run: _Run) -> QueryPlan:
        prompt = prompts.PLAN.format(doc_name=filing.doc_id, question=question)
        run.calls += 1
        try:
            plan = generate_model(self._client, prompt, QueryPlan, system=prompts.SYSTEM)
        except ValidationError:
            run.notes.append("Planner output was invalid; fell back to the raw question.")
            return QueryPlan(approach="", search_queries=[question])
        plan.search_queries = plan.search_queries[:4]
        plan.line_items = plan.line_items[:6]
        return plan

    def _initial_context(
        self, question: str, plan: QueryPlan, filing: FilingContext, run: _Run
    ) -> str:
        queries = list(dict.fromkeys([question, *plan.search_queries]))[:5]
        k = self._cfg.context_chunks
        hits = rrf_fuse([filing.retriever.retrieve(q, k) for q in queries], k=k)
        for hit in hits:
            self._remember_chunk(run, hit.chunk)

        items: list[LineItem] = []
        seen: set[tuple[int, int]] = set()
        for name in plan.line_items:
            for item in filing.line_items.search(name, k=2):
                if (item.page, item.row) not in seen and len(items) < self._cfg.context_line_items:
                    seen.add((item.page, item.row))
                    items.append(item)
        for item in items:
            self._remember_item(run, item)

        excerpts = "\n\n".join(
            f"[{i}] (page {h.chunk.page}, {h.chunk.kind})\n{h.chunk.text}"
            for i, h in enumerate(hits, start=1)
        )
        lines = "\n".join(format_item(i) for i in items) or "(none found)"
        return f"RETRIEVED EXCERPTS:\n{excerpts or '(none)'}\n\nCANDIDATE LINE ITEMS:\n{lines}"

    def _next_step(
        self, question: str, plan: QueryPlan, context: str, run: _Run, *, force: bool
    ) -> AgentStep | None:
        history = "\n".join(
            f"{n}. {t.step.action}({_args(t.step)}) -> {t.observation}"
            for n, t in enumerate(run.history[-self._cfg.history_turns :], start=1)
        )
        prompt = prompts.STEP.format(
            question=question,
            approach=plan.approach or "(none)",
            context=context,
            history=history or "(none yet)",
            sql_help=SCHEMA_HELP,
            force=prompts.FORCE_FINAL if force else "",
        )
        run.calls += 1
        try:
            return generate_model(self._client, prompt, AgentStep, system=prompts.SYSTEM)
        except ValidationError as exc:
            run.tool_errors += 1
            run.notes.append("A step had invalid JSON and was skipped.")
            logger.warning("Invalid agent step: %s", exc.errors()[:1])
            return None

    # ------------------------------------------------------------------ tools
    def _execute(self, step: AgentStep, filing: FilingContext, run: _Run) -> str:
        try:
            text = self._dispatch(step, filing, run)
        except (CalculatorError, SqlToolError) as exc:
            run.tool_errors += 1
            text = f"Error: {exc}"
        return text[: self._cfg.max_observation_chars]

    def _dispatch(self, step: AgentStep, filing: FilingContext, run: _Run) -> str:
        if step.action == "search_filing":
            query = _require(step.query, "search_filing needs 'query'")
            hits = filing.retriever.retrieve(query, self._cfg.search_results)
            for hit in hits:
                self._remember_chunk(run, hit.chunk)
            return (
                "\n\n".join(f"(page {h.chunk.page}) {h.chunk.text}" for h in hits) or "No results."
            )
        if step.action == "find_line_items":
            query = _require(step.query, "find_line_items needs 'query'")
            items = filing.line_items.search(query, k=8, statement=step.statement)
            for item in items:
                self._remember_item(run, item)
            return "\n".join(format_item(i) for i in items) or "No matching line items."
        if step.action == "calculate":
            expression = _require(step.expression, "calculate needs 'expression'")
            value = evaluate(expression)
            result: float | str = format_result(value) if isinstance(value, bool) else value
            run.calculations.append(
                Calculation(
                    description=step.thought,
                    code=expression,
                    result=result,
                    inputs=self._provenance(expression, run),
                )
            )
            return f"{expression} = {format_result(value)}"
        if step.action == "sql":
            query = _require(step.sql, "sql needs 'sql'")
            return run_select(filing.line_items.sqlite(), query)
        raise CalculatorError(f"unsupported action {step.action!r}")

    # ------------------------------------------------------------------ evidence
    @staticmethod
    def _remember_chunk(run: _Run, chunk: Chunk) -> None:
        run.evidence.setdefault(
            chunk.chunk_id,
            Evidence(doc_id=chunk.doc_id, page=chunk.page, kind=chunk.kind, content=chunk.text),
        )

    @staticmethod
    def _remember_item(run: _Run, item: LineItem) -> None:
        run.evidence.setdefault(
            f"{item.doc_id}:p{item.page}:r{item.row}",
            Evidence(doc_id=item.doc_id, page=item.page, kind="table", content=format_item(item)),
        )

    @staticmethod
    def _provenance(expression: str, run: _Run) -> list[Evidence]:
        """Evidence whose numbers appear in the expression: where the inputs came from."""
        used = set(extract_numbers(expression))
        return [e for e in run.evidence.values() if used & set(extract_numbers(e.content))][:6]

    # ------------------------------------------------------------------ result
    def _finish(self, final: AgentStep, context: str, run: _Run) -> Answer:
        text = (final.answer or INSUFFICIENT).strip()
        abstained = text.lower().startswith("insufficient")
        sources = [context, *(t.observation for t in run.history)]
        computed = [c.result for c in run.calculations if isinstance(c.result, float)]
        verification = verify_answer(text, sources, computed)
        if not verification.grounded:
            shown = ", ".join(f"{n:g}" for n in verification.ungrounded)
            run.notes.append(f"Numbers not traceable to the filing or a calculation: {shown}.")
        return Answer(
            answer=text,
            evidence=list(run.evidence.values()),
            calculations=run.calculations,
            confidence=confidence_score(
                final.confidence, verification, abstained=abstained, tool_errors=run.tool_errors
            ),
            notes=run.notes,
            llm_calls=run.calls,
        )


def _require(value: str | None, message: str) -> str:
    if not value or not value.strip():
        raise CalculatorError(message)
    return value


def _args(step: AgentStep) -> str:
    for value in (step.query, step.expression, step.sql):
        if value:
            return value
    return ""
