# Design notes

## Principles

1. **Compute with tools, never in the LLM's head.** All arithmetic goes through a calculator that
   parses expressions into an AST and interprets a whitelist of node types. Nothing is `exec`'d.
2. **Every claim is traceable.** Chunks never cross pages; every answer carries `Evidence`
   (document, 0-based page, text) and `Calculation` (expression, result, inputs).
3. **Depend on interfaces.** Code uses `LLMClient`; Gemini is one implementation. Tests use a
   scripted `FakeClient`, so the whole pipeline is exercised offline.
4. **Measure before believing.** Each retrieval change was measured on dev and test splits.

## Key decisions

| Decision | Why | Trade-off |
|---|---|---|
| Rebuild table rows from word positions | Plain extraction emits one cell per line, separating a label from its numbers | Page-level heuristic; two-column prose pages keep plain text |
| Repeat the year header in every table chunk | A row is meaningless without its column years | Slightly larger chunks |
| Tag primary-statement pages from the heading | Questions often name the statement ("per the balance sheet") | Heading heuristic; notes can be mis-tagged |
| Statement-aware ranking: tagged pages first when the question names a statement | Largest measured gain (test recall@5 0.48 to 0.65) | Only helps questions that name a statement |
| Small accounting-synonym lexicon | Bridges "capex" and "purchases of PP&E" without LLM calls | Hand-maintained; entries partly inspired by questions seen during development, so treat its gain as optimistic |
| Line-item table (BM25 over labels + SQLite) | Direct, deterministic lookup of statement values with year and unit | Parsing quality varies across filings |
| Read-only SQL via SQLite authorizer | Safety enforced by the database, not string matching | Models can still write wrong SQL; errors are returned as observations |
| Grounding verifier with digit-aware tolerance | Cheap, deterministic hallucination check | Not a correctness proof; coincidental matches possible for short numbers |
| Confidence = model estimate x grounding x tool errors | Simple, explainable indicator | Not calibrated; do not read it as a probability |
| Resumable runs, disk cache, quota stop | Free-tier quotas are small and sessions die | Cached answers persist: change `--run-name` when prompts change |

## Agent loop

1. **Plan** (1 call): approach, search queries and line items in the filing's own wording.
2. **Gather** (0 calls): fuse retrieval over the question and planned queries; look up planned line items.
3. **Act** (up to `max_steps` calls): `search_filing`, `find_line_items`, `calculate`, `sql`, `final_answer`.
   Tool errors are returned as observations so the model can recover.
4. **Verify** (0 calls): every number in the answer must appear in retrieved text or a calculation.
5. **Report**: answer, calculations, evidence, warnings, confidence.

Typical cost is 3 to 6 generation calls per question. Quota errors propagate and stop the run.

## Known limitations

- One filing per question: no cross-document retrieval or comparison across companies.
- Charts are only visible through page images in `mmrag`; their data is not extracted.
- The parsers and heuristics were developed on 10-K style PDFs; scanned filings need OCR.
- The LLM-driven components (planner, agent, judge, embeddings) are tested with fakes in this
  repository; their real-API behaviour should be validated on a small dev run first.
- LLM judging shares the answering model by default, which can favour its own style.
