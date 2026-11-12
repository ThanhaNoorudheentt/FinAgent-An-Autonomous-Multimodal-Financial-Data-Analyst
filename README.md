# FinAgent: An Autonomous Multimodal Financial Data Analyst

FinAgent answers questions about SEC filings (10-K, 10-Q, 8-K, earnings PDFs). It finds the right
pages, reads tables as tables, **computes with tools instead of trusting LLM arithmetic**, checks
its own numbers against the source, and returns an evidence-grounded report.

```
PDF ──► layout-aware parsing ──► chunks (+ statement tags) ──► BM25 / dense / hybrid retrieval
           │                                                        │
           └─► line-item table (SQLite) ────────────────┐           ▼
                                                        ▼     Statement-aware ranking
 question ──► Planner ──► retrieve ──► Agent loop: find_line_items | search | calculate | sql
                                                        │
                                           Verifier (number grounding) ──► Report
```

## Three systems, one benchmark

| System | Parsing | Retrieval | Extra | Tools / planning |
|---|---|---|---|---|
| `rag` — conventional RAG | plain text | BM25 / dense / hybrid | – | none |
| `mmrag` — multimodal RAG | table rows + statement tags | + synonyms, statement-aware | page images sent to the model | none |
| `agent` — agentic RAG | same as `mmrag` | same | line-item table | planner, calculator, SQL, verifier |

All three share the LLM, the judge and the evaluation harness, so differences are attributable.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                       # add FINAGENT_GOOGLE_API_KEY (Google AI Studio)

finagent download-pdfs                     # the 84 FinanceBench filings
finagent eval-retrieval --variant plain --split dev      # free, no API key
finagent eval-retrieval --variant full  --split dev

# Ask about any filing and get an analyst report
finagent ask --pdf data/raw/financebench_pdfs/3M_2018_10K.pdf \
  --question "What was FY2018 capital expenditure in USD millions?" --report-out report.md
```

## Commands

| Command | Purpose | API calls |
|---|---|---|
| `download`, `download-pdfs` | benchmark data and filings | none |
| `eval-retrieval --variant plain\|layout\|layout+syn\|full --retriever bm25\|dense\|hybrid` | recall@k, MRR, per-question rows (`--out`) | none (bm25) / embeddings |
| `eval-finqa --system direct\|program` | FinQA with gold context: LLM arithmetic vs calculator | 1 per question |
| `eval-financebench --system rag\|mmrag\|agent` | end-to-end FinanceBench, judged | 1 / 1 / ~4-8, plus 1 judge |
| `ask` | one question, one filing, report output | as above |
| `compare run1.jsonl run2.jsonl ...` | accuracy, evidence recall, LLM calls, latency side by side | none |

Runs are **resumable** and **quota-aware**: finished questions are skipped, LLM calls are cached
on disk, and an exhausted daily quota stops the run cleanly (exit code 3). Use a new `--run-name`
whenever you change prompts, chunking or `k`. Use `--split dev` while developing and `--split test`
only for final numbers.

## What has been measured so far

Retrieval (BM25, all 84 filings, recall = share of gold evidence pages found in the top-k chunks):

| variant | dev recall@5 / @10 | test recall@5 / @10 | test MRR |
|---|---|---|---|
| plain text (baseline) | 0.289 / 0.389 | 0.362 / 0.438 | 0.24 |
| + table-row layout | 0.267 / 0.411 | 0.362 / 0.448 | 0.25 |
| + accounting synonyms | 0.367 / 0.544 | 0.476 / 0.598 | 0.31 |
| + statement-aware ranking (`full`) | **0.544 / 0.678** | **0.648 / 0.708** | **0.44** |

End-to-end accuracy for `rag`, `mmrag`, `agent` and the FinQA comparison requires your Gemini quota;
see `docs/EVALUATION.md` for the exact commands and budget. Fill the results table there.

## Project aims: status

| Aim | Status |
|---|---|
| 1. Multimodal pipeline: text, tables, visuals | Text and tables: done (row reconstruction, statement tagging, line items). Visuals: page images sent to Gemini in `mmrag`; chart *data* is not extracted. |
| 2. Evidence across sections, pages, documents | Sections and pages within a filing: done. **Cross-document retrieval is not implemented** (one filing per question). |
| 3. Agentic planning and tool selection | Done: planner, tool-choosing loop, step budget. |
| 4. Python/SQL tools for numbers | Done: AST-based calculator (no `exec`), read-only SQL over extracted line items. |
| 5. Evidence-grounded reports with confidence | Done: Markdown report with calculations, sources, verification notes. Confidence is a heuristic, not calibrated. |
| 6. Benchmarking and comparison | Harness, splits, judge and comparison table done; **end-to-end numbers need your API runs**. |

## Quality gates

```bash
make check        # ruff, ruff format --check, mypy --strict, pytest (no network)
```

`pytest` runs ~200 tests offline using a scripted fake LLM and synthetic PDFs. Tests marked `live`
(none by default) would call the real API.

## Layout

```
src/finagent/
  llm/          LLMClient protocol, GeminiClient (cache, rate limit, retry, quota), FakeClient
  ingestion/    pdf parsing, layout rows, statement tagging, chunking, line items, page rendering
  indexing/     numpy vector index, build-and-cache per filing
  retrieval/    BM25, dense, hybrid (RRF), statement-aware, accounting synonyms
  tools/        safe calculator, read-only SQL
  agent/        planner/step schemas, prompts, agent loop, verifier
  systems/      direct + program-of-thought (FinQA), RAG / multimodal RAG, agent adapter
  evaluation/   loaders, metrics, judge, harness, splits, retrieval eval
  reporting/    Markdown analyst report
  pipeline.py   builds a FilingContext per PDF; cli.py command line
docs/           DESIGN.md, EVALUATION.md, REFERENCES.md
```

### Google Colab

```python
!git clone <your-repo-url> /content/finagent && pip install -q -e /content/finagent
import os
from google.colab import userdata, drive
os.environ["FINAGENT_GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
drive.mount("/content/drive")
os.environ["FINAGENT_CACHE_DIR"] = "/content/drive/MyDrive/finagent/cache"
os.environ["FINAGENT_RUNS_DIR"] = "/content/drive/MyDrive/finagent/runs"
os.environ["FINAGENT_INDEX_DIR"] = "/content/drive/MyDrive/finagent/index"
!cd /content/finagent && finagent -v eval-financebench --system agent --split dev
```
