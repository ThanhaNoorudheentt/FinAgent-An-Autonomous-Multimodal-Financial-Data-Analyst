# FinAgent: An Autonomous Multimodal Financial Data Analyst

- **Stage 0:** config, typed schemas, resilient Gemini client (disk cache, rate limiting,
  retries), benchmark loaders, resumable evaluation harness, no-tools FinQA baseline.
- **Stage 1:** PDF parsing, page-bounded chunking, BM25 / dense / hybrid (RRF) retrieval, and a
  conventional-RAG baseline on FinanceBench scored by an LLM judge plus evidence recall.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
cp .env.example .env        # then add your Google AI Studio key
```

## Usage

```bash
finagent download                         # FinQA test + FinanceBench open sample
finagent -v eval-finqa --limit 20         # baseline run; results in runs/*.jsonl
```

### Stage 1: RAG on FinanceBench

```bash
finagent download-pdfs --limit 20          # filings for the first 20 questions
# or place PDFs yourself at data/raw/financebench_pdfs/<doc_name>.pdf  (e.g. 3M_2018_10K.pdf)
finagent -v eval-financebench --retriever bm25   --limit 20    # free: no embeddings
finagent -v eval-financebench --retriever dense  --limit 20
finagent -v eval-financebench --retriever hybrid --limit 20 --judge-model <other-model>
```

Results go to `runs/financebench-<run-name>.jsonl`, judge reasons to `*.judge.jsonl`.
The run name defaults to `rag-<retriever>-k<k>`; **use a new `--run-name` whenever you change
chunking, k or prompts**, because finished questions are skipped on resume.
Page numbers are 0-based everywhere (verified against FinanceBench gold pages).
Embedding a filing costs roughly 5 API calls; indexes are cached in `.cache/index`.

### Retrieval-only evaluation (no generation or judge calls)

```bash
finagent eval-retrieval --retriever bm25 --limit 150      # free, needs no API key
finagent eval-retrieval --retriever dense --limit 150     # embedding calls only
```

Reports recall@k (share of gold evidence pages found in the top-k chunks) and MRR. Use it to
choose chunking and retrievers *before* spending generation quota on end-to-end runs.

### API quotas

Free-tier quotas are per model and can be as low as ~20 generation calls per day; a FinanceBench
question uses two (answer + judge). When a daily quota is exhausted the run **stops cleanly**
(exit code 3), keeps all finished results, and resumes where it left off when you re-run the same
command. Spread load with `--judge-model <other-model>`, or enable billing for larger limits.

Runs are resumable: re-running the same command skips finished questions, so an interrupted
Colab session never re-spends API budget. LLM calls are cached on disk in `.cache/llm`.

### Google Colab

```python
!git clone <your-repo-url> && cd finagent && pip install -e .
from google.colab import userdata, drive
import os; os.environ["FINAGENT_GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
drive.mount("/content/drive")
os.environ["FINAGENT_CACHE_DIR"] = "/content/drive/MyDrive/finagent/cache"
os.environ["FINAGENT_RUNS_DIR"] = "/content/drive/MyDrive/finagent/runs"
!finagent -v eval-finqa --limit 20
```

## Quality gates

```bash
ruff check . && ruff format --check . && mypy && pytest -m "not live"
```

## Layout

| Path | Purpose |
|---|---|
| `src/finagent/llm/` | `LLMClient` protocol, `GeminiClient`, `FakeClient`, cache, rate limit/retry |
| `src/finagent/schemas.py` | `Evidence`, `Calculation`, `Answer`, benchmark example models |
| `src/finagent/evaluation/` | dataset loaders, metrics, LLM judge, resumable harness |
| `src/finagent/ingestion/` | PDF text extraction, page-bounded chunking |
| `src/finagent/indexing/` | numpy vector index, build-and-cache per filing |
| `src/finagent/retrieval/` | BM25, dense, hybrid (reciprocal rank fusion) |
| `src/finagent/systems/` | QA systems: direct FinQA baseline, FinanceBench RAG |
