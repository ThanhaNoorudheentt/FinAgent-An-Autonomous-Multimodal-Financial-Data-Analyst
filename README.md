# FinAgent: An Autonomous Multimodal Financial Data Analyst

Stage 0 foundation: configuration, typed schemas, a resilient Gemini client (disk cache,
rate limiting, retries), benchmark loaders, a resumable evaluation harness, and a no-tools
FinQA baseline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

This installs FinAgent and its dependencies, including:
- Pydantic for configuration and data validation.
- Google Gen AI for Gemini integration.
- Pytest and other development tools.

pre-commit install
cp .env.example .env        # then add your Google AI Studio key
```

## Usage

```bash
finagent download                         # FinQA test + FinanceBench open sample
finagent -v eval-finqa --limit 20         # baseline run; results in runs/*.jsonl
```

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
| `src/finagent/systems/` | QA systems (Stage 0: direct baseline) |
