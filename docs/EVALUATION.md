# Evaluation protocol

## Splits

`--split dev` is 30% of FinanceBench questions and `--split test` is 70%, assigned by hashing the
question id, so the split is identical on every machine. Tune on dev; run test once for the report.

## Experiments, in order

1. **Retrieval (free).** Choose the ingestion and retriever configuration.
   ```bash
   for v in plain layout layout+syn full; do finagent eval-retrieval --variant $v --split dev; done
   finagent eval-retrieval --variant full --retriever hybrid --split dev   # embeddings only
   ```
2. **FinQA tool ablation** (gold context; measures what the calculator adds).
   ```bash
   finagent eval-finqa --system direct  --split test --limit 100
   finagent eval-finqa --system program --split test --limit 100
   ```
3. **FinanceBench, three systems.** Same model, same judge.
   ```bash
   finagent eval-financebench --system rag   --retriever hybrid --split dev
   finagent eval-financebench --system mmrag --retriever hybrid --split dev
   finagent eval-financebench --system agent --retriever hybrid --split dev
   finagent compare runs/financebench-*-dev.jsonl
   ```
4. **Final** run the chosen configurations with `--split test` once and report those numbers.

## Budget (generation calls, approximate)

| Run | Questions | Calls |
|---|---|---|
| `rag` or `mmrag` | 45 dev / 105 test | 2 per question (answer + judge) |
| `agent` | 45 dev / 105 test | 4 to 7 per question + 1 judge |
| FinQA `direct` / `program` | 100 | 1 per question |

The free tier (about 20 generation calls per day per model at the time of writing) cannot cover a
full comparison; enable billing or split work across models and days. The harness resumes exactly.

## Metrics

- **Accuracy**: LLM-judge verdict against the gold answer (FinanceBench); numeric match against
  `exe_ans` or the rounded gold string (FinQA).
- **Evidence recall**: share of gold (document, page) pairs among retrieved evidence. A proxy:
  an answer can be right from other pages with the same facts.
- **LLM calls and latency** per question: the cost of agentic reasoning.
- **Verification notes**: how often answers contain numbers not traceable to the filing.

## Results (fill in after your runs)

| System | Retriever | Split | n | Accuracy | Evidence recall | Calls/q | Latency (s) |
|---|---|---|---|---|---|---|---|
| rag | hybrid | test | | | | | |
| mmrag | hybrid | test | | | | | |
| agent | hybrid | test | | | | | |

Measured so far (no API needed): retrieval table in the README.
