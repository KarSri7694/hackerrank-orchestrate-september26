# Buy or Wait? implementation

The implementation uses a deterministic finance core with a small two-model review layer:

- `buywait/domain.py` contains framework-independent domain models.
- `buywait/reconciliation.py` and `buywait/recurrence.py` reconstruct lifecycle/evidence state and recurring streams.
- `buywait/core.py` contains deterministic forecasting, simulation, optimization, plan generation, and ranking.
- `buywait/application.py` exposes use cases over repository and exchange-rate ports.
- `buywait/adapters/` contains CSV, FX, cache, and evidence adapters.
- `agent/runner.py` sends the complete case to a decision model, then a critic model.

Run deterministic-only generation from the repository root:

```text
python code/main.py --no-ai --output output.csv
```

This writes a submission-ready `output.csv` in request order, validates it, and writes deterministic grounded explanations. Use deterministic-only mode for tests and debugging when no model provider is available.

Enable the decision-and-critic pipeline by copying `.env.example` to `.env` and configuring a provider:

```text
Copy-Item .env.example .env
# Edit .env: set AI_MODE=enabled and OPENAI_API_KEY=...
python code/main.py --require-ai --output output.csv
```

Configuration is loaded from `.env` with `python-dotenv`. `OPENAI_BASE_URL` supports compatible OpenAI-style endpoints. `OPENAI_REASONING_EFFORT` controls model reasoning effort (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`, subject to model support) and defaults to `medium`. `--require-ai` fails clearly when the provider/key is unavailable or vision is disabled while the dataset contains image evidence; without it, `AI_MODE=auto` uses deterministic-only mode. When enabled, both model calls receive the full structured user case and relevant images. They can only choose among code-generated, simulator-safe candidates, using four read-only tools: inspect a source, inspect the forecast, calculate eligible payment terms, and verify a candidate. The payment calculator reports exact full, partial, and installment amounts, counts, dates, and fees from code-generated options only. The final run writes `code/evaluation/usage_report.md`; optional `OPENAI_INPUT_COST_PER_MILLION` and `OPENAI_OUTPUT_COST_PER_MILLION` enable cost estimates.

`fastmcp` is declared in the repository-level `requirements.txt`.

Run the sample regression evaluator:

```text
python code/evaluation/main.py --samples
```

It uses the same application and evidence-aware reconciliation path as production, compares the six labeled fields semantically, and prints expected-versus-actual diffs. With no API key, evidence extraction is intentionally empty; with a configured key, message and image facts are extracted before evaluation. Every provider call made during an AI run is flushed immediately to `evaluation/model_turns.jsonl`, including evidence-extraction retries and critic turns; reasoning content is retained for inspection, while credentials and image data URLs are redacted. Use `--trace-path <path>` to choose another trace file.

Validate an already generated submission against the dataset and deterministic 90-day simulator:

```text
python code/evaluation/output_validation.py --output output.csv --dataset dataset
```
