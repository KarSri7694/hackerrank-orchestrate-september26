# Buy or Wait? implementation

The implementation uses a hexagonal architecture:

- `buywait/domain.py` contains framework-independent domain models.
- `buywait/reconciliation.py` and `buywait/recurrence.py` reconstruct lifecycle/evidence state and recurring streams.
- `buywait/core.py` contains deterministic forecasting, simulation, optimization, plan generation, and ranking.
- `buywait/application.py` exposes use cases over repository and exchange-rate ports.
- `buywait/adapters/` contains CSV, FX, cache, and evidence adapters.
- `tools/server.py` is the FastMCP inbound adapter.

Run deterministic-only generation from the repository root:

```text
python code/main.py --no-ai --output output.csv
```

This writes a submission-ready `output.csv` in request order, validates it, and writes deterministic grounded explanations. Use deterministic-only mode for tests and debugging when no model provider is available.

Run the MCP server:

```text
fastmcp run code/tools/server.py:mcp
```

Available tools are `get_case`, `inspect_evidence`, `calculator`, `simulate_plan`, and `optimize_spending`. `calculator` provides exact decimal arithmetic for model reasoning; `simulate_plan` remains authoritative for financial safety. The tools expose application services; all financial arithmetic and plan verification stay in the deterministic core.

Enable the bounded evidence extractor and critic loop by copying `.env.example` to `.env` and configuring a provider:

```text
Copy-Item .env.example .env
# Edit .env: set AI_MODE=enabled and OPENAI_API_KEY=...
python code/main.py --require-ai --output output.csv
```

Configuration is loaded from `.env` with `python-dotenv`. `OPENAI_BASE_URL` supports compatible OpenAI-style endpoints. `OPENAI_REASONING_EFFORT` controls model reasoning effort (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`, subject to model support) and defaults to `medium`; it is sent as `reasoning.effort` for Responses critic calls and `reasoning_effort` for Chat Completions extraction calls. `OPENAI_TIMEOUT` controls ordinary API/critic calls and `OPENAI_EVIDENCE_TIMEOUT` separately bounds slower thinking/VLM extraction calls; reasoning is not disabled. `--require-ai` fails clearly when the provider/key is unavailable or vision is disabled while the dataset contains image evidence; without it, `AI_MODE=auto` may use deterministic-only mode. When enabled, the extractor converts relevant messages and images into structured evidence facts, deterministic Python validates and reconciles them, and the bounded critic may only submit evidence-backed corrections for deterministic verification. The final run writes `code/evaluation/usage_report.md`; optional `OPENAI_INPUT_COST_PER_MILLION` and `OPENAI_OUTPUT_COST_PER_MILLION` enable cost estimates.

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
