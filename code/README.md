# Buy or Wait? implementation

The implementation uses a hexagonal architecture:

- `buywait/domain.py` contains framework-independent domain models.
- `buywait/core.py` contains deterministic reconciliation, forecasting, simulation, plan generation, and ranking.
- `buywait/application.py` exposes use cases over repository and exchange-rate ports.
- `buywait/adapters/` contains CSV, FX, cache, and evidence adapters.
- `tools/server.py` is the FastMCP inbound adapter.

Run the deterministic solver from the repository root:

```text
python code/main.py
```

It prints one structured JSON decision per request for manual export. It does not write the final `output.csv` or generate decision explanations.

Run the MCP server:

```text
fastmcp run code/tools/server.py:mcp
```

Available tools are `get_case`, `inspect_evidence`, `simulate_plan`, and `optimize_spending`. The tools expose application services; all financial arithmetic and plan verification stay in the deterministic core.

Enable the bounded OpenAI agent loop by copying `.env.example` to `.env` and filling in the key:

```text
Copy-Item .env.example .env
# Edit .env: set AI_MODE=enabled and OPENAI_API_KEY=...
python code/main.py
```

Configuration is loaded from `.env` with `python-dotenv`. `OPENAI_BASE_URL` supports compatible OpenAI-style endpoints. The runner defaults to four turns and eight tool calls, sends only relevant local evidence images as base64 `input_image` parts, and falls back to deterministic solving in `auto` mode.

`fastmcp` is declared in the repository-level `requirements.txt`.
