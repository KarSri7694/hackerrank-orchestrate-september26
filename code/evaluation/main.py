from __future__ import annotations

import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buywait.application import Application  # noqa: E402
from invariants import validate_decision  # noqa: E402
from sample_regression import evaluate_samples, print_progress, print_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Buy or Wait evaluation harness")
    parser.add_argument("--samples", action="store_true", help="run field-by-field regression against sample_requests.csv")
    parser.add_argument("--no-ai", action="store_true", help="run deterministic evaluation without the model review")
    parser.add_argument("--mode", choices=("deterministic", "agent"),
                        help="evaluation path: solver only or the decision-and-critic pipeline")
    parser.add_argument("--require-ai", action="store_true", help="fail if the configured AI provider is unavailable")
    parser.add_argument("--limit", type=int, help="evaluate only the first N sample requests")
    parser.add_argument("--offset", type=int, default=0, help="skip this many sample requests before applying --limit")
    parser.add_argument("--trace-path", default=str(Path("evaluation") / "model_turns.jsonl"),
                        help="JSONL file receiving every model turn (default: evaluation/model_turns.jsonl)")
    args = parser.parse_args()
    root = CODE_DIR.parent
    if args.samples:
        from agent.config import AgentConfig
        config = AgentConfig.from_dotenv(root / ".env")
        mode = args.mode or ("deterministic" if args.no_ai else "agent")
        if args.no_ai and args.mode and args.mode != "deterministic":
            raise SystemExit("--no-ai is only compatible with --mode deterministic")
        if args.require_ai and (mode == "deterministic" or config.ai_mode == "disabled" or not config.api_key):
            raise SystemExit("--require-ai needs configured AI_MODE and OPENAI_API_KEY in .env")
        if mode == "agent" and config.ai_mode != "disabled" and config.api_key:
            from agent.trace import ModelTurnRecorder
            recorder = ModelTurnRecorder(root / args.trace_path,
                                         run_id=datetime.now(timezone.utc).isoformat())
        decision_provider = None
        if mode == "agent" and config.ai_mode != "disabled" and config.api_key:
            from agent.runner import AgentRunner
            def decision_provider(app, request_id):
                return AgentRunner(app, root / "dataset", config=config,
                                   trace_writer=recorder).run(request_id).decision
        else:
            recorder = None
        try:
            print_report(evaluate_samples(root / "dataset", progress=print_progress,
                                          decision_provider=decision_provider, limit=args.limit,
                                          offset=args.offset))
        finally:
            if recorder is not None:
                recorder.close()
        return
    app = Application(root / "dataset")
    for request_id in app.repository.request_ids():
        validate_decision(app.decision(request_id), app.repository.requests[request_id].requested_amount)
    print(f"validated {len(app.repository.request_ids())} decisions")


if __name__ == "__main__":
    main()
