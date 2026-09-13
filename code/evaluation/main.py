from __future__ import annotations

import sys
import argparse
import tempfile
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
    parser.add_argument("--no-ai", action="store_true", help="run deterministic evaluation without evidence extraction or the critic")
    parser.add_argument("--mode", choices=("deterministic", "extractor", "agent"),
                        help="evaluation path: solver only, extraction plus solver, or full AgentRunner")
    parser.add_argument("--fresh-evidence", action="store_true",
                        help="use an empty temporary evidence cache and reject legacy cache reuse")
    parser.add_argument("--require-ai", action="store_true", help="fail if the configured AI provider is unavailable")
    parser.add_argument("--trace-path", default=str(Path("evaluation") / "model_turns.jsonl"),
                        help="JSONL file receiving every model turn (default: evaluation/model_turns.jsonl)")
    args = parser.parse_args()
    root = CODE_DIR.parent
    if args.samples:
        extractor = None
        from agent.config import AgentConfig
        config = AgentConfig.from_dotenv(root / ".env")
        mode = args.mode or ("deterministic" if args.no_ai else "agent")
        if args.no_ai and args.mode and args.mode != "deterministic":
            raise SystemExit("--no-ai is only compatible with --mode deterministic")
        if args.require_ai and (mode == "deterministic" or config.ai_mode == "disabled" or not config.api_key):
            raise SystemExit("--require-ai needs configured AI_MODE and OPENAI_API_KEY in .env")
        if mode != "deterministic" and config.ai_mode != "disabled" and config.api_key:
            from agent.evidence_extractor import OpenAIEvidenceExtractor
            from agent.trace import ModelTurnRecorder
            recorder = ModelTurnRecorder(root / args.trace_path,
                                         run_id=datetime.now(timezone.utc).isoformat())
            extractor = OpenAIEvidenceExtractor(config, trace_writer=recorder)
        decision_provider = None
        if mode == "agent" and extractor is not None:
            from agent.runner import AgentRunner
            from tools import server
            def decision_provider(app, request_id):
                server.configure_application(app)
                return AgentRunner(app, server.mcp, root / "dataset", config=config,
                                   trace_writer=recorder).run(request_id).decision
        else:
            recorder = None
        temporary_cache = tempfile.TemporaryDirectory() if args.fresh_evidence else None
        cache_path = Path(temporary_cache.name) / "facts.json" if temporary_cache else None
        try:
            print_report(evaluate_samples(root / "dataset", evidence_extractor=extractor, progress=print_progress,
                                          decision_provider=decision_provider, evidence_cache_path=cache_path,
                                          allow_legacy_evidence_cache=not args.fresh_evidence))
        finally:
            if recorder is not None:
                recorder.close()
            if temporary_cache is not None:
                temporary_cache.cleanup()
        return
    app = Application(root / "dataset")
    for request_id in app.repository.request_ids():
        validate_decision(app.decision(request_id), app.repository.requests[request_id].requested_amount)
    print(f"validated {len(app.repository.request_ids())} decisions")


if __name__ == "__main__":
    main()
