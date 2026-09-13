from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from agent.config import AgentConfig
from agent.usage import UsageTracker

from buywait.application import Application
from buywait.output import write_output
from evaluation.output_validation import validate_output


def main() -> None:
    """Generate and validate a submission-ready Buy or Wait output.csv."""
    parser = argparse.ArgumentParser(description="Generate Buy or Wait submission output")
    parser.add_argument("--output", type=Path, default=None, help="CSV destination (default: repository-root output.csv)")
    parser.add_argument("--require-ai", action="store_true", help="fail instead of silently omitting AI evidence extraction")
    parser.add_argument("--no-ai", action="store_true", help="deterministic-only mode for debugging and tests")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_path = args.output or root / "output.csv"
    config = AgentConfig.from_dotenv(root / ".env")
    # --require-ai is an explicit production contract: provider failures must
    # stop the run instead of silently turning evidence into an empty result.
    run_config = replace(config, ai_mode="enabled") if args.require_ai else config
    usage = UsageTracker(config.base_url, config.input_cost_per_million, config.output_cost_per_million)
    extractor = None
    ai_available = config.ai_mode != "disabled" and bool(config.api_key)
    if args.require_ai and (args.no_ai or not ai_available):
        raise SystemExit("--require-ai needs configured AI_MODE and OPENAI_API_KEY in .env")
    recorder = None
    if ai_available and not args.no_ai:
        from agent.evidence_extractor import OpenAIEvidenceExtractor
        from agent.trace import ModelTurnRecorder
        recorder = ModelTurnRecorder(root / "evaluation" / "model_turns.jsonl",
                                     run_id=datetime.now(timezone.utc).isoformat())
        extractor = OpenAIEvidenceExtractor(run_config, usage_tracker=usage,
                                            trace_writer=recorder)
        if args.require_ai:
            extractor.fail_open = False
    app = Application(root / "dataset", evidence_extractor=extractor)
    if args.require_ai and not config.vision_enabled and any(
            ref.source_type == "image" for ref in app.repository.evidence_refs.values()):
        raise SystemExit("--require-ai needs OPENAI_VISION_ENABLED=true because the dataset contains image evidence")
    runner = None
    if extractor is not None:
        from tools import server
        from agent.runner import AgentRunner
        server.configure_application(app)
        runner = AgentRunner(app, server.mcp, root / "dataset", config=run_config,
                             usage_tracker=usage, trace_writer=recorder)
    decisions = []
    request_ids = app.repository.request_ids()
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        for index, request_id in enumerate(request_ids, 1):
            decision = runner.run(request_id).decision if runner else app.decision(request_id)
            decisions.append(app.finalized_decision(request_id, decision))
            # Publish each completed row immediately for live inspection. The
            # final validated result is still atomically replaced below.
            write_output(output_path, decisions)
            print(f"[{index}/{len(request_ids)}] processed {request_id}", flush=True)
        write_output(temporary_path, decisions)
        validate_output(temporary_path, app)
        os.replace(temporary_path, output_path)
        usage.report(root / "code" / "evaluation" / "usage_report.md", len(decisions), app.evidence_service.cache_hits)
        print(f"wrote and validated {len(decisions)} rows to {output_path}")
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
