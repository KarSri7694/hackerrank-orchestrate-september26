from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from agent.config import AgentConfig
from agent.usage import UsageTracker

from buywait.application import Application
from buywait.output import write_output
from evaluation.output_validation import validate_output


def appended_request_ids(processed_ids: tuple[str, ...], observed_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Return only append-only requests; reject an edited historical queue."""
    if observed_ids[:len(processed_ids)] != processed_ids:
        raise ValueError("requests.csv changed an already processed request; only append new rows")
    return observed_ids[len(processed_ids):]


def main() -> None:
    """Generate and validate a submission-ready Buy or Wait output.csv."""
    parser = argparse.ArgumentParser(description="Generate Buy or Wait submission output")
    parser.add_argument("--output", type=Path, default=None, help="CSV destination (default: repository-root output.csv)")
    parser.add_argument("--require-ai", action="store_true", help="fail instead of silently omitting AI evidence extraction")
    parser.add_argument("--no-ai", action="store_true", help="deterministic-only mode for debugging and tests")
    watch_group = parser.add_mutually_exclusive_group()
    watch_group.add_argument("--watch", dest="watch", action="store_true", default=None,
                             help="poll for append-only request rows after the current batch")
    watch_group.add_argument("--no-watch", dest="watch", action="store_false",
                             help="exit after the current batch, even for an AI run")
    parser.add_argument("--poll-interval", type=float, default=None,
                        help="seconds between requests.csv checks (default: REQUESTS_POLL_SECONDS)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_path = args.output or root / "output.csv"
    config = AgentConfig.from_dotenv(root / ".env")
    poll_seconds = args.poll_interval if args.poll_interval is not None else config.request_poll_seconds
    if poll_seconds <= 0:
        parser.error("--poll-interval must be greater than zero")
    # --require-ai is an explicit production contract: provider failures must
    # stop the run instead of silently turning evidence into an empty result.
    run_config = replace(config, ai_mode="enabled") if args.require_ai else config
    usage = UsageTracker(config.base_url, config.input_cost_per_million, config.output_cost_per_million)
    ai_available = config.ai_mode != "disabled" and bool(config.api_key)
    if args.require_ai and (args.no_ai or not ai_available):
        raise SystemExit("--require-ai needs configured AI_MODE and OPENAI_API_KEY in .env")
    recorder = None
    if ai_available and not args.no_ai:
        from agent.trace import ModelTurnRecorder
        recorder = ModelTurnRecorder(root / "evaluation" / "model_turns.jsonl",
                                     run_id=datetime.now(timezone.utc).isoformat())
    dataset_path = root / "dataset"

    def build_application() -> Application:
        app = Application(dataset_path)
        if args.require_ai and not config.vision_enabled and any(
                ref.source_type == "image" for ref in app.repository.evidence_refs.values()):
            raise SystemExit("--require-ai needs OPENAI_VISION_ENABLED=true because the dataset contains image evidence")
        return app

    def build_runner(app: Application):
        if not ai_available or args.no_ai:
            return None
        from agent.runner import AgentRunner
        return AgentRunner(app, dataset_path, config=run_config,
                           usage_tracker=usage, trace_writer=recorder)

    app = build_application()
    runner = build_runner(app)
    watch_enabled = (runner is not None) if args.watch is None else args.watch
    decisions = []
    processed_ids = app.repository.request_ids()
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    def process(request_ids: tuple[str, ...], active_app: Application, active_runner) -> None:
        for request_id in request_ids:
            decision = active_runner.run(request_id).decision if active_runner else active_app.decision(request_id)
            decisions.append(active_app.finalized_decision(request_id, decision))
            # The checkpoint is deliberately separate from the final output.
            # A reader never sees a partial submission at output_path.
            write_output(temporary_path, decisions)
            print(f"[{len(decisions)}] processed {request_id}", flush=True)

    def publish(active_app: Application) -> None:
        write_output(temporary_path, decisions)
        validate_output(temporary_path, active_app)
        os.replace(temporary_path, output_path)
        usage.report(root / "code" / "evaluation" / "usage_report.md", len(decisions))
        print(f"wrote and validated {len(decisions)} rows to {output_path}", flush=True)

    try:
        process(processed_ids, app, runner)
        publish(app)
        if watch_enabled:
            print(f"watching {dataset_path / 'requests.csv'} every {poll_seconds:g}s for appended rows", flush=True)
        while watch_enabled:
            time.sleep(poll_seconds)
            try:
                next_app = build_application()
                next_ids = next_app.repository.request_ids()
                new_ids = appended_request_ids(processed_ids, next_ids)
            except (OSError, ValueError) as exc:
                # A producer may be in the middle of appending a CSV row.
                # Keep the validated output and retry on the next poll.
                print(f"waiting for a valid append to requests.csv: {exc}", flush=True)
                continue
            if not new_ids:
                continue
            app = next_app
            runner = build_runner(app)
            process(new_ids, app, runner)
            processed_ids = next_ids
            publish(app)
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
