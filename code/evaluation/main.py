from __future__ import annotations

import sys
import argparse
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buywait.application import Application  # noqa: E402
from invariants import validate_decision  # noqa: E402
from sample_regression import evaluate_samples, print_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Buy or Wait evaluation harness")
    parser.add_argument("--samples", action="store_true", help="run field-by-field regression against sample_requests.csv")
    args = parser.parse_args()
    root = CODE_DIR.parent
    if args.samples:
        extractor = None
        from agent.config import AgentConfig
        config = AgentConfig.from_dotenv(root / ".env")
        if config.ai_mode != "disabled" and config.api_key:
            from agent.evidence_extractor import OpenAIEvidenceExtractor
            extractor = OpenAIEvidenceExtractor(config)
        print_report(evaluate_samples(root / "dataset", evidence_extractor=extractor))
        return
    app = Application(root / "dataset")
    for request_id in app.repository.request_ids():
        validate_decision(app.decision(request_id), app.repository.requests[request_id].requested_amount)
    print(f"validated {len(app.repository.request_ids())} decisions")


if __name__ == "__main__":
    main()
