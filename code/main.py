from __future__ import annotations

import json
from pathlib import Path
from agent.config import AgentConfig

from buywait.application import Application


def main() -> None:
    """Run the deterministic solver and print structured decisions for manual export."""
    root = Path(__file__).resolve().parents[1]
    app = Application(root / "dataset")
    config = AgentConfig.from_dotenv(root / ".env")
    runner = None
    if config.ai_mode != "disabled":
        from tools.server import mcp
        from agent.runner import AgentRunner
        runner = AgentRunner(app, mcp, root / "dataset", config=config)
    for request_id in app.repository.request_ids():
        decision = (runner.run(request_id).decision if runner else app.decision(request_id))
        print(json.dumps({
            "request_id": decision.request_id,
            "amount_safe_to_pay": str(decision.amount_safe_to_pay),
            "affordability_status": decision.affordability_status.value,
            "recommended_payment_method": decision.recommended_payment_method.value,
            "payment_plan": [{"date": p.date.isoformat(), "amount": str(p.amount)} for p in decision.payment_plan],
            "earliest_date_for_full_payment": decision.earliest_date_for_full_payment.isoformat() if decision.earliest_date_for_full_payment else None,
            "spending_changes": [c.__dict__ for c in decision.spending_changes],
            "trace": decision.trace,
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
