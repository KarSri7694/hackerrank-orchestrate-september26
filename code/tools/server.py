from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

from fastmcp import FastMCP

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buywait.application import Application  # noqa: E402
from buywait.domain import Payment, SpendingChange  # noqa: E402


def _dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "dataset"


app = Application(_dataset_dir())
mcp = FastMCP("buy-or-wait")


def configure_application(application: Application) -> None:
    """Bind MCP tools to the same application state used by the agent."""
    global app
    app = application


@mcp.tool
def get_case(request_id: str) -> dict[str, object]:
    """Return reconstructed financial state and baseline affordability for a request."""
    return app.case(request_id)


@mcp.tool
def inspect_evidence(evidence_id: str) -> list[dict[str, object]]:
    """Return structured facts from one relevant message or image."""
    ref = app.repository.evidence(evidence_id)
    return {"evidence_id": evidence_id, "source_type": ref.source_type,
            "sent_at": ref.sent_at.isoformat() if ref.sent_at else None, "text": ref.text,
            "image_available": bool(ref.image_path), "facts": [
                {"evidence_id": f.evidence_id, "effect": f.effect, "related_event_id": f.related_event_id,
                 "amount": str(f.amount) if f.amount is not None else None, "currency": f.currency,
                 "effective_date": f.effective_date.isoformat() if f.effective_date else None,
                "status": f.status.value if f.status else None, "confidence": f.confidence,
                 "source_type": f.source_type, "sent_at": f.sent_at.isoformat() if f.sent_at else None,
                 "category": f.category, "direction": f.direction, "description": f.description,
                 "recurring": f.recurring, "recurrence_days": f.recurrence_days}
                for f in app.evidence_service.inspect(evidence_id)]}


@mcp.tool
def simulate_plan(request_id: str, payments: list[dict[str, str]], spending_changes: list[dict[str, str | None]] | None = None) -> dict[str, object]:
    """Prove whether a proposed dated payment plan preserves the minimum balance."""
    parsed_payments = tuple(Payment(date.fromisoformat(p["date"]), Decimal(p["amount"])) for p in payments)
    parsed_changes = tuple(SpendingChange(c["event_id"], c["action"], Decimal(c["new_amount"]) if c.get("new_amount") else None) for c in (spending_changes or []))
    result = app.simulate_plan(request_id, parsed_payments, parsed_changes)
    return {"safe": result.safe, "minimum_projected_balance": str(result.minimum_projected_balance), "required_minimum_balance": str(result.required_minimum_balance), "first_violation_date": result.first_violation_date.isoformat() if result.first_violation_date else None, "shortfall": str(result.shortfall), "ending_balance": str(result.ending_balance)}


@mcp.tool
def optimize_spending(request_id: str, payments: list[dict[str, str]]) -> dict[str, object]:
    """Find legal flexible-spending changes that make a payment plan safe."""
    parsed = tuple(Payment(date.fromisoformat(p["date"]), Decimal(p["amount"])) for p in payments)
    return app.optimize_spending(request_id, parsed)
