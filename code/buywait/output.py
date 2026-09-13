from __future__ import annotations

import csv
from pathlib import Path

from .presentation import decimal_text, payment_plan_text, spending_changes_text


OUTPUT_COLUMNS = (
    "request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
    "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation",
)


def decision_row(decision) -> dict[str, str]:
    return {
        "request_id": decision.request_id,
        "amount_safe_to_pay": decimal_text(decision.amount_safe_to_pay),
        "affordability_status": decision.affordability_status.value,
        "recommended_payment_method": decision.recommended_payment_method.value,
        "payment_plan": payment_plan_text(decision),
        "earliest_date_for_full_payment": decision.earliest_date_for_full_payment.isoformat() if decision.earliest_date_for_full_payment else "",
        "spending_changes_needed": spending_changes_text(decision),
        "decision_explanation": decision.decision_explanation,
    }


def write_output(path: str | Path, decisions) -> None:
    """Write the submission schema only, in the caller-provided request order."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision_row(decision))
