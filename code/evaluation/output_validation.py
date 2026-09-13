from __future__ import annotations

import csv
import argparse
import sys
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buywait.core import _calendar_months_between, baseline, expand_option
from buywait.domain import AffordabilityStatus, Payment, PaymentMethod, SpendingChange
from buywait.output import OUTPUT_COLUMNS


class OutputValidationError(ValueError):
    pass


def _validate_explanation(value: str, method: PaymentMethod) -> None:
    """Reject plainly contradictory explanations without judging wording style."""
    text = value.casefold()
    negative = ("do not proceed", "not affordable", "no eligible payment")
    if method == PaymentMethod.NOT_RECOMMENDED:
        if not any(phrase in text for phrase in negative):
            raise OutputValidationError("not_recommended explanation must state that proceeding is unsafe or unavailable")
        return
    if any(phrase in text for phrase in negative):
        raise OutputValidationError("explanation contradicts a recommended payment plan")
    if method == PaymentMethod.WAIT and "wait" not in text:
        raise OutputValidationError("wait explanation must describe waiting")
    if method == PaymentMethod.FULL and "wait until" in text:
        raise OutputValidationError("full-payment explanation contradicts the selected method")


def _payments(value: str) -> tuple[Payment, ...]:
    if value == "none":
        return ()
    try:
        return tuple(Payment(date.fromisoformat(part.rsplit(":", 1)[0]), Decimal(part.rsplit(":", 1)[1])) for part in value.split("|"))
    except (ValueError, InvalidOperation, IndexError) as exc:
        raise OutputValidationError("invalid payment_plan") from exc


def _changes(value: str) -> tuple[SpendingChange, ...]:
    if value == "none":
        return ()
    changes = []
    try:
        for part in value.split("|"):
            bits = part.split(":")
            if bits[0] == "stop" and len(bits) == 2:
                changes.append(SpendingChange(bits[1], "stop"))
            elif bits[0] == "reduce_to" and len(bits) == 3:
                changes.append(SpendingChange(bits[1], "reduce_to", Decimal(bits[2])))
            else:
                raise OutputValidationError("invalid spending_changes_needed")
    except InvalidOperation as exc:
        raise OutputValidationError("invalid spending_changes_needed") from exc
    return tuple(changes)


def validate_output(path: str | Path, application) -> None:
    """Validate a submission CSV against the same deterministic state used to generate it."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise OutputValidationError("output columns must exactly match the required schema")
        rows = list(reader)
    request_ids = application.repository.request_ids()
    if len(rows) != len(request_ids):
        raise OutputValidationError("output row count does not match requests.csv")
    if tuple(row.get("request_id", "") for row in rows) != request_ids:
        raise OutputValidationError("output request IDs must appear exactly once in requests.csv order")
    for row in rows:
        request_id = row["request_id"]
        request = application.repository.requests[request_id]
        try:
            safe = Decimal(row["amount_safe_to_pay"])
            status = AffordabilityStatus(row["affordability_status"])
            method = PaymentMethod(row["recommended_payment_method"])
        except (InvalidOperation, ValueError) as exc:
            raise OutputValidationError(f"{request_id}: invalid amount/status/method") from exc
        if not Decimal("0") <= safe <= request.requested_amount:
            raise OutputValidationError(f"{request_id}: amount_safe_to_pay is out of bounds")
        payments, changes = _payments(row["payment_plan"]), _changes(row["spending_changes_needed"])
        if payments != tuple(sorted(payments, key=lambda payment: payment.date)):
            raise OutputValidationError(f"{request_id}: payment plan is not chronological")
        if payments and payments[-1].date > request.desired_completion_date:
            raise OutputValidationError(f"{request_id}: payment plan misses requested completion date")
        earliest = row["earliest_date_for_full_payment"]
        try:
            earliest_date = date.fromisoformat(earliest) if earliest else None
        except ValueError as exc:
            raise OutputValidationError(f"{request_id}: invalid earliest_date_for_full_payment") from exc
        ctx = application._context(request_id)
        expected_safe, expected_earliest = baseline(ctx, application.fx)
        if safe != expected_safe:
            raise OutputValidationError(f"{request_id}: amount_safe_to_pay does not match reconciled baseline")
        if earliest_date != expected_earliest:
            raise OutputValidationError(f"{request_id}: earliest_date_for_full_payment does not match baseline")
        if method not in ctx.profile.accepted_payment_methods and method not in {PaymentMethod.WAIT, PaymentMethod.NOT_RECOMMENDED}:
            raise OutputValidationError(f"{request_id}: method is not accepted by user")
        if method == PaymentMethod.PARTIAL:
            if not request.allows_partial_payment or len(payments) != 2 or payments[0].date != request.request_date or payments[0].amount != safe or sum((p.amount for p in payments), Decimal("0")) != request.requested_amount:
                raise OutputValidationError(f"{request_id}: invalid partial-payment plan")
        if method == PaymentMethod.FULL:
            if len(payments) != 1 or payments[0].date != request.request_date or payments[0].amount != request.requested_amount:
                raise OutputValidationError(f"{request_id}: invalid full-payment plan")
        if method == PaymentMethod.INSTALLMENTS:
            matching_options = [option for option in ctx.payment_options if option.payment_method == PaymentMethod.INSTALLMENTS and expand_option(option) == payments]
            if not matching_options:
                raise OutputValidationError(f"{request_id}: installment plan does not match an offered option")
            if ctx.profile.max_installment_months is None or _calendar_months_between(payments[0].date, payments[-1].date) > ctx.profile.max_installment_months:
                raise OutputValidationError(f"{request_id}: installment plan exceeds maximum duration")
        if method == PaymentMethod.WAIT:
            if (PaymentMethod.FULL not in ctx.profile.accepted_payment_methods or len(payments) != 1
                    or payments[0].amount != request.requested_amount or payments[0].date <= request.request_date):
                raise OutputValidationError(f"{request_id}: invalid wait plan")
        if status == AffordabilityStatus.NOW and earliest_date != request.request_date:
            raise OutputValidationError(f"{request_id}: affordable_now must use request_date as earliest full payment")
        if status == AffordabilityStatus.NOW and (method != PaymentMethod.FULL or len(payments) != 1):
            raise OutputValidationError(f"{request_id}: affordable_now must be a full payment today")
        if status == AffordabilityStatus.LATER and method != PaymentMethod.WAIT:
            raise OutputValidationError(f"{request_id}: affordable_later must use wait")
        if status == AffordabilityStatus.WITH_PLAN and not payments:
            raise OutputValidationError(f"{request_id}: affordable_with_plan requires a payment plan")
        if status == AffordabilityStatus.NOT_AFFORDABLE and payments:
            raise OutputValidationError(f"{request_id}: not_affordable cannot include a payment plan")
        if method == PaymentMethod.NOT_RECOMMENDED and payments:
            raise OutputValidationError(f"{request_id}: not_recommended cannot include a payment plan")
        if len(changes) > 3 or len({change.event_id for change in changes}) != len(changes):
            raise OutputValidationError(f"{request_id}: invalid spending change count")
        application._validate_changes(ctx, changes)
        if payments and not application.simulate_plan(request_id, payments, changes).safe:
            raise OutputValidationError(f"{request_id}: selected payment plan fails 90-day safety simulation")
        explanation = row["decision_explanation"].strip()
        if not explanation:
            raise OutputValidationError(f"{request_id}: decision explanation is required")
        try:
            _validate_explanation(explanation, method)
        except OutputValidationError as exc:
            raise OutputValidationError(f"{request_id}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Buy or Wait output.csv")
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--dataset", default="dataset")
    args = parser.parse_args()
    from buywait.application import Application
    validate_output(args.output, Application(args.dataset))
    print(f"validated {args.output}")


if __name__ == "__main__":
    main()
