from __future__ import annotations

from decimal import Decimal

from .domain import DecisionCore, PaymentMethod


def decimal_text(value: Decimal | None) -> str:
    """Render Decimal values for CSV and explanations without exponent notation."""
    return "" if value is None else format(value, "f")


def payment_plan_text(decision: DecisionCore) -> str:
    if not decision.payment_plan:
        return "none"
    return "|".join(f"{payment.date.isoformat()}:{decimal_text(payment.amount)}" for payment in decision.payment_plan)


def spending_changes_text(decision: DecisionCore) -> str:
    if not decision.spending_changes:
        return "none"
    values = []
    for change in decision.spending_changes:
        values.append(
            f"stop:{change.event_id}" if change.action == "stop"
            else f"reduce_to:{change.event_id}:{decimal_text(change.new_amount)}"
        )
    return "|".join(values)


def decision_explanation(decision: DecisionCore, *, currency: str,
                         minimum_balance: Decimal, projected_minimum: Decimal | None,
                         financing_fee: Decimal | None = None) -> str:
    """Create a short reproducible explanation from verified solver state only."""
    method = decision.recommended_payment_method
    amount = decimal_text(sum((payment.amount for payment in decision.payment_plan), Decimal("0")))
    minimum = decimal_text(minimum_balance)
    if method == PaymentMethod.NOT_RECOMMENDED:
        return (f"Do not proceed by the requested deadline. No eligible payment option keeps the "
                f"{currency} {minimum} minimum balance protected.")
    if method == PaymentMethod.WAIT:
        payment = decision.payment_plan[0]
        return (f"Wait until {payment.date.isoformat()} and then pay {currency} {decimal_text(payment.amount)} in full. "
                f"Paying earlier would put the {currency} {minimum} minimum balance at risk.")
    if method == PaymentMethod.INSTALLMENTS:
        first = decision.payment_plan[0]
        text = (f"Use {len(decision.payment_plan)} installments of {currency} {decimal_text(first.amount)} "
                f"starting {first.date.isoformat()}. The verified schedule keeps the {currency} {minimum} minimum protected.")
        if financing_fee and financing_fee > 0:
            text += f" Financing cost is {currency} {decimal_text(financing_fee)}."
    elif method == PaymentMethod.PARTIAL:
        first, second = decision.payment_plan
        text = (f"Pay {currency} {decimal_text(first.amount)} today and {currency} {decimal_text(second.amount)} on "
                f"{second.date.isoformat()}. The verified schedule keeps the {currency} {minimum} minimum protected.")
    else:
        payment = decision.payment_plan[0]
        text = (f"Pay {currency} {decimal_text(payment.amount)} today. The 90-day forecast remains above the "
                f"{currency} {minimum} minimum balance.")
    if decision.spending_changes:
        text += f" Required spending changes: {spending_changes_text(decision)}."
    if projected_minimum is not None:
        text += f" Projected minimum balance: {currency} {decimal_text(projected_minimum)}."
    return text
