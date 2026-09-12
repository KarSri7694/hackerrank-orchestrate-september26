from __future__ import annotations

from decimal import Decimal


def validate_decision(decision, requested_amount: Decimal) -> None:
    assert Decimal("0") <= decision.amount_safe_to_pay <= requested_amount
    assert len(decision.spending_changes) <= 3
    assert list(decision.payment_plan) == sorted(decision.payment_plan, key=lambda p: p.date)
    if decision.recommended_payment_method.value == "partial_payment":
        assert len(decision.payment_plan) == 2
        assert decision.payment_plan[0].amount == decision.amount_safe_to_pay
        assert sum((p.amount for p in decision.payment_plan), Decimal("0")) == requested_amount
    if decision.affordability_status.value == "not_affordable":
        assert not decision.payment_plan

