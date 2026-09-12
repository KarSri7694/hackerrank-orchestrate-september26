from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buywait.core import expand_option, solve  # noqa: E402
from buywait.domain import (EventStatus, FinancialProfile, FinancialRequest,
                            PaymentMethod, PaymentOption, RequestContext)  # noqa: E402


def ctx(methods, *, balance="1000", minimum="500", requested="400", deadline=date(2025, 2, 28), options=()):
    request = FinancialRequest("r", "u", date(2025, 1, 1), "purchase", Decimal(requested), deadline, True, "test")
    profile = FinancialProfile("u", "USD", Decimal(balance), Decimal(minimum), (), frozenset(), frozenset(), frozenset(), frozenset(methods), None)
    return RequestContext(request, profile, (), tuple(options), ())


class PlanSelectionTests(unittest.TestCase):
    def test_schedule_expansion_is_exact_and_rejects_missing_frequency(self):
        option = PaymentOption("o", "r", PaymentMethod.INSTALLMENTS, Decimal("100"), 3, date(2025, 1, 3), 28, Decimal("0"), Decimal("300"))
        self.assertEqual([p.date for p in expand_option(option)], [date(2025, 1, 3), date(2025, 1, 31), date(2025, 2, 28)])
        malformed = PaymentOption("bad", "r", PaymentMethod.INSTALLMENTS, Decimal("100"), 3, date(2025, 1, 3), None, Decimal("0"), Decimal("300"))
        self.assertEqual(expand_option(malformed), ())

    def test_installment_month_limit_uses_calendar_months(self):
        option = PaymentOption("o", "r", PaymentMethod.INSTALLMENTS, Decimal("100"), 3, date(2025, 1, 15), 29, Decimal("0"), Decimal("300"))
        context = ctx({PaymentMethod.INSTALLMENTS}, deadline=date(2025, 3, 15), options=(option,))
        context = context.__class__(context.request, FinancialProfile("u", "USD", Decimal("1000"), Decimal("500"), (), frozenset(), frozenset(), frozenset(), frozenset({PaymentMethod.INSTALLMENTS}), 2), context.events, context.payment_options, context.evidence_refs)
        self.assertEqual(solve(context, type("FX", (), {"convert": lambda self, amount, source, target, on_date: amount})()).recommended_payment_method, PaymentMethod.INSTALLMENTS)

    def test_unsafe_full_payment_is_not_a_candidate(self):
        from buywait.domain import FinancialEvent
        event = FinancialEvent("e", "u", "expense", "bill", "rent", "debit", Decimal("450"), "USD", date(2025, 1, 1), date(2025, 1, 1), EventStatus.SCHEDULED, None, "fixed", None)
        context = ctx({PaymentMethod.FULL}, balance="1000", minimum="500", requested="400")
        context = context.__class__(context.request, context.profile, (event,), context.payment_options, context.evidence_refs)
        decision = solve(context, type("FX", (), {"convert": lambda self, amount, source, target, on_date: amount})())
        self.assertEqual(decision.recommended_payment_method, PaymentMethod.NOT_RECOMMENDED)

    def test_wait_requires_full_payment_preference(self):
        option = PaymentOption("o", "r", PaymentMethod.INSTALLMENTS, Decimal("100"), 1, date(2025, 1, 5), None, Decimal("0"), Decimal("100"))
        context = ctx({PaymentMethod.INSTALLMENTS}, requested="400", options=(option,))
        decision = solve(context, type("FX", (), {"convert": lambda self, amount, source, target, on_date: amount})())
        self.assertNotEqual(decision.recommended_payment_method, PaymentMethod.WAIT)


if __name__ == "__main__":
    unittest.main()
