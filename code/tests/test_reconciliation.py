from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buywait.core import baseline, simulate, solve  # noqa: E402
from buywait.domain import (EventStatus, EvidenceFact, FinancialEvent, FinancialProfile,
                            FinancialRequest, PaymentMethod, RequestContext)  # noqa: E402
from buywait.reconciliation import reconcile_events  # noqa: E402


def event(event_id="e1", amount=Decimal("100"), direction="debit", status=EventStatus.SCHEDULED):
    return FinancialEvent(event_id, "u1", "expense", "test", "groceries", direction, amount, "USD", date(2025, 1, 1), date(2025, 1, 1), status, None, "fixed", None)


def context(events):
    request = FinancialRequest("r1", "u1", date(2025, 1, 1), "purchase", Decimal("1"), date(2025, 1, 30), False, "test")
    profile = FinancialProfile("u1", "USD", Decimal("1000"), Decimal("500"), (), frozenset(), frozenset(), frozenset(), frozenset({PaymentMethod.FULL}), None)
    return RequestContext(request, profile, tuple(events), (), ())


class ReconciliationTests(unittest.TestCase):
    def test_cancellation_wins(self):
        resolved = reconcile_events((event(),), (EvidenceFact("m", "cancel", "e1"),))
        self.assertEqual(resolved[0].status, EventStatus.CANCELLED)

    def test_amount_and_date_amendment(self):
        fact = EvidenceFact("m", "amend", "e1", Decimal("275"), "USD", date(2025, 1, 7), EventStatus.SETTLED)
        resolved = reconcile_events((event(),), (fact,))[0]
        self.assertEqual(resolved.amount, Decimal("275"))
        self.assertEqual(resolved.event_date, date(2025, 1, 7))
        self.assertEqual(resolved.settlement_date, date(2025, 1, 7))
        self.assertEqual(resolved.status, EventStatus.SETTLED)

    def test_missing_image_amount_is_restored(self):
        raw = event(amount=None)
        fact = EvidenceFact("image_01", "confirm", "e1", Decimal("725"), "USD")
        self.assertEqual(reconcile_events((raw,), (fact,))[0].amount, Decimal("725"))

    def test_pending_credit_is_ignored_but_pending_debit_reserved(self):
        credit = event("credit", Decimal("100"), "credit", EventStatus.PENDING)
        debit = event("debit", Decimal("100"), "debit", EventStatus.PENDING)
        result = simulate(context((credit, debit)), type("FX", (), {"convert": lambda self, amount, source, target, on_date: amount})())
        self.assertEqual(result.ending_balance, Decimal("900"))

    def test_failed_event_has_no_cash_impact(self):
        failed = event(status=EventStatus.FAILED)
        result = simulate(context((failed,)), type("FX", (), {"convert": lambda self, amount, source, target, on_date: amount})())
        self.assertEqual(result.ending_balance, Decimal("1000"))

    def test_conflicting_cancel_and_amendment_uses_cancellation(self):
        facts = (EvidenceFact("m1", "amend", "e1", Decimal("900")), EvidenceFact("m2", "cancel", "e1"))
        resolved = reconcile_events((event(),), facts)[0]
        self.assertEqual(resolved.status, EventStatus.CANCELLED)
        self.assertEqual(resolved.amount, Decimal("100"))

    def test_downstream_baseline_and_decision_receive_resolved_context(self):
        raw = event(amount=None)
        ctx = context((raw,))
        ctx = ctx.__class__(ctx.request, ctx.profile, ctx.events, ctx.payment_options, ctx.evidence_refs,
                            (EvidenceFact("image_01", "confirm", "e1", Decimal("25"), "USD"),))
        fx = type("FX", (), {"convert": lambda self, amount, source, target, on_date: amount})()
        safe, _ = baseline(ctx, fx)
        decision = solve(ctx, fx)
        self.assertEqual(safe, Decimal("1"))
        self.assertEqual(decision.trace["minimum_balance"], "500")


if __name__ == "__main__":
    unittest.main()
