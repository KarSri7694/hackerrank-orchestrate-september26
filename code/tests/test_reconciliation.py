from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buywait.core import _effective_events, baseline, simulate, solve  # noqa: E402
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

    def test_irregular_history_is_not_forecast_as_monthly_recurrence(self):
        irregular = (
            event("a", Decimal("10")),
            FinancialEvent("b", "u1", "expense", "test", "groceries", "debit", Decimal("10"), "USD", date(2024, 1, 1), date(2024, 1, 1), EventStatus.SETTLED, None, "fixed", None),
            FinancialEvent("c", "u1", "expense", "test", "groceries", "debit", Decimal("10"), "USD", date(2024, 3, 15), date(2024, 3, 15), EventStatus.SETTLED, None, "fixed", None),
        )
        ctx = context(irregular)
        ctx = ctx.__class__(ctx.request, ctx.profile, tuple(FinancialEvent(e.event_id, e.user_id, e.event_type, e.description, e.category, e.direction, e.amount, e.currency, date(2024, 1, 1) if e.event_id == "a" else e.event_date, e.settlement_date, e.status, e.linked_event_id, e.flexibility, e.minimum_allowed_amount) for e in ctx.events), ctx.payment_options, ctx.evidence_refs)
        fx = type("FX", (), {"convert": lambda self, amount, source, target, on_date: amount})()
        generated = _effective_events(ctx, fx)
        self.assertFalse(any(row[0] > ctx.request.request_date and row[3].category == "groceries" for row in generated))

    def test_monthly_recurrence_uses_calendar_dates_and_explicit_events_win(self):
        history = tuple(FinancialEvent(str(i), "u1", "expense", "rent", "rent", "debit", Decimal("10"), "USD", date(2024, m, 15), date(2024, m, 15), EventStatus.SETTLED, None, "fixed", None) for i, m in enumerate((1, 2, 3), 1))
        explicit = FinancialEvent("future", "u1", "expense", "rent", "rent", "debit", Decimal("10"), "USD", date(2024, 4, 15), date(2024, 4, 15), EventStatus.SCHEDULED, None, "fixed", None)
        req = FinancialRequest("r1", "u1", date(2024, 4, 1), "purchase", Decimal("1"), date(2024, 4, 30), False, "test")
        ctx = RequestContext(req, context(()).profile, history + (explicit,), (), ())
        fx = type("FX", (), {"convert": lambda self, amount, source, target, on_date: amount})()
        generated = _effective_events(ctx, fx)
        self.assertEqual([row[0] for row in generated], [date(2024, 4, 15), date(2024, 5, 15), date(2024, 6, 15)])

    def test_terminal_event_stops_historical_recurrence(self):
        history = tuple(FinancialEvent(str(i), "u1", "income", "Payroll", "salary", "credit", Decimal("100"), "USD", date(2024, m, 15), date(2024, m, 15), EventStatus.SETTLED, None, "fixed", None) for i, m in enumerate((1, 2, 3), 1))
        terminal = FinancialEvent("final", "u1", "income", "Payroll", "salary", "credit", Decimal("100"), "USD", date(2024, 4, 15), date(2024, 4, 15), EventStatus.SETTLED, None, "fixed", None)
        terminal = terminal.__class__(terminal.event_id, terminal.user_id, terminal.event_type, "Final employer payroll", terminal.category, terminal.direction, terminal.amount, terminal.currency, terminal.event_date, terminal.settlement_date, terminal.status, terminal.linked_event_id, terminal.flexibility, terminal.minimum_allowed_amount)
        req = FinancialRequest("r1", "u1", date(2024, 5, 1), "purchase", Decimal("1"), date(2024, 5, 30), False, "test")
        ctx = RequestContext(req, context(()).profile, history + (terminal,), (), ())
        fx = type("FX", (), {"convert": lambda self, amount, source, target, on_date: amount})()
        self.assertFalse(any(row[0] > req.request_date and row[3].category == "salary" for row in _effective_events(ctx, fx)))

    def test_explicit_same_day_event_prevents_inferred_duplicate(self):
        history = tuple(FinancialEvent(str(i), "u1", "income", "Payroll", "salary", "credit", Decimal("100"), "USD", date(2024, m, 15), date(2024, m, 15), EventStatus.SETTLED, None, "fixed", None) for i, m in enumerate((1, 2, 3), 1))
        explicit = FinancialEvent("scheduled", "u1", "income", "Next salary", "salary", "credit", Decimal("100"), "USD", date(2024, 4, 15), date(2024, 4, 15), EventStatus.SCHEDULED, None, "fixed", None)
        req = FinancialRequest("r1", "u1", date(2024, 4, 1), "purchase", Decimal("1"), date(2024, 4, 30), False, "test")
        ctx = RequestContext(req, context(()).profile, history + (explicit,), (), ())
        fx = type("FX", (), {"convert": lambda self, amount, source, target, on_date: amount})()
        rows = [row for row in _effective_events(ctx, fx) if row[0] == date(2024, 4, 15)]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][3].event_id, "scheduled")


if __name__ == "__main__":
    unittest.main()
