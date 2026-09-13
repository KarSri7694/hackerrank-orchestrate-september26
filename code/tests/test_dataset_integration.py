from __future__ import annotations

import sys
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buywait.application import Application  # noqa: E402
from buywait.adapters.evidence_cache import JsonEvidenceCache  # noqa: E402
from buywait.core import _effective_events, _optimized_changes, _option_order, legal_changes, simulate, solve, validate_changes  # noqa: E402
from buywait.domain import (EvidenceFact, EventStatus, FinancialEvent, FinancialProfile,
                            FinancialRequest, PaymentMethod, RequestContext, SpendingChange)  # noqa: E402
from buywait.reconciliation import reconcile_events  # noqa: E402
from buywait.recurrence import detect_recurrences, recurring_event_ids  # noqa: E402
from buywait.evidence import EvidenceService  # noqa: E402


FX = type("FX", (), {"convert": lambda self, amount, source, target, on_date: amount})()


def event(event_id, day, *, amount="100", status=EventStatus.SETTLED, linked=None, description="rent", flexibility="fixed"):
    return FinancialEvent(event_id, "u", "expense", description, "rent", "debit", Decimal(amount), "USD", day, day if status == EventStatus.SETTLED else None, status, linked, flexibility, Decimal("10") if flexibility != "fixed" else None)


def context(events, *, protected=frozenset(), reducible=frozenset(), stoppable=frozenset()):
    request = FinancialRequest("r", "u", date(2025, 1, 1), "purchase", Decimal("1"), date(2025, 2, 1), False, "text")
    profile = FinancialProfile("u", "USD", Decimal("1000"), Decimal("500"), ("debt",), protected, reducible, stoppable, frozenset({PaymentMethod.FULL}), None)
    return RequestContext(request, profile, tuple(events), (), ())


class DatasetIntegrationTests(unittest.TestCase):
    def test_terminal_payroll_identity_closes_prior_payroll_stream(self):
        from buywait.recurrence import stream_key
        self.assertEqual(stream_key("salary", "credit", "Payroll credit"),
                         stream_key("salary", "credit", "Final employer payroll"))
        self.assertNotEqual(stream_key("salary", "credit", "Cobalt payroll"),
                            stream_key("salary", "credit", "Riverline payroll"))

    def test_category_only_activity_does_not_become_a_recurring_stream(self):
        exact = [FinancialEvent(
            f"subscription-{month}", "u", "expense", "Dining subscription", "dining", "debit", Decimal("20"), "USD",
            date(2024, month, 15), date(2024, month, 15), EventStatus.SETTLED, None, "fixed", None
        ) for month in (1, 2, 3)]
        variable = [FinancialEvent(
            f"meal-{index}", "u", "expense", description, "dining", "debit", Decimal("30"), "USD", day, day,
            EventStatus.SETTLED, None, "fixed", None
        ) for index, (description, day) in enumerate((
            ("Cafe", date(2024, 1, 2)), ("Takeaway", date(2024, 1, 9)),
            ("Restaurant", date(2024, 1, 16)), ("Bakery", date(2024, 1, 23)),
        ), 1)]
        streams = detect_recurrences(tuple(exact + variable))
        self.assertEqual(len(streams), 1)
        self.assertEqual(streams[0].representative.description, "Dining subscription")

    def test_recurrence_does_not_merge_same_label_across_currencies(self):
        rows = []
        for currency, amount in (("USD", "100"), ("EUR", "200")):
            rows.extend(FinancialEvent(
                f"{currency}-{month}", "u", "expense", "Shared service", "subscription", "debit",
                Decimal(amount), currency, date(2024, month, 15), date(2024, month, 15),
                EventStatus.SETTLED, None, "fixed", None
            ) for month in (1, 2, 3))
        streams = detect_recurrences(tuple(rows))
        self.assertEqual({stream.representative.currency for stream in streams}, {"USD", "EUR"})
        self.assertEqual({stream.representative.amount for stream in streams}, {Decimal("100"), Decimal("200")})

    def test_internal_transfer_rows_cannot_seed_a_recurring_cash_stream(self):
        transfers = tuple(FinancialEvent(
            f"transfer-{month}", "u", "internal_transfer", "Savings move", "transfer", direction, Decimal("100"),
            "USD", date(2024, month, 15), date(2024, month, 15), EventStatus.SETTLED, None, "fixed", None
        ) for month in (1, 2, 3) for direction in ("debit", "credit"))
        self.assertFalse(detect_recurrences(transfers))
        request = FinancialRequest("r", "u", date(2024, 4, 1), "purchase", Decimal("1"), date(2024, 4, 30), False, "test")
        ctx = RequestContext(request, context(()).profile, transfers, (), ())
        self.assertFalse(any(item[3].event_type == "internal_transfer" for item in _effective_events(ctx, FX)))

    def test_explicit_event_on_one_shared_category_stream_does_not_suppress_other_stream(self):
        history = []
        for source in ("Employer A", "Employer B"):
            for month in (10, 11, 12):
                history.append(FinancialEvent(
                    f"{source}-{month}", "u", "income", source, "salary", "credit", Decimal("100"),
                    "USD", date(2024, month, 15), date(2024, month, 15), EventStatus.SETTLED,
                    None, "fixed", None))
        explicit_a = FinancialEvent(
            "a-next", "u", "income", "Employer A", "salary", "credit", Decimal("100"), "USD",
            date(2025, 1, 15), date(2025, 1, 15), EventStatus.SCHEDULED, None, "fixed", None)
        req = FinancialRequest("r", "u", date(2025, 1, 1), "purchase", Decimal("1"), date(2025, 2, 1), False, "text")
        ctx = RequestContext(req, context(()).profile, tuple(history + [explicit_a]), (), ())
        rows = _effective_events(ctx, FX)
        b_rows = [row for row in rows if row[3].description == "Employer B" and row[0] == date(2025, 1, 15)]
        self.assertEqual(len(b_rows), 1)

    def test_generic_confirmed_event_does_not_duplicate_an_unambiguous_stream(self):
        history = [FinancialEvent(
            f"salary-{month}", "u", "income", "Primary household salary", "salary", "credit", Decimal("100"),
            "USD", date(2024, month, 15), date(2024, month, 15), EventStatus.SETTLED, None, "fixed", None
        ) for month in (10, 11, 12)]
        confirmed = FinancialEvent(
            "next-salary", "u", "income", "Next confirmed salary", "salary", "credit", Decimal("100"),
            "USD", date(2025, 1, 15), date(2025, 1, 15), EventStatus.SCHEDULED, None, "fixed", None
        )
        req = FinancialRequest("r", "u", date(2025, 1, 1), "purchase", Decimal("1"), date(2025, 2, 1), False, "text")
        ctx = RequestContext(req, context(()).profile, tuple(history + [confirmed]), (), ())
        rows = [row for row in _effective_events(ctx, FX) if row[0] == date(2025, 1, 15)]
        self.assertEqual(len(rows), 1)

    def test_generic_event_uses_unique_amount_match_when_salary_streams_are_distinct(self):
        history = []
        for source, amount in (("Employer A", "100"), ("Employer B", "200")):
            history.extend(FinancialEvent(
                f"{source}-{month}", "u", "income", source, "salary", "credit", Decimal(amount), "USD",
                date(2024, month, 15), date(2024, month, 15), EventStatus.SETTLED, None, "fixed", None
            ) for month in (10, 11, 12))
        confirmed = FinancialEvent(
            "next-salary", "u", "income", "Next confirmed salary", "salary", "credit", Decimal("100"),
            "USD", date(2025, 1, 15), date(2025, 1, 15), EventStatus.SCHEDULED, None, "fixed", None
        )
        req = FinancialRequest("r", "u", date(2025, 1, 1), "purchase", Decimal("1"), date(2025, 2, 1), False, "text")
        ctx = RequestContext(req, context(()).profile, tuple(history + [confirmed]), (), ())
        rows = [row for row in _effective_events(ctx, FX) if row[0] == date(2025, 1, 15)]
        self.assertEqual({row[3].description for row in rows}, {"Next confirmed salary", "Employer B"})

    def test_monthly_recurrence_keeps_month_end_anchor_after_february(self):
        history = tuple(FinancialEvent(
            f"salary-{index}", "u", "income", "Month-end salary", "salary", "credit", Decimal("100"), "USD",
            day, day, EventStatus.SETTLED, None, "fixed", None
        ) for index, day in enumerate((date(2023, 11, 30), date(2023, 12, 31), date(2024, 1, 31)), 1))
        stream = detect_recurrences(history)[0]
        self.assertEqual(stream.next_date(date(2024, 1, 31)), date(2024, 2, 29))
        self.assertEqual(stream.next_date(stream.next_date(date(2024, 1, 31))), date(2024, 3, 31))

    def test_recurrence_uses_settlement_date_for_cash_cadence(self):
        history = tuple(FinancialEvent(
            f"salary-{index}", "u", "income", "salary", "salary", "credit", Decimal("100"), "USD",
            date(2024, month, 1), date(2024, month, 5), EventStatus.SETTLED, None, "fixed", None
        ) for index, month in enumerate((10, 11, 12), 1))
        stream = detect_recurrences(history)[0]
        self.assertEqual(stream.cadence_days, 31)
        self.assertEqual(stream.next_date(date(2024, 12, 5)), date(2025, 1, 5))

    def test_recurring_amount_uses_conservative_bound_for_variable_streams(self):
        debits = tuple(FinancialEvent(
            f"bill-{index}", "u", "expense", "variable bill", "utilities", "debit", Decimal(amount), "USD",
            date(2024, month, 15), date(2024, month, 15), EventStatus.SETTLED, None, "fixed", None
        ) for index, (month, amount) in enumerate(((1, "100"), (2, "140"), (3, "120")), 1))
        credits = tuple(FinancialEvent(
            f"pay-{index}", "u", "income", "salary", "salary", "credit", Decimal(amount), "USD",
            date(2024, month, 15), date(2024, month, 15), EventStatus.SETTLED, None, "fixed", None
        ) for index, (month, amount) in enumerate(((1, "1000"), (2, "900"), (3, "950")), 1))
        streams = detect_recurrences(debits + credits)
        self.assertEqual(next(stream for stream in streams if stream.representative.category == "utilities").representative.amount, Decimal("140"))
        self.assertEqual(next(stream for stream in streams if stream.representative.category == "salary").representative.amount, Decimal("900"))

    def test_varied_non_salary_credits_do_not_create_inferred_income_stream(self):
        events = tuple(FinancialEvent(
            f"payout-{index}", "u", "income", description, "payout", "credit", Decimal("100"), "USD",
            day, day, EventStatus.SETTLED, None, "fixed", None
        ) for index, (description, day) in enumerate((
            ("Driver payout", date(2024, 1, 1)), ("Task payout", date(2024, 1, 15)),
            ("Delivery payout", date(2024, 1, 29)), ("Driver payout", date(2024, 2, 12)),
        ), 1))
        self.assertFalse(any(stream.representative.category == "payout"
                             for stream in detect_recurrences(events)))

    def test_salary_category_gig_payouts_do_not_masquerade_as_payroll_recurrence(self):
        events = tuple(FinancialEvent(
            f"payout-{index}", "u", "income", description, "salary", "credit", Decimal("100"), "USD",
            day, day, EventStatus.SETTLED, None, "fixed", None
        ) for index, (description, day) in enumerate((
            ("Driver platform payout", date(2024, 1, 1)), ("Task marketplace payout", date(2024, 1, 15)),
            ("Delivery platform payout", date(2024, 1, 29)), ("Driver platform payout", date(2024, 2, 12)),
        ), 1))
        self.assertFalse(detect_recurrences(events))

    def test_pending_stream_status_suppresses_inferred_future_credit(self):
        history = tuple(FinancialEvent(
            f"payout-{month}", "u", "income", "Driver payout", "payout", "credit", Decimal("100"), "USD",
            date(2024, month, 1), date(2024, month, 1), EventStatus.SETTLED, None, "fixed", None
        ) for month in (10, 11, 12))
        pending = FinancialEvent("pending", "u", "stream_status", "Driver payout", "payout", "credit", None, "USD",
                                 date(2025, 1, 1), None, EventStatus.PENDING, None, "fixed", None)
        self.assertFalse(detect_recurrences(history + (pending,)))

    def test_settled_occurrence_after_pending_stream_reenables_recurrence(self):
        history = tuple(FinancialEvent(
            f"payout-{month}", "u", "income", "Driver payout", "payout", "credit", Decimal("100"), "USD",
            date(2024, month, 1), date(2024, month, 1), EventStatus.SETTLED, None, "fixed", None
        ) for month in (10, 11, 12))
        pending = FinancialEvent("pending", "u", "stream_status", "Driver payout", "payout", "credit", None, "USD",
                                 date(2025, 1, 1), None, EventStatus.PENDING, None, "fixed", None)
        settled = FinancialEvent("new", "u", "income", "Driver payout", "payout", "credit", Decimal("100"), "USD",
                                 date(2025, 1, 1), date(2025, 1, 1), EventStatus.SETTLED, None, "fixed", None)
        self.assertTrue(detect_recurrences(history + (pending, settled)))

    def test_pending_stream_evidence_without_date_uses_sent_date_for_status_marker(self):
        fact = EvidenceFact(
            "pending-evidence", "stream_update", category="payout", direction="credit",
            status=EventStatus.PENDING, stream_source="Driver payout",
            sent_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        resolved = reconcile_events((), (fact,))
        self.assertEqual(len(resolved), 1)
        self.assertEqual((resolved[0].event_type, resolved[0].event_date, resolved[0].status),
                         ("stream_status", date(2025, 1, 1), EventStatus.PENDING))

    def test_pending_debit_with_past_settlement_remains_reserved_at_request_boundary(self):
        pending = FinancialEvent(
            "pending-old", "u", "expense", "Card authorization", "shopping", "debit", Decimal("40"),
            "USD", date(2024, 12, 20), date(2024, 12, 21), EventStatus.PENDING, None, "fixed", None)
        req = FinancialRequest("r", "u", date(2025, 1, 1), "purchase", Decimal("1"), date(2025, 2, 1), False, "text")
        ctx = RequestContext(req, context(()).profile, (pending,), (), ())
        rows = _effective_events(ctx, FX)
        self.assertEqual([(day, amount) for day, amount, _, _ in rows], [(date(2025, 1, 1), Decimal("-40"))])

    def test_mcp_inspect_missing_evidence_is_structured_not_found(self):
        from tools import server
        app = Application(Path(__file__).resolve().parents[2] / "dataset")
        server.configure_application(app)
        result = server.inspect_evidence.fn("does_not_exist")
        self.assertFalse(result["found"])
        self.assertEqual(result["facts"], [])

    def test_mcp_calculator_uses_exact_decimal_arithmetic(self):
        from tools import server
        self.assertEqual(server.calculator.fn("add", "0.10", "0.20")["result"], "0.30")
        self.assertEqual(server.calculator.fn("percentage", "250", "12.5")["result"], "31.25")
        self.assertEqual(server.calculator.fn("divide", "1", "0"), {"ok": False, "error": "division_by_zero"})
        self.assertEqual(server.calculator.fn("add", "not-a-number", "1"), {"ok": False, "error": "invalid_decimal_operand"})
        self.assertEqual(server.calculator.fn("unknown", "1", "2"), {"ok": False, "error": "unsupported_operation"})
        self.assertEqual(server.calculator.fn("add", "NaN", "1"), {"ok": False, "error": "invalid_decimal_operand"})

    def test_all_application_use_cases_share_registered_evidence(self):
        class Extractor:
            model = "integration-extractor-v2"
            def __init__(self):
                self.calls = 0

            def extract(self, reference):
                self.calls += 1
                if reference.evidence_id != "message_18":
                    return ()
                return (EvidenceFact("message_18", "stream_update", amount=Decimal("30780000"), currency="IDR", effective_date=date(2025, 8, 15), status=EventStatus.SCHEDULED, category="salary", direction="credit", recurring=False),)

        extractor = Extractor()
        app = Application(Path(__file__).resolve().parents[2] / "dataset", evidence_extractor=extractor,
                          evidence_cache_path=Path(tempfile.mkdtemp()) / "facts.json")
        ctx = app._context("request_26")
        self.assertTrue(any(item.event_id.startswith("evidence:message_18") for item in ctx.events))
        app.decision("request_26")
        app.simulate_plan("request_26", ())
        app.optimize_spending("request_26", ())
        self.assertEqual(extractor.calls, 1)

    def test_image_reference_also_produces_structured_facts(self):
        class Extractor:
            def extract(self, reference):
                if reference.evidence_id == "image_06":
                    return (EvidenceFact("image_06", "image_amount", reference.related_event_id, Decimal("123"), "INR"),)
                return ()

        app = Application(Path(__file__).resolve().parents[2] / "dataset", evidence_extractor=Extractor(),
                          evidence_cache_path=Path(tempfile.mkdtemp()) / "facts.json")
        ctx = app._context("request_33")
        related = app.repository.evidence("image_06").related_event_id
        self.assertEqual(next(item for item in ctx.events if item.event_id == related).amount, Decimal("123"))

    def test_message_metadata_and_compact_case(self):
        root = Path(__file__).resolve().parents[2] / "dataset"
        app = Application(root)
        ref = app.repository.evidence("message_01")
        self.assertEqual(ref.source_type, "employer")
        self.assertIsNotNone(ref.sent_at)
        case = app.case("request_26")
        self.assertIn("request_text", case)
        self.assertIn("financial_priorities", case)
        for item in case["evidence_refs"]:
            self.assertNotIn("text", item)
            self.assertNotIn("image_path", item)

    def test_stream_fact_without_event_id_creates_recurring_override(self):
        history = [event(str(index), date(2024, month, 15), amount="100") for index, month in enumerate((10, 11, 12), 1)]
        fact = EvidenceFact("m", "stream_update", amount=Decimal("150"), currency="USD", effective_date=date(2025, 1, 15), status=EventStatus.SCHEDULED, category="rent", direction="debit", description="rent", recurring=True, recurrence_days=31, stream_source="rent")
        resolved = reconcile_events(history, (fact,))
        self.assertTrue(any(item.event_id.startswith("evidence:") for item in resolved))
        timeline = _effective_events(replace(context(resolved), evidence_facts=()), FX)
        self.assertEqual(timeline[0][1], Decimal("-150"))

    def test_stream_termination_is_scoped_to_its_stable_identity(self):
        employer_a = [FinancialEvent(f"a-{month}", "u", "income", "Employer A", "salary", "credit", Decimal("100"), "USD", date(2024, month, 15), date(2024, month, 15), EventStatus.SETTLED, None, "fixed", None) for month in (10, 11, 12)]
        employer_b = [FinancialEvent(f"b-{month}", "u", "income", "Employer B", "salary", "credit", Decimal("200"), "USD", date(2024, month, 20), date(2024, month, 20), EventStatus.SETTLED, None, "fixed", None) for month in (10, 11, 12)]
        fact = EvidenceFact("message_a", "terminate", category="salary", direction="credit", effective_date=date(2025, 1, 1), stream_source="Employer A")
        resolved = reconcile_events(tuple(employer_a + employer_b), (fact,))
        self.assertTrue(all(item.status == EventStatus.CANCELLED for item in resolved if item.event_id.startswith("a-")))
        self.assertTrue(all(item.status == EventStatus.SETTLED for item in resolved if item.event_id.startswith("b-")))

    def test_internal_transfer_pair_is_excluded_without_cancelling_one_side(self):
        debit = FinancialEvent("transfer_debit", "u", "transfer", "Move savings", "transfer", "debit", Decimal("100"), "USD", date(2025, 1, 2), date(2025, 1, 2), EventStatus.SETTLED, None, "fixed", None)
        credit = FinancialEvent("transfer_credit", "u", "transfer", "Move savings", "transfer", "credit", Decimal("100"), "USD", date(2025, 1, 2), date(2025, 1, 2), EventStatus.SETTLED, "transfer_debit", "fixed", None)
        resolved = reconcile_events((debit, credit), (EvidenceFact("message_transfer", "internal_transfer", "transfer_debit"),))
        self.assertEqual({item.event_type for item in resolved}, {"internal_transfer"})
        self.assertEqual({item.status for item in resolved}, {EventStatus.SETTLED})
        self.assertEqual(simulate(context(resolved), FX).ending_balance, Decimal("1000"))

    def test_request_level_internal_transfer_finds_one_unique_pair(self):
        debit = FinancialEvent("debit", "u", "transfer", "Savings move", "transfer", "debit", Decimal("100"), "USD", date(2025, 1, 2), date(2025, 1, 2), EventStatus.SETTLED, None, "fixed", None)
        credit = FinancialEvent("credit", "u", "transfer", "Savings move", "transfer", "credit", Decimal("100"), "USD", date(2025, 1, 2), date(2025, 1, 2), EventStatus.SETTLED, None, "fixed", None)
        fact = EvidenceFact("request_message", "internal_transfer", amount=Decimal("100"), currency="USD", effective_date=date(2025, 1, 2))
        resolved = reconcile_events((debit, credit), (fact,))
        self.assertEqual({item.event_type for item in resolved}, {"internal_transfer"})

    def test_request_level_internal_transfer_rejects_ambiguous_pair(self):
        debit = FinancialEvent("debit", "u", "transfer", "Move", "transfer", "debit", Decimal("100"), "USD", date(2025, 1, 2), date(2025, 1, 2), EventStatus.SETTLED, None, "fixed", None)
        credits = tuple(FinancialEvent(f"credit-{index}", "u", "transfer", "Move", "transfer", "credit", Decimal("100"), "USD", date(2025, 1, 2), date(2025, 1, 2), EventStatus.SETTLED, None, "fixed", None) for index in (1, 2))
        resolved = reconcile_events((debit,) + credits, (EvidenceFact("m", "internal_transfer", amount=Decimal("100"), currency="USD", effective_date=date(2025, 1, 2)),))
        self.assertTrue(all(item.event_type == "transfer" for item in resolved))

    def test_request_level_internal_transfer_leaves_one_sided_event_unchanged(self):
        debit = FinancialEvent("debit", "u", "transfer", "Move", "transfer", "debit", Decimal("100"), "USD", date(2025, 1, 2), date(2025, 1, 2), EventStatus.SETTLED, None, "fixed", None)
        resolved = reconcile_events((debit,), (EvidenceFact("m", "internal_transfer", amount=Decimal("100"), currency="USD", effective_date=date(2025, 1, 2)),))
        self.assertEqual(resolved[0].event_type, "transfer")

    def test_critic_can_correct_a_request_level_internal_transfer(self):
        debit = FinancialEvent("debit", "u", "transfer", "Move", "transfer", "debit", Decimal("100"), "USD", date(2025, 1, 1), date(2025, 1, 1), EventStatus.SETTLED, None, "fixed", None)
        credit = FinancialEvent("credit", "u", "transfer", "Move", "transfer", "credit", Decimal("100"), "USD", date(2025, 1, 1), date(2025, 1, 1), EventStatus.SETTLED, None, "fixed", None)
        request = FinancialRequest("r", "u", date(2025, 1, 1), "purchase", Decimal("1"), date(2025, 1, 2), False, "transfer")
        profile = FinancialProfile("u", "USD", Decimal("501"), Decimal("500"), (), frozenset(), frozenset(), frozenset(), frozenset({PaymentMethod.FULL}), None)
        from buywait.domain import EvidenceReference
        raw = RequestContext(request, profile, (debit, credit), (), (EvidenceReference("message_transfer", "bank", "u", request_id="r", text="Transfer"),))
        app = Application.__new__(Application)
        app.repository = type("Repo", (), {"context": lambda _self, _id: raw, "evidence": lambda _self, _id: raw.evidence_refs[0]})()
        app.fx = FX
        app.evidence_service = type("Evidence", (), {"facts_for": lambda _self, _refs: (), "inspect": lambda _self, _id: ()})()
        app.decision = lambda _id: solve(raw, FX)
        fact = EvidenceFact("message_transfer", "internal_transfer")
        result = app.evaluate_correction("r", (fact,), ("message_transfer",))
        self.assertTrue(result.accepted)
        self.assertEqual(result.decision.recommended_payment_method, PaymentMethod.FULL)

    def test_explicit_new_recurring_stream_forecasts_but_one_time_income_does_not(self):
        recurring = EvidenceFact("m_salary", "stream_update", amount=Decimal("100"), currency="USD", effective_date=date(2025, 1, 15), status=EventStatus.SCHEDULED, category="salary", direction="credit", description="New Employer", recurring=True, recurrence_days=30, stream_source="New Employer")
        one_time = EvidenceFact("m_bonus", "one_time", amount=Decimal("75"), currency="USD", effective_date=date(2025, 1, 16), status=EventStatus.SCHEDULED, category="bonus", direction="credit", description="Referral bonus", recurring=False, stream_source="Referral bonus")
        resolved = reconcile_events((), (recurring, one_time))
        ctx = context(resolved)
        dates = [row[0] for row in _effective_events(ctx, FX) if row[3].event_id.startswith("evidence:")]
        self.assertIn(date(2025, 2, 15), dates)
        self.assertEqual(dates.count(date(2025, 1, 16)), 1)
        self.assertEqual(sum(row[3].event_id == "evidence:m_bonus:1" for row in _effective_events(ctx, FX)), 1)

    def test_new_evidenced_stream_in_fixed_rate_currency_survives_validation(self):
        from buywait.domain import EvidenceReference
        raw = context(())
        raw = replace(raw, evidence_refs=(EvidenceReference("new_stream", "employer", "u", text="Salary EUR 100 on 2025-01-15"),))
        app = Application.__new__(Application)
        app.fx = type("FX", (), {"rates": {(date(2025, 1, 15), "EUR", "USD"): Decimal("1.1")}})()
        fact = EvidenceFact("new_stream", "stream_update", amount=Decimal("100"), currency="EUR", effective_date=date(2025, 1, 15), category="salary", direction="credit", recurring=True, recurrence_days=30, stream_source="New employer")
        self.assertEqual(app._validated_facts(raw, (fact,)), (fact,))

    def test_incomplete_stream_update_does_not_seed_recurrence(self):
        fact = EvidenceFact("m", "stream_update", amount=Decimal("100"), currency="USD",
                            effective_date=date(2025, 1, 15), category="salary", direction="credit")
        resolved = reconcile_events((), (fact,))
        self.assertEqual(detect_recurrences(resolved), ())
        rows = _effective_events(context(resolved), FX)
        self.assertEqual([row[0] for row in rows], [date(2025, 1, 15)])

    def test_evidence_validation_rejects_non_finite_and_conflicting_recurrence_values(self):
        from buywait.domain import EvidenceReference
        app = Application.__new__(Application)
        raw = replace(context(()), evidence_refs=(EvidenceReference("m", "employer", "u", text="pay"),))
        facts = (
            EvidenceFact("m", "stream_update", amount=Decimal("NaN"), currency="USD",
                         effective_date=date(2025, 1, 15), category="salary", direction="credit"),
            EvidenceFact("m", "stream_update", amount=Decimal("100"), currency="USD",
                         effective_date=date(2025, 1, 15), category="salary", direction="credit",
                         recurring=False, recurrence_days=30),
        )
        self.assertEqual(app._validated_facts(raw, facts), ())

    def test_aggregate_salary_update_replaces_old_salary_forecasts_only_after_effective_date(self):
        salaries = []
        for source, amount, day in (("Employer A", "100", 10), ("Employer B", "200", 12)):
            for month in (10, 11, 12):
                salaries.append(FinancialEvent(f"{source}-{month}", "u", "income", source, "salary", "credit", Decimal(amount), "USD", date(2024, month, day), date(2024, month, day), EventStatus.SETTLED, None, "fixed", None))
        stipend = [FinancialEvent(f"stipend-{month}", "u", "income", "Grant", "stipend", "credit", Decimal("50"), "USD", date(2024, month, 5), date(2024, month, 5), EventStatus.SETTLED, None, "fixed", None) for month in (10, 11, 12)]
        aggregate = EvidenceFact("message_total", "aggregate_stream_update", amount=Decimal("250"), currency="USD", effective_date=date(2025, 2, 15), status=EventStatus.SCHEDULED, category="salary", direction="credit", recurring=True, recurrence_days=30, stream_source="household")
        resolved = reconcile_events(tuple(salaries + stipend), (aggregate,))
        req = FinancialRequest("r", "u", date(2025, 1, 1), "purchase", Decimal("1"), date(2025, 4, 1), False, "text")
        ctx = RequestContext(req, context(()).profile, resolved, (), ())
        rows = _effective_events(ctx, FX)
        salary_on_effective = [amount for day, amount, _, item in rows if day == date(2025, 2, 15) and item.category == "salary"]
        self.assertEqual(salary_on_effective, [Decimal("250")])
        self.assertTrue(any(day > date(2025, 2, 15) and item.category == "stipend" for day, _, _, item in rows))
        self.assertFalse(any(day >= date(2025, 2, 15) and item.category == "salary" and amount != Decimal("250") for day, amount, _, item in rows))

    def test_persistent_evidence_cache_prevents_reextraction_in_fresh_service(self):
        from buywait.domain import EvidenceReference
        reference = EvidenceReference("message_cache", "bank", "u", text="Amount 12 USD")
        repository = type("Repo", (), {"evidence": lambda _self, _id: reference})()
        class Extractor:
            model = "cache-test"
            def __init__(self): self.calls = 0
            def extract(self, _ref):
                self.calls += 1
                return (EvidenceFact("message_cache", "amend", amount=Decimal("12"), currency="USD"),)
        with tempfile.TemporaryDirectory() as temporary:
            cache = JsonEvidenceCache(Path(temporary) / "facts.json")
            first = Extractor()
            self.assertEqual(EvidenceService(repository, first, cache).inspect("message_cache")[0].amount, Decimal("12"))
            second = Extractor()
            self.assertEqual(EvidenceService(repository, second, JsonEvidenceCache(Path(temporary) / "facts.json")).inspect("message_cache")[0].amount, Decimal("12"))
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 0)

    def test_malformed_persistent_cache_falls_back_without_crashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "facts.json"
            path.write_text("{not valid json", encoding="utf-8")
            cache = JsonEvidenceCache(path)
            self.assertIsNone(cache.get("anything"))
            cache.put("anything", ())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"anything": []})
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_malformed_cached_fact_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "facts.json"
            path.write_text(json.dumps({"bad": [{"evidence_id": "bad", "effect": "amend", "amount": "not-a-number"}]}), encoding="utf-8")
            self.assertIsNone(JsonEvidenceCache(path).get("bad"))

    def test_empty_ai_extraction_is_not_reused_as_persistent_success(self):
        from buywait.domain import EvidenceReference
        reference = EvidenceReference("message_empty", "bank", "u", text="A fact may be extracted")
        repository = type("Repo", (), {"evidence": lambda _self, _id: reference})()
        class Extractor:
            model = "ai-empty-test"
            cache_empty_results = False
            def __init__(self): self.calls = 0
            def extract(self, _ref):
                self.calls += 1
                return ()
        with tempfile.TemporaryDirectory() as temporary:
            first = Extractor()
            EvidenceService(repository, first, JsonEvidenceCache(Path(temporary) / "facts.json")).inspect("message_empty")
            second = Extractor()
            EvidenceService(repository, second, JsonEvidenceCache(Path(temporary) / "facts.json")).inspect("message_empty")
        self.assertEqual((first.calls, second.calls), (1, 1))

    def test_newer_same_source_then_settled_then_safer_precedence(self):
        base = event("e", date(2025, 1, 1))
        older = EvidenceFact("a", "amend", "e", Decimal("120"), sent_at=datetime(2025, 1, 1, tzinfo=timezone.utc), source_type="bank")
        newer = EvidenceFact("b", "amend", "e", Decimal("130"), sent_at=datetime(2025, 1, 2, tzinfo=timezone.utc), source_type="bank")
        settled = EvidenceFact("c", "amend", "e", Decimal("125"), status=EventStatus.SETTLED, source_type="merchant")
        self.assertEqual(reconcile_events((base,), (older, newer, settled))[0].amount, Decimal("125"))

    def test_linked_lifecycle_and_exact_duplicates_count_once(self):
        pending = event("pending", date(2024, 12, 20), status=EventStatus.PENDING)
        settled = event("settled", date(2025, 1, 2), status=EventStatus.SETTLED, linked="pending")
        duplicate = replace(settled, event_id="duplicate")
        resolved = reconcile_events((pending, settled, duplicate), ())
        self.assertEqual([item.event_id for item in resolved], ["settled"])

    def test_past_pending_debit_is_reserved_on_request_date(self):
        pending = event("pending", date(2024, 12, 1), status=EventStatus.PENDING)
        result = simulate(context((pending,)), FX)
        self.assertEqual(result.ending_balance, Decimal("900"))

    def test_recurrence_drives_forecast_and_change_eligibility(self):
        history = [event(str(index), date(2024, month, 15), flexibility="reducible") for index, month in enumerate((10, 11, 12), 1)]
        ctx = context(history, reducible=frozenset({"rent"}))
        recurring = recurring_event_ids(ctx.events)
        self.assertEqual(recurring, frozenset({"3"}))
        self.assertEqual({change.event_id for group in legal_changes(ctx, list(ctx.events)) for change in group}, recurring)

    def test_strict_change_validation(self):
        history = [event(str(index), date(2024, month, 15), flexibility="reducible") for index, month in enumerate((10, 11, 12), 1)]
        ctx = context(history, protected=frozenset({"rent"}), reducible=frozenset({"rent"}))
        with self.assertRaises(ValueError):
            validate_changes(ctx, (SpendingChange("3", "reduce_to", Decimal("50")),))
        ctx = context(history, reducible=frozenset({"rent"}))
        with self.assertRaises(ValueError):
            validate_changes(ctx, (SpendingChange("3", "reduce_to", Decimal("5")),))
        with self.assertRaises(ValueError):
            validate_changes(ctx, (SpendingChange("3", "stop"),))

    def test_payment_option_suffix_is_numeric(self):
        self.assertLess(_option_order("payment_option_9"), _option_order("payment_option_10"))

    def test_multi_change_optimizer_relaxes_reductions_after_combining(self):
        from buywait.domain import Payment
        events = []
        for category in ("dining", "streaming"):
            for index, month in enumerate((10, 11, 12), 1):
                events.append(FinancialEvent(f"{category}-{index}", "u", "expense", category, category, "debit", Decimal("100"), "USD", date(2024, month, 15), date(2024, month, 15), EventStatus.SETTLED, None, "reducible", Decimal("0")))
        request = FinancialRequest("r", "u", date(2025, 1, 1), "purchase", Decimal("400"), date(2025, 2, 1), False, "text")
        profile = FinancialProfile("u", "USD", Decimal("1000"), Decimal("500"), (), frozenset(), frozenset({"dining", "streaming"}), frozenset(), frozenset({PaymentMethod.FULL}), None)
        ctx = RequestContext(request, profile, tuple(events), (), ())
        payments = (Payment(request.request_date, request.requested_amount),)
        changes = _optimized_changes(ctx, FX, payments)
        self.assertEqual(len(changes or ()), 2)
        self.assertTrue(any((change.new_amount or Decimal("0")) > 0 for change in changes))
        self.assertTrue(simulate(ctx, FX, payments, tuple(changes)).safe)


if __name__ == "__main__":
    unittest.main()
