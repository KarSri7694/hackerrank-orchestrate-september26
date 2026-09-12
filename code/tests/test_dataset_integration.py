from __future__ import annotations

import sys
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
from buywait.recurrence import recurring_event_ids  # noqa: E402
from buywait.evidence import EvidenceService  # noqa: E402


FX = type("FX", (), {"convert": lambda self, amount, source, target, on_date: amount})()


def event(event_id, day, *, amount="100", status=EventStatus.SETTLED, linked=None, description="rent", flexibility="fixed"):
    return FinancialEvent(event_id, "u", "expense", description, "rent", "debit", Decimal(amount), "USD", day, day if status == EventStatus.SETTLED else None, status, linked, flexibility, Decimal("10") if flexibility != "fixed" else None)


def context(events, *, protected=frozenset(), reducible=frozenset(), stoppable=frozenset()):
    request = FinancialRequest("r", "u", date(2025, 1, 1), "purchase", Decimal("1"), date(2025, 2, 1), False, "text")
    profile = FinancialProfile("u", "USD", Decimal("1000"), Decimal("500"), ("debt",), protected, reducible, stoppable, frozenset({PaymentMethod.FULL}), None)
    return RequestContext(request, profile, tuple(events), (), ())


class DatasetIntegrationTests(unittest.TestCase):
    def test_all_application_use_cases_share_registered_evidence(self):
        class Extractor:
            def __init__(self):
                self.calls = 0

            def extract(self, reference):
                self.calls += 1
                if reference.evidence_id != "message_18":
                    return ()
                return (EvidenceFact("message_18", "stream_update", amount=Decimal("30780000"), currency="IDR", effective_date=date(2025, 8, 15), status=EventStatus.SCHEDULED, category="salary", direction="credit", recurring=False),)

        extractor = Extractor()
        app = Application(Path(__file__).resolve().parents[2] / "dataset", evidence_extractor=extractor)
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
                    return (EvidenceFact("image_06", "image_amount", reference.related_event_id, Decimal("123"), "USD"),)
                return ()

        app = Application(Path(__file__).resolve().parents[2] / "dataset", evidence_extractor=Extractor())
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
