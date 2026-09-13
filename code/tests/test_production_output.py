from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from agent.usage import UsageTracker  # noqa: E402
from agent.evidence_extractor import OpenAIEvidenceExtractor, _json_payload  # noqa: E402
from agent.schemas import parse_json_object, response_text  # noqa: E402
from buywait.application import Application  # noqa: E402
from buywait.adapters.csv_repository import _assert_event_links, _assert_evidence_event_links, _assert_unique  # noqa: E402
from buywait.domain import (AffordabilityStatus, DecisionCore, EvidenceFact, EvidenceReference,
                            EventStatus, FinancialEvent, Payment, PaymentMethod, SpendingChange)  # noqa: E402
from buywait.output import OUTPUT_COLUMNS, decision_row, write_output  # noqa: E402
from buywait.presentation import decision_explanation  # noqa: E402
from buywait.evidence_validation import text_supports_amount, text_supports_date  # noqa: E402
from evaluation.output_validation import OutputValidationError, _validate_explanation, validate_output  # noqa: E402


class ProductionOutputTests(unittest.TestCase):
    def test_fail_open_payroll_fallback_requires_explicit_amount_and_date(self):
        from buywait.evidence import _explicit_payroll_fallback
        explicit = EvidenceReference("pay", "employer", "u", text="Payroll update: salary is EUR 1661 from 2026-01-15.")
        fact = _explicit_payroll_fallback(explicit)[0]
        self.assertEqual((fact.amount, fact.currency, fact.effective_date, fact.category, fact.direction),
                         (Decimal("1661"), "EUR", date(2026, 1, 15), "salary", "credit"))
        self.assertEqual(_explicit_payroll_fallback(EvidenceReference("vague", "employer", "u", text="Payroll is changing soon.")), ())


    def test_ingestion_rejects_unknown_or_cross_user_lifecycle_links(self):
        event = FinancialEvent("e1", "u1", "expense", "bill", "rent", "debit", Decimal("1"), "USD", date(2026, 1, 1), date(2026, 1, 1), EventStatus.SETTLED, None, "fixed", None)
        with self.assertRaises(ValueError):
            _assert_event_links({"e1": event.__class__(**{**event.__dict__, "linked_event_id": "missing"})})
        other = event.__class__(**{**event.__dict__, "event_id": "e2", "user_id": "u2"})
        linked = event.__class__(**{**event.__dict__, "linked_event_id": "e2"})
        with self.assertRaises(ValueError):
            _assert_event_links({"e1": linked, "e2": other})

    def test_ingestion_rejects_evidence_links_to_missing_or_other_user_events(self):
        event = FinancialEvent("e1", "u1", "expense", "bill", "rent", "debit", Decimal("1"), "USD", date(2026, 1, 1), date(2026, 1, 1), EventStatus.SETTLED, None, "fixed", None)
        with self.assertRaises(ValueError):
            _assert_evidence_event_links([{"related_event_id": "missing", "user_id": "u1"}], {"e1": event}, "messages.csv")
        with self.assertRaises(ValueError):
            _assert_evidence_event_links([{"related_event_id": "e1", "user_id": "u2"}], {"e1": event}, "images.csv")
    def test_dataset_adapter_rejects_duplicate_or_blank_identifiers(self):
        with self.assertRaises(ValueError):
            _assert_unique([{"id": "x"}, {"id": "x"}], "id", "test.csv")
        with self.assertRaises(ValueError):
            _assert_unique([{"id": ""}], "id", "test.csv")

    def test_date_support_rejects_model_invented_date_but_accepts_numeric_multilingual_date(self):
        self.assertFalse(text_supports_date("Gaji berikutnya sudah dikonfirmasi.", date(2024, 1, 15)))
        self.assertTrue(text_supports_date("Gaji berlaku mulai 2025-08-15.", date(2025, 8, 15)))
        self.assertTrue(text_supports_date("Pembayaran dikonfirmasi pada 15/08/2025.", date(2025, 8, 15)))
        self.assertFalse(text_supports_amount("Gaji berikutnya sudah dikonfirmasi.", Decimal("1422.85"), "EUR"))
        self.assertTrue(text_supports_amount("Gaji menjadi IDR 42750000.", Decimal("42750000"), "IDR"))
        self.assertTrue(text_supports_amount("€ 12.50", Decimal("12.50"), "EUR"))
        self.assertTrue(text_supports_amount("₹ 83.05", Decimal("83.05"), "INR"))

    def test_csv_serializer_has_exact_schema_and_plain_values(self):
        decision = DecisionCore(
            "request_1", Decimal("1E+3"), AffordabilityStatus.WITH_PLAN,
            PaymentMethod.PARTIAL, (Payment(date(2026, 1, 2), Decimal("10.50")), Payment(date(2026, 1, 3), Decimal("989.50"))),
            None, (SpendingChange("event_1", "reduce_to", Decimal("5.25")),), {}, "grounded explanation",
        )
        row = decision_row(decision)
        self.assertEqual(tuple(row), OUTPUT_COLUMNS)
        self.assertEqual(row["amount_safe_to_pay"], "1000")
        self.assertEqual(row["payment_plan"], "2026-01-02:10.50|2026-01-03:989.50")
        self.assertEqual(row["spending_changes_needed"], "reduce_to:event_1:5.25")
        self.assertEqual(row["earliest_date_for_full_payment"], "")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "output.csv"
            write_output(path, (decision,))
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), OUTPUT_COLUMNS)
                self.assertEqual(next(reader)["payment_plan"], row["payment_plan"])

    def test_explanation_is_grounded_and_deterministic(self):
        decision = DecisionCore("r", Decimal("50"), AffordabilityStatus.NOW, PaymentMethod.FULL,
                                (Payment(date(2026, 1, 2), Decimal("100")),), date(2026, 1, 2), (), {})
        text = decision_explanation(decision, currency="USD", minimum_balance=Decimal("20"), projected_minimum=Decimal("35"))
        self.assertIn("Pay USD 100 today", text)
        self.assertIn("USD 20", text)
        self.assertIn("USD 35", text)

    def test_output_explanation_rejects_plain_contradiction(self):
        with self.assertRaises(OutputValidationError):
            _validate_explanation("Do not proceed; this is not affordable.", PaymentMethod.FULL)
        with self.assertRaises(OutputValidationError):
            _validate_explanation("Pay it later.", PaymentMethod.WAIT)

    def test_case_exposes_raw_resolved_events_facts_and_forecast(self):
        class Extractor:
            model = "case-context-test"
            def extract(self, reference, stream_context=()):
                return (EvidenceFact(reference.evidence_id, "amend", reference.related_event_id, amount=Decimal("1"), currency="USD"),) if reference.related_event_id else ()
        with tempfile.TemporaryDirectory() as temporary:
            app = Application(ROOT / "dataset", evidence_extractor=Extractor(), evidence_cache_path=Path(temporary) / "facts.json")
            case = app.case("request_26")
        self.assertTrue(case["raw_financial_events"])
        self.assertTrue(case["resolved_financial_events"])
        self.assertIn("forecast_diagnostics", case)
        self.assertIn("extracted_evidence", case)
        self.assertEqual(set(case["raw_financial_events"][0]), {"event_id", "event_type", "description", "category", "direction", "amount", "currency", "event_date", "settlement_date", "status", "linked_event_id", "flexibility", "minimum_allowed_amount", "recurrence_days"})

    def test_future_message_is_not_in_request_context(self):
        app = Application(ROOT / "dataset")
        request = app.repository.requests["request_26"]
        future = EvidenceReference("future_message", "bank", request.user_id, sent_at=datetime.combine(request.request_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc), text="future")
        app.repository.evidence_by_user[request.user_id].append(future)
        self.assertNotIn("future_message", {ref.evidence_id for ref in app.repository.context("request_26").evidence_refs})

    def test_extractor_gets_compact_stream_context_and_invalid_fact_is_dropped(self):
        received = []
        class Extractor:
            model = "stream-context-test"
            def extract(self, reference, stream_context=()):
                received.append(stream_context)
                if reference.evidence_id == "image_06":
                    return (EvidenceFact("image_06", "image_amount", "unknown", Decimal("12"), "INR"),)
                return ()
        with tempfile.TemporaryDirectory() as temporary:
            app = Application(ROOT / "dataset", evidence_extractor=Extractor(), evidence_cache_path=Path(temporary) / "facts.json")
            ctx = app._context("request_33")
        self.assertTrue(received)
        self.assertTrue(all("event_id" in event for group in received for event in group))
        self.assertFalse(any(event.event_id == "unknown" for event in ctx.events))

    def test_missing_image_cannot_apply_cached_or_model_fact(self):
        app = Application(ROOT / "dataset")
        request = app.repository.requests["request_26"]
        related = app.repository.events_by_user[request.user_id][0]
        missing = EvidenceReference(
            "missing_image", "image", request.user_id, request_id="request_26",
            related_event_id=related.event_id, image_path=str(ROOT / "dataset" / "media" / "images" / "does_not_exist.png"),
        )
        app.repository.evidence_by_user[request.user_id].append(missing)
        raw = app.repository.context("request_26")
        facts = (EvidenceFact("missing_image", "image_amount", related.event_id, Decimal("12"), "USD"),)
        self.assertEqual(app._validated_facts(raw, facts), ())

    def test_linked_evidence_context_does_not_mix_same_category_other_streams(self):
        app = Application(ROOT / "dataset")
        ref = app.repository.evidence("message_14")
        context = app._stream_context_for_evidence(ref)
        self.assertTrue(context)
        related = app.repository.event(ref.related_event_id)
        self.assertEqual({item["description"] for item in context}, {related.description})

    def test_usage_report_counts_actual_calls_and_cache_hits_separately(self):
        class Usage:
            input_tokens, output_tokens, total_tokens = 10, 5, 15
        class Response:
            usage = Usage()
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "usage.md"
            tracker = UsageTracker("test", Decimal("1"), Decimal("2"))
            tracker.record("critic", "model", Response())
            tracker.report(report, 3, cache_hits=2)
            text = report.read_text(encoding="utf-8")
        self.assertIn("Actual model calls: 1", text)
        self.assertIn("Evidence cache hits: 2", text)
        self.assertIn("critic", text)

    def test_older_same_content_cache_entry_is_reused_and_normalized(self):
        reference = EvidenceReference("image_cache", "image", "u", related_event_id="event_1")
        class Repository:
            def evidence(self, _evidence_id):
                return reference
        class Extractor:
            model = "legacy-cache-test"
            def __init__(self): self.calls = 0
            def extract(self, _reference):
                self.calls += 1
                return ()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "facts.json"
            from buywait.adapters.evidence_cache import JsonEvidenceCache
            legacy = JsonEvidenceCache(path)
            content_hash = hashlib.sha256(b"").hexdigest()
            legacy.put(f"image_cache:legacy-cache-test:evidence-facts-v21:{content_hash}",
                       (EvidenceFact("image_cache", "stream_update", related_event_id="event_1", amount=Decimal("12"), currency="USD"),))
            extractor = Extractor()
            service = __import__("buywait.evidence", fromlist=["EvidenceService"]).EvidenceService(
                Repository(), extractor, JsonEvidenceCache(path))
            facts = service.inspect("image_cache")
        self.assertEqual(extractor.calls, 0)
        self.assertEqual(facts[0].effect, "image_amount")
        self.assertEqual(service.cache_hits, 1)

    def test_null_extractor_never_reads_persistent_model_cache(self):
        reference = EvidenceReference("cached_model_fact", "employer", "u", text="Salary update")
        class Repository:
            def evidence(self, _evidence_id):
                return reference
        with tempfile.TemporaryDirectory() as temporary:
            from buywait.adapters.evidence_cache import JsonEvidenceCache
            cache = JsonEvidenceCache(Path(temporary) / "facts.json")
            cache.put("cached_model_fact:NullEvidenceExtractor:evidence-facts-v22:stale", (
                EvidenceFact("cached_model_fact", "cancel", amount=Decimal("999")),
            ))
            service = __import__("buywait.evidence", fromlist=["EvidenceService"]).EvidenceService(
                Repository(), None, JsonEvidenceCache(Path(temporary) / "facts.json"))
            self.assertEqual(service.inspect("cached_model_fact"), ())
            self.assertEqual(service.inspect("cached_model_fact"), ())
            self.assertEqual(service.cache_hits, 0)

    def test_cached_text_fact_with_unsupported_date_is_rejected_and_reextracted(self):
        reference = EvidenceReference("dated_cache", "employer", "u", text="Gaji berikutnya sudah dikonfirmasi.")
        class Repository:
            def evidence(self, _evidence_id):
                return reference
        class Extractor:
            model = "dated-cache-test"
            cache_empty_results = False
            def __init__(self): self.calls = 0
            def extract(self, _reference):
                self.calls += 1
                return ()
        with tempfile.TemporaryDirectory() as temporary:
            from buywait.adapters.evidence_cache import JsonEvidenceCache
            cache = JsonEvidenceCache(Path(temporary) / "facts.json")
            key = "dated_cache:dated-cache-test:evidence-facts-v22:" + hashlib.sha256(reference.text.encode()).hexdigest()
            cache.put(key, (EvidenceFact("dated_cache", "stream_update", amount=Decimal("10"), currency="USD",
                                         effective_date=date(2024, 1, 15)),))
            extractor = Extractor()
            service = __import__("buywait.evidence", fromlist=["EvidenceService"]).EvidenceService(
                Repository(), extractor, JsonEvidenceCache(Path(temporary) / "facts.json"))
            facts = service.inspect("dated_cache")
        self.assertEqual(facts, ())
        self.assertEqual(extractor.calls, 1)

    def test_text_amount_validation_accepts_common_thousands_separators(self):
        from buywait.evidence_validation import text_supports_amount
        self.assertTrue(text_supports_amount("Pay EUR 1,037.52", Decimal("1037.52"), "EUR"))
        self.assertTrue(text_supports_amount("Pagado EUR 1.037,52", Decimal("1037.52"), "EUR"))
        self.assertTrue(text_supports_amount("Pay EUR 1,037", Decimal("1037"), "EUR"))

    def test_text_date_validation_accepts_numeric_east_asian_date_form(self):
        from buywait.evidence_validation import text_supports_date
        self.assertTrue(text_supports_date("工资将在 2025年8月15日 发放", date(2025, 8, 15)))
        self.assertTrue(text_supports_date("工资将在 2025年08月15日 发放", date(2025, 8, 15)))

    def test_extractor_amount_validation_accepts_zar_symbol(self):
        class Client:
            model = "test"
            def chat_create(self, **_kwargs):
                return type("Response", (), {"output_text": '{"facts":[{"effect":"stream_update","amount":"10000","currency":"ZAR","effective_date":null,"category":"salary","direction":"credit","recurring":false}]}'} )()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        facts = extractor.extract(EvidenceReference("message", "employer", "u", text="Salary R 10,000"))
        self.assertEqual(facts[0].amount, Decimal("10000"))

    def test_extracted_fact_cannot_override_authoritative_evidence_metadata(self):
        reference = EvidenceReference("metadata_fact", "employer", "u", text="Salary update",
                                      sent_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
        class Repository:
            def evidence(self, _evidence_id):
                return reference
        class Extractor:
            def extract(self, _reference):
                return (EvidenceFact("metadata_fact", "stream_update", source_type="bank",
                                     sent_at=datetime(2030, 1, 1, tzinfo=timezone.utc)),)
        service = __import__("buywait.evidence", fromlist=["EvidenceService"]).EvidenceService(
            Repository(), Extractor())
        fact = service.inspect("metadata_fact")[0]
        self.assertEqual(fact.source_type, "employer")
        self.assertEqual(fact.sent_at, reference.sent_at)

    def test_registered_correction_cannot_override_authoritative_evidence_metadata(self):
        reference = EvidenceReference("registered_metadata", "employer", "u",
                                      sent_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
        class Repository:
            def evidence(self, _evidence_id):
                return reference
        service = __import__("buywait.evidence", fromlist=["EvidenceService"]).EvidenceService(
            Repository(), extractor=None)
        fact = EvidenceFact("registered_metadata", "stream_update", source_type="bank",
                             sent_at=datetime(2030, 1, 1, tzinfo=timezone.utc))
        service.register((fact,))
        registered = service.inspect("registered_metadata")[0]
        self.assertEqual(registered.source_type, "employer")
        self.assertEqual(registered.sent_at, reference.sent_at)

    def test_invalidated_nonempty_cache_reextracts_even_when_empty_results_are_cached(self):
        reference = EvidenceReference("invalidated_cache", "employer", "u", text="Salary update")
        class Repository:
            def evidence(self, _evidence_id):
                return reference
        class Extractor:
            model = "invalidated-cache-test"
            cache_empty_results = True
            def __init__(self): self.calls = 0
            def extract(self, _reference):
                self.calls += 1
                return (EvidenceFact("invalidated_cache", "stream_update", amount=Decimal("12"), currency="USD"),)
        with tempfile.TemporaryDirectory() as temporary:
            from buywait.adapters.evidence_cache import JsonEvidenceCache
            cache = JsonEvidenceCache(Path(temporary) / "facts.json")
            key = "invalidated_cache:invalidated-cache-test:evidence-facts-v22:" + hashlib.sha256(reference.text.encode()).hexdigest()
            cache.put(key, (EvidenceFact("invalidated_cache", "stream_update", amount=Decimal("12"), currency="USD", effective_date=date(2099, 1, 1)),))
            extractor = Extractor()
            service = __import__("buywait.evidence", fromlist=["EvidenceService"]).EvidenceService(
                Repository(), extractor, JsonEvidenceCache(Path(temporary) / "facts.json"))
            facts = service.inspect("invalidated_cache")
        self.assertEqual(extractor.calls, 1)
        self.assertEqual(facts[0].amount, Decimal("12"))

    def test_usage_tracker_accepts_chat_completion_token_names(self):
        class Usage:
            prompt_tokens, completion_tokens, total_tokens = 7, 3, 10
        response = type("Response", (), {"usage": Usage()})()
        tracker = UsageTracker("test")
        tracker.record("evidence_extraction", "model", response)
        self.assertEqual((tracker.records[0].input_tokens, tracker.records[0].output_tokens, tracker.records[0].total_tokens), (7, 3, 10))

    def test_evidence_parser_accepts_final_json_after_thinking_trace(self):
        class Client:
            model = "test"
            def chat_create(self, **_kwargs):
                return type("Response", (), {"output_text": '<think>{"scratch":"private"}</think>{"facts":[{"effect":"amend","related_event_id":"event_1","amount":"12","currency":"USD","effective_date":null,"status":null,"category":null,"direction":null,"description":null,"recurring":null,"recurrence_days":null,"confidence":1,"stream_source":null}]}'})()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        facts = extractor.extract(EvidenceReference("message", "bank", "u", related_event_id="event_1", text="12 USD"))
        self.assertEqual(facts[0].amount, Decimal("12"))

    def test_evidence_json_parser_keeps_outer_facts_object(self):
        payload = _json_payload('{"facts":[{"effect":"amend","amount":"12"}]}')
        self.assertEqual(payload["facts"][0]["effect"], "amend")

    def test_extractor_uses_unique_stream_context_when_multilingual_fact_omits_labels(self):
        class Client:
            model = "test"
            def chat_create(self, **_kwargs):
                return type("Response", (), {"output_text": '{"facts":[{"effect":"amend","amount":"42750000","currency":"IDR","effective_date":"2025-08-15"}]}'} )()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        facts = extractor.extract(EvidenceReference("message", "employer", "u", text="Gaji berubah: 42.750.000 IDR pada 2025-08-15"),
                                  ({"category": "salary", "direction": "credit", "currency": "IDR", "description": "Payroll credit"},))
        self.assertEqual((facts[0].category, facts[0].direction, facts[0].stream_source),
                         ("salary", "credit", "Payroll credit"))

    def test_extractor_bridges_named_source_to_unique_generic_historical_stream(self):
        class Client:
            model = "test"
            def chat_create(self, **_kwargs):
                return type("Response", (), {"output_text": '{"facts":[{"effect":"stream_update","amount":"42","currency":"IDR","effective_date":"2025-08-15","category":"salary","direction":"credit","stream_source":"Cobalt Systems","recurring":true,"recurrence_days":30}]}'} )()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        facts = extractor.extract(EvidenceReference("message", "employer", "u", text="pay changed to 42 IDR on 2025-08-15"),
                                  ({"category": "salary", "direction": "credit", "currency": "IDR", "description": "Payroll credit"},))
        self.assertEqual(facts[0].stream_source, "Payroll credit")

    def test_extractor_normalizes_string_recurrence_cadence_to_integer(self):
        class Client:
            model = "test"
            def chat_create(self, **_kwargs):
                return type("Response", (), {"output_text": '{"facts":[{"effect":"stream_update","amount":"42","currency":"USD","effective_date":"2025-08-15","category":"salary","direction":"credit","recurring":true,"recurrence_days":"30"}]}'} )()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        facts = extractor.extract(EvidenceReference("message", "employer", "u", text="Salary 42 USD on 2025-08-15"))
        self.assertEqual(facts[0].recurrence_days, 30)
        self.assertIsInstance(facts[0].recurrence_days, int)

    def test_extractor_parses_european_thousands_and_decimal_amount(self):
        class Client:
            model = "test"
            def chat_create(self, **_kwargs):
                return type("Response", (), {"output_text": '{"facts":[{"effect":"stream_update","amount":"1.037,52","currency":"EUR","effective_date":"2025-08-15","category":"salary","direction":"credit","recurring":false}]}'} )()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        facts = extractor.extract(EvidenceReference("message", "employer", "u", text="Gehalt EUR 1.037,52 am 15.08.2025"))
        self.assertEqual(facts[0].amount, Decimal("1037.52"))

    def test_extractor_accepts_integral_decimal_cadence_but_rejects_fractional(self):
        class Client:
            model = "test"
            def __init__(self, value): self.value = value
            def chat_create(self, **_kwargs):
                return type("Response", (), {"output_text":
                    '{"facts":[{"effect":"stream_update","amount":"42","currency":"USD","effective_date":"2025-08-15","category":"salary","direction":"credit","recurring":true,"recurrence_days":"' + self.value + '"}]}'} )()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client("30.0"), "test", "low", True
        facts = extractor.extract(EvidenceReference("message", "employer", "u", text="Salary 42 USD on 2025-08-15"))
        self.assertEqual(facts[0].recurrence_days, 30)
        extractor.client = Client("30.5")
        self.assertEqual(extractor.extract(EvidenceReference("message", "employer", "u", text="Salary 42 USD on 2025-08-15")), ())

    def test_extractor_repairs_effect_token_misplaced_as_stream_category(self):
        class Client:
            model = "test"
            def chat_create(self, **_kwargs):
                return type("Response", (), {"output_text": '{"facts":[{"effect":"stream_update","amount":"1422.85","currency":"EUR","category":"amend","direction":"debit","recurring":true,"recurrence_days":30}]}'} )()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        facts = extractor.extract(
            EvidenceReference("message", "employer", "u", text="Next salary is reduced to EUR 1422.85."),
            ({"category": "salary", "direction": "credit", "currency": "EUR", "description": "Payroll credit"},),
        )
        self.assertEqual((facts[0].category, facts[0].direction), ("salary", "credit"))

    def test_extractor_rejects_model_placeholder_zero_amount(self):
        class Client:
            model = "test"
            def chat_create(self, **_kwargs):
                return type("Response", (), {"output_text": '{"facts":[{"effect":"stream_update","amount":"0","currency":"USD","effective_date":"2026-01-15","category":"salary","direction":"credit","recurring":true,"recurrence_days":30}]}'} )()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        facts = extractor.extract(EvidenceReference("message", "employer", "u", text="regular payroll confirmed"))
        self.assertEqual(facts, ())

    def test_extractor_rejects_nonzero_amount_not_present_in_text(self):
        class Client:
            model = "test"
            def chat_create(self, **_kwargs):
                return type("Response", (), {"output_text": '{"facts":[{"effect":"stream_update","amount":"999","currency":"USD","effective_date":"2026-01-15","category":"salary","direction":"credit","recurring":false}]}'} )()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        self.assertEqual(extractor.extract(EvidenceReference("message", "employer", "u", text="Regular payroll confirmed")), ())

    def test_extractor_rejects_date_not_present_in_text(self):
        class Client:
            model = "test"
            def chat_create(self, **_kwargs):
                return type("Response", (), {"output_text": '{"facts":[{"effect":"stream_update","amount":null,"currency":"USD","effective_date":"2026-01-15","category":"salary","direction":"credit","recurring":false}]}'} )()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        self.assertEqual(extractor.extract(EvidenceReference("message", "employer", "u", text="Regular payroll confirmed")), ())

    def test_linked_image_amount_is_normalized_to_image_amount_effect(self):
        class Client:
            model = "test"
            def chat_create(self, **_kwargs):
                return type("Response", (), {"output_text": '{"facts":[{"effect":"stream_update","amount":"123","currency":"USD","related_event_id":"event_1"}]}'} )()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        facts = extractor.extract(EvidenceReference("image", "image", "u", related_event_id="event_1"))
        self.assertEqual(facts[0].effect, "image_amount")

    def test_response_text_reads_output_message_content_when_output_text_is_empty(self):
        content = type("Content", (), {"type": "output_text", "text": '{"facts":[]}'})()
        response = type("Response", (), {"output_text": "", "output": [type("Message", (), {"content": [content]})()]})()
        self.assertEqual(response_text(response), '{"facts":[]}')

    def test_response_text_falls_back_to_reasoning_content_when_chat_content_empty(self):
        message = type("Message", (), {"content": "", "reasoning_content": '{"facts":[]}'})()
        choice = type("Choice", (), {"message": message})()
        response = type("Response", (), {"output_text": "", "choices": [choice], "output": []})()
        self.assertEqual(response_text(response), '{"facts":[]}')

    def test_evidence_extractor_passes_separate_longer_timeout(self):
        class Client:
            model = "test"
            def __init__(self): self.kwargs = None
            def chat_create(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"output_text": '{"facts":[]}'})()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        extractor.evidence_timeout = 123
        extractor.extract(EvidenceReference("message", "bank", "u", text="no financial fact"))
        self.assertEqual(extractor.client.kwargs["timeout"], 123)

    def test_auto_evidence_provider_failure_falls_back_to_empty_facts(self):
        reference = EvidenceReference("auto_failure", "employer", "u", text="Salary update")
        class Repository:
            def evidence(self, _evidence_id):
                return reference
        class Extractor:
            model = "auto-failure"
            fail_open = True
            def extract(self, _reference):
                raise RuntimeError("provider unavailable")
        from buywait.evidence import EvidenceService
        service = EvidenceService(Repository(), Extractor())
        self.assertEqual(service.inspect("auto_failure"), ())
        self.assertEqual(service.extraction_calls, 1)

    def test_enabled_evidence_provider_failure_remains_fail_closed(self):
        reference = EvidenceReference("enabled_failure", "employer", "u", text="Salary update")
        class Repository:
            def evidence(self, _evidence_id):
                return reference
        class Extractor:
            model = "enabled-failure"
            fail_open = False
            def extract(self, _reference):
                raise RuntimeError("provider unavailable")
        from buywait.evidence import EvidenceService
        service = EvidenceService(Repository(), Extractor())
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            service.inspect("enabled_failure")

    def test_parse_json_object_extracts_final_object_after_thinking(self):
        self.assertEqual(parse_json_object('scratch {not-json}\n```json\n{"agree":true}\n```'), {"agree": True})

    def test_json_parsers_prefer_schema_object_over_valid_scratch_json(self):
        self.assertEqual(parse_json_object('{"scratch":"private"} then {"agree":true}'), {"agree": True})
        self.assertEqual(_json_payload('{"scratch":"private"} then {"facts":[]}'), {"facts": []})

    def test_extractor_retries_when_thinking_response_has_no_facts(self):
        class Client:
            model = "test"
            def __init__(self):
                self.calls = 0
            def chat_create(self, **_kwargs):
                self.calls += 1
                raw = '{"facts":[]}' if self.calls == 1 else '{"facts":[{"effect":"amend","amount":"12","currency":"USD","effective_date":null}]}'
                return type("Response", (), {"output_text": raw})()
        extractor = OpenAIEvidenceExtractor.__new__(OpenAIEvidenceExtractor)
        extractor.client, extractor.model, extractor.image_detail, extractor.vision_enabled = Client(), "test", "low", True
        facts = extractor.extract(EvidenceReference("message", "bank", "u", related_event_id="event_1", text="12 USD"))
        self.assertEqual(facts[0].amount, Decimal("12"))
        self.assertEqual(extractor.client.calls, 2)

    def test_output_validator_rejects_wrong_schema_before_reading_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            path.write_text("wrong\nvalue\n", encoding="utf-8")
            with self.assertRaises(OutputValidationError):
                validate_output(path, None)

    def test_output_validator_rejects_safe_amount_not_from_reconciled_baseline(self):
        app = Application(ROOT / "dataset")
        request_id = app.repository.request_ids()[0]
        decision = app.finalized_decision(request_id)
        row = decision_row(decision)
        expected = Decimal(row["amount_safe_to_pay"])
        requested = app.repository.requests[request_id].requested_amount
        row["amount_safe_to_pay"] = str(expected + 1 if expected < requested else expected - 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaises(OutputValidationError):
                validate_output(path, app)

    def test_production_runner_emits_row_progress_and_checkpoint(self):
        # The CLI implementation writes the temporary checkpoint after every
        # row; this contract test protects the immediate-observability path
        # without invoking the full dataset or an AI provider.
        source = (ROOT / "code" / "main.py").read_text(encoding="utf-8")
        self.assertIn("write_output(temporary_path, decisions)", source)
        self.assertIn("flush=True", source)


if __name__ == "__main__":
    unittest.main()
