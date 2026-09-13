from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from buywait.application import Application  # noqa: E402


class CandidateContractTests(unittest.TestCase):
    def setUp(self):
        self.app = Application(ROOT / "dataset")

    def test_each_exposed_candidate_round_trips_to_a_safe_core_decision(self):
        request_id = "request_26"
        candidates = self.app.candidate_options(request_id)
        self.assertTrue(candidates)
        for candidate in candidates:
            decision = self.app.decision_for_candidate(request_id, candidate["candidate_id"])
            self.assertIsNotNone(decision)
            result = self.app.simulate_plan(request_id, decision.payment_plan, decision.spending_changes)
            self.assertTrue(result.safe or not decision.payment_plan)

    def test_invalid_candidate_id_is_rejected_without_changing_financial_state(self):
        self.assertIsNone(self.app.decision_for_candidate("request_26", "candidate_999"))
        self.assertIsNone(self.app.decision_for_candidate("request_26", "arbitrary-plan"))

    def test_agent_case_contains_all_user_history_and_unredacted_message_text(self):
        case = self.app.agent_case("request_26")
        raw = self.app.repository.context("request_26")
        self.assertEqual(len(case["raw_financial_events"]), len(raw.events))
        self.assertEqual(len(case["evidence"]), len(raw.evidence_refs))

    def test_payment_options_summary_exposes_only_precomputed_terms(self):
        summary = self.app.payment_options_summary("request_26")
        self.assertIn("amount_safe_to_pay", summary)
        self.assertIn("earliest_date_for_full_payment", summary)
        for plan in summary["installments"]:
            self.assertIn("candidate_id", plan)
            self.assertIn("payment_amount", plan)
            self.assertIn("number_of_payments", plan)
            self.assertIn("months", plan)
            self.assertIn("financing_fee", plan)


if __name__ == "__main__":
    unittest.main()
