from __future__ import annotations

import json
import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from agent.config import AgentConfig  # noqa: E402
from agent.runner import AgentRunner  # noqa: E402
from buywait.domain import (AffordabilityStatus, CorrectionEvaluation, DecisionCore,
                            Payment, PaymentMethod)  # noqa: E402
from tools.server import mcp  # noqa: E402


def decision(method=PaymentMethod.FULL, amount="100"):
    payments = (Payment(date(2025, 1, 1), Decimal(amount)),) if method != PaymentMethod.NOT_RECOMMENDED else ()
    return DecisionCore("r", Decimal("25"), AffordabilityStatus.NOW if payments else AffordabilityStatus.NOT_AFFORDABLE,
                        method, payments, date(2025, 1, 1), (), {})


class Response:
    output = ()
    def __init__(self, payload):
        self.output_text = json.dumps(payload)


class FakeClient:
    model = "fake"
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = 0
    def create(self, **_kwargs):
        self.calls += 1
        return Response(next(self.payloads))


class FakeApplication:
    def __init__(self, decisions, evaluations=()):
        self.decisions = list(decisions)
        self.index = 0
        self.evaluations = iter(evaluations)
        self.applied = []
        self.evaluated = []
    def decision(self, _request_id):
        return self.decisions[self.index]
    def case(self, _request_id):
        return {"request_id": "r", "evidence_refs": [{"evidence_id": "message_1"}]}
    def evaluate_correction(self, request_id, facts, support):
        self.evaluated.append((request_id, facts, support))
        return next(self.evaluations)
    def apply_correction(self, facts):
        self.applied.append(facts)
        self.index = min(self.index + 1, len(self.decisions) - 1)


def agree():
    return {"agree": True, "issue_type": None, "supporting_evidence_ids": [], "correction": None, "summary": "agrees"}


def correction(issue="wrong_stream_termination", effect="terminate", related=None):
    return {
        "agree": False, "issue_type": issue, "supporting_evidence_ids": ["message_1"],
        "correction": {"evidence_id": "message_1", "effect": effect, "related_event_id": related,
                       "amount": None, "currency": None, "effective_date": "2025-01-01",
                       "status": "scheduled", "category": "salary", "direction": "credit",
                       "description": "explicit evidence", "recurring": False, "recurrence_days": None,
                       "flexibility": "fixed", "minimum_allowed_amount": None},
        "summary": "Evidence identifies a state error.",
    }


class AgentCriticTests(unittest.TestCase):
    def runner(self, app, payloads, rounds=2):
        config = AgentConfig(api_key="test", ai_mode="enabled", max_critique_rounds=rounds)
        return AgentRunner(app, mcp, ROOT / "dataset", model_client=FakeClient(payloads), config=config)

    def test_agent_agrees_with_deterministic_result(self):
        app = FakeApplication([decision()])
        result = self.runner(app, [agree()]).run("r")
        self.assertEqual(result.decision.recommended_payment_method, PaymentMethod.FULL)
        self.assertFalse(app.applied)
        self.assertEqual(result.decision.trace["agent_critique"], "agent_agreed")

    def test_terminated_salary_correction_changes_result(self):
        initial, corrected = decision(), decision(PaymentMethod.NOT_RECOMMENDED)
        fact_payload = correction()
        from buywait.domain import EvidenceFact
        accepted = CorrectionEvaluation(True, "verified", corrected, (EvidenceFact("message_1", "terminate", category="salary", direction="credit", effective_date=date(2025, 1, 1)),))
        app = FakeApplication([initial, corrected], [accepted])
        result = self.runner(app, [fact_payload, agree()]).run("r")
        self.assertEqual(result.decision.recommended_payment_method, PaymentMethod.NOT_RECOMMENDED)
        self.assertEqual(app.applied[0][0].effect, "terminate")

    def test_internal_transfer_correction_is_structured(self):
        initial, corrected = decision(), decision(PaymentMethod.WAIT)
        payload = correction("internal_transfer", "internal_transfer", "transfer_1")
        from buywait.domain import EvidenceFact
        accepted = CorrectionEvaluation(True, "verified", corrected, (EvidenceFact("message_1", "internal_transfer", "transfer_1"),))
        app = FakeApplication([initial, corrected], [accepted])
        result = self.runner(app, [payload, agree()]).run("r")
        self.assertEqual(result.decision.recommended_payment_method, PaymentMethod.WAIT)
        self.assertEqual(app.evaluated[0][1][0].effect, "internal_transfer")

    def test_vague_or_unverifiable_critique_is_rejected(self):
        app = FakeApplication([decision()])
        vague = {"agree": False, "issue_type": "incorrect_recurrence", "supporting_evidence_ids": [], "correction": None, "summary": "safer"}
        result = self.runner(app, [vague]).run("r")
        self.assertFalse(app.evaluated)
        self.assertEqual(result.decision.trace["agent_critique"], "rejected_vague_or_missing_correction")

    def test_unsafe_correction_is_rejected(self):
        base = decision()
        app = FakeApplication([base], [CorrectionEvaluation(False, "correction produced an unsafe plan", base)])
        result = self.runner(app, [correction()]).run("r")
        self.assertEqual(result.decision.recommended_payment_method, base.recommended_payment_method)
        self.assertFalse(app.applied)
        self.assertIn("unsafe plan", result.decision.trace["agent_critique"])

    def test_safe_plan_preference_without_fact_correction_cannot_replace_ranking(self):
        app = FakeApplication([decision()])
        result = self.runner(app, [{"agree": False, "issue_type": "incorrect_recurrence", "supporting_evidence_ids": ["message_1"], "correction": None, "summary": "I prefer installments."}]).run("r")
        self.assertEqual(result.decision.recommended_payment_method, PaymentMethod.FULL)
        self.assertFalse(app.applied)

    def test_loop_stops_at_configured_maximum_rounds(self):
        first, second, third = decision(), decision(PaymentMethod.WAIT), decision(PaymentMethod.NOT_RECOMMENDED)
        from buywait.domain import EvidenceFact
        fact = EvidenceFact("message_1", "terminate", category="salary", direction="credit", effective_date=date(2025, 1, 1))
        app = FakeApplication([first, second, third], [CorrectionEvaluation(True, "verified", second, (fact,)), CorrectionEvaluation(True, "verified", third, (fact,))])
        client = FakeClient([correction(), correction(), correction()])
        config = AgentConfig(api_key="test", ai_mode="enabled", max_critique_rounds=2)
        result = AgentRunner(app, mcp, ROOT / "dataset", model_client=client, config=config).run("r")
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(app.applied), 2)
        self.assertEqual(result.decision.recommended_payment_method, third.recommended_payment_method)
        self.assertEqual(result.decision.trace["agent_critique"], "critique_round_limit_reached")


if __name__ == "__main__":
    unittest.main()
