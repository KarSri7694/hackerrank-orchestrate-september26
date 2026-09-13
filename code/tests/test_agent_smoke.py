from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from agent.config import AgentConfig  # noqa: E402
from agent.runner import AgentRunner  # noqa: E402
from buywait.application import Application  # noqa: E402


class Response:
    output = ()

    def __init__(self, payload):
        self.output_text = json.dumps(payload)


class FakeClient:
    model = "fake"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return Response(next(self.responses))


class AgentSmokeTests(unittest.TestCase):
    def setUp(self):
        self.app = Application(ROOT / "dataset")
        self.config = AgentConfig(api_key="test", ai_mode="enabled", max_turns=1)

    def test_models_receive_full_case_and_only_four_read_only_tools(self):
        client = FakeClient([
            {"request_id": "request_26", "candidate_id": "candidate_001", "explanation": "Best option.", "evidence_used": []},
            {"approved": True, "candidate_id": "candidate_001", "explanation": "Verified."},
        ])
        result = AgentRunner(self.app, ROOT / "dataset", model_client=client, config=self.config).run("request_26")
        self.assertTrue(result.used_ai)
        self.assertEqual(result.decision, self.app.candidate_decisions("request_26")[0])
        self.assertEqual([item["name"] for item in client.calls[0]["tools"]],
                         ["inspect_source", "inspect_forecast", "calculate_payment_options", "verify_candidate"])
        case = json.loads(client.calls[0]["input"][0]["content"][0]["text"])
        self.assertIn("raw_financial_events", case)
        self.assertIn("evidence", case)
        self.assertIn("forecast", case)
        self.assertIn("candidates", case)
        critic = json.loads(client.calls[1]["input"][0]["content"][0]["text"])
        self.assertIn("decision_model_output", critic)
        self.assertIn("case", critic)

    def test_invalid_critic_candidate_keeps_valid_decision_selection(self):
        client = FakeClient([
            {"request_id": "request_26", "candidate_id": "candidate_001", "explanation": "Best option.", "evidence_used": []},
            {"approved": False, "candidate_id": "not_a_candidate", "explanation": "No."},
        ])
        result = AgentRunner(self.app, ROOT / "dataset", model_client=client, config=self.config).run("request_26")
        self.assertEqual(result.decision, self.app.candidate_decisions("request_26")[0])

    def test_direct_tools_are_read_only(self):
        runner = AgentRunner(self.app, ROOT / "dataset", config=self.config)
        event_id = self.app.repository.context("request_26").events[0].event_id
        self.assertTrue(runner._dispatch("request_26", "inspect_source", {"source_type": "event", "source_id": event_id})["found"])
        self.assertIn("projected_events", runner._dispatch("request_26", "inspect_forecast", {}))
        terms = runner._dispatch("request_26", "calculate_payment_options", {})
        self.assertIn("installments", terms)
        self.assertIn("amount_safe_to_pay", terms)
        self.assertTrue(runner._dispatch("request_26", "verify_candidate", {"candidate_id": "candidate_001"})["found"])


if __name__ == "__main__":
    unittest.main()
