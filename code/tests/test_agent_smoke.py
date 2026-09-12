from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from agent.runner import AgentRunner  # noqa: E402
from agent.config import AgentConfig  # noqa: E402
from tools.server import mcp  # noqa: E402
from buywait.application import Application  # noqa: E402


class Item:
    def __init__(self, kind, **values):
        self.type = kind
        self.__dict__.update(values)

    def model_dump(self):
        return dict(self.__dict__)


class Response:
    def __init__(self, output, output_text=""):
        self.output = output
        self.output_text = output_text


class FakeOpenAI:
    model = "fake-multimodal-model"

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return Response([Item("function_call", name="get_case", call_id="call_1", arguments=json.dumps({"request_id": "request_26"}))])
        payload = {"request_id": "request_26", "selected_method": "full_payment", "selected_payment_option_id": None, "payments": [{"date": "2025-08-03", "amount": "15656000"}], "spending_changes": [], "reasoning_summary": "Verified with the case and simulator.", "evidence_used": []}
        return Response([], json.dumps(payload))


class AgentSmokeTest(unittest.TestCase):
    def test_bounded_loop_dispatches_mcp_tool_and_validates_plan(self):
        app = Application(ROOT / "dataset")
        fake = FakeOpenAI()
        config = AgentConfig(api_key="test-key", ai_mode="enabled", base_url="https://example.invalid/v1")
        result = AgentRunner(app, mcp, ROOT / "dataset", model_client=fake, config=config).run("request_26")
        self.assertTrue(result.used_ai)
        self.assertEqual(result.decision.requested_method if hasattr(result.decision, "requested_method") else result.decision.recommended_payment_method.value, "full_payment")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["tools"][0]["type"], "function")


if __name__ == "__main__":
    unittest.main()
