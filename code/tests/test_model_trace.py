from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from agent.trace import ModelTurnRecorder  # noqa: E402
from agent.config import AgentConfig  # noqa: E402
from agent.openai_client import OpenAIResponsesClient  # noqa: E402


class ModelTraceTests(unittest.TestCase):
    def test_reasoning_effort_is_loaded_from_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_REASONING_EFFORT=HIGH\n", encoding="utf-8")
            config = AgentConfig.from_dotenv(path)
        self.assertEqual(config.reasoning_effort, "high")

    def test_invalid_reasoning_effort_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_REASONING_EFFORT=turbo\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                AgentConfig.from_dotenv(path)

    def test_client_passes_effort_to_both_api_shapes(self):
        class Completions:
            def __init__(self): self.kwargs = []
            def create(self, **kwargs):
                self.kwargs.append(kwargs)
                return {"ok": True}
        class Responses:
            def __init__(self): self.kwargs = []
            def create(self, **kwargs):
                self.kwargs.append(kwargs)
                return {"ok": True}
        class API:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": Completions()})()
                self.responses = Responses()
        client = OpenAIResponsesClient.__new__(OpenAIResponsesClient)
        client.model = "test"
        client.reasoning_effort = "low"
        client.client = API()
        client.usage_tracker = None
        client.trace_writer = None
        client._trace_context = {}
        client.create(model="test")
        client.chat_create(model="test", messages=[])
        self.assertEqual(client.client.responses.kwargs[0]["reasoning"], {"effort": "low"})
        self.assertEqual(client.client.chat.completions.kwargs[0]["reasoning_effort"], "low")

    def test_each_turn_is_flushed_and_sensitive_payloads_are_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "turns.jsonl"
            recorder = ModelTurnRecorder(path, run_id="run-1")
            recorder.record(kind="critic", model="test-model",
                            request={"api_key": "secret", "messages": [{"content": "inspect"},
                                      {"image_url": "data:image/png;base64,secret-image"}]},
                            response={"reasoning_content": "private reasoning", "output": "{}"},
                            context={"request_id": "request_01", "turn": 0})
            self.assertTrue(path.read_text(encoding="utf-8").strip())
            recorder.record(kind="evidence_extraction", model="test-model",
                            request={"evidence_id": "message_01"}, response={"facts": []},
                            context={"request_id": "request_02", "evidence_id": "message_01"})
            recorder.close()
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["sequence"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["context"]["request_id"], "request_01")
        self.assertEqual(rows[1]["context"]["request_id"], "request_02")
        self.assertEqual(rows[0]["request"]["api_key"], "[REDACTED]")
        self.assertEqual(rows[0]["request"]["messages"][1]["image_url"], "[IMAGE DATA URL REDACTED]")
        self.assertEqual(rows[0]["response"]["reasoning_content"], "private reasoning")


if __name__ == "__main__":
    unittest.main()
