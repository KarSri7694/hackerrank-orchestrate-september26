from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import AgentConfig
from .image_inputs import image_data_url
from .openai_client import OpenAIResponsesClient
from .prompts import CRITIC_PROMPT, DECISION_PROMPT
from .schemas import (CRITIC_REVIEW_SCHEMA, DECISION_SELECTION_SCHEMA,
                      parse_json_object, response_function_calls,
                      response_output_items, response_text)


@dataclass
class AgentResult:
    decision: object
    used_ai: bool
    turns: int
    tool_calls: int
    fallback_reason: str | None = None


class AgentRunner:
    """A deliberately small two-call controller around deterministic choices."""

    def __init__(self, application, dataset_dir: str | Path, *, model_client=None,
                 config: AgentConfig | None = None, usage_tracker=None, trace_writer=None):
        self.application = application
        self.dataset_dir = Path(dataset_dir)
        self.model_client = model_client
        self.config = config or AgentConfig()
        self.usage_tracker = usage_tracker
        self.trace_writer = trace_writer

    def run(self, request_id: str) -> AgentResult:
        fallback = self.application.decision(request_id)
        if self.config.ai_mode == "disabled" or (self.config.ai_mode == "auto" and not self.config.api_key):
            return AgentResult(fallback, False, 0, 0, "AI disabled or OPENAI_API_KEY missing")
        try:
            client = self.model_client or OpenAIResponsesClient(
                self.config, usage_tracker=self.usage_tracker, trace_writer=self.trace_writer)
            case = self.application.agent_case(request_id)
            total_turns = total_calls = 0
            for _attempt in range(self.config.max_decision_retries + 1):
                selection, turns, calls = self._ask(
                    client, request_id, case, DECISION_PROMPT, DECISION_SELECTION_SCHEMA, "decision")
                total_turns += turns
                total_calls += calls
                if not self._valid_selection(selection, request_id):
                    continue
                chosen = self._selected(request_id, selection) or fallback
                review_case = {"case": case, "decision_model_output": selection,
                               "default_candidate_id": self._candidate_id(case, chosen)}
                review, review_turns, review_calls = self._ask(
                    client, request_id, review_case, CRITIC_PROMPT, CRITIC_REVIEW_SCHEMA, "critic")
                total_turns += review_turns
                total_calls += review_calls
                if not self._valid_review(review):
                    continue
                if review["approved"] is False:
                    chosen = self._selected(request_id, review) or chosen
                return AgentResult(chosen, True, total_turns, total_calls)
            return AgentResult(
                fallback, False, total_turns, total_calls,
                f"malformed AI response after {self.config.max_decision_retries + 1} attempt(s)",
            )
        except Exception as exc:
            if self.config.ai_mode == "enabled":
                raise
            return AgentResult(fallback, False, 0, 0, f"agent failure: {type(exc).__name__}")

    def _ask(self, client, request_id, payload, instructions, schema, phase):
        history = [{"role": "user", "content": self._content(payload, request_id)}]
        calls = 0
        for turn in range(self.config.max_turns):
            setter = getattr(client, "set_trace_context", None)
            if callable(setter):
                setter(request_id=request_id, phase=phase, turn=turn)
            response = client.create(
                model=client.model, instructions=instructions, input=history,
                tools=self._tools(), tool_choice="auto", store=False,
                max_output_tokens=self.config.max_output_tokens,
                text={"format": {"type": "json_schema", "name": f"{phase}_selection",
                                  "strict": True, "schema": schema}}, usage_kind=phase,
            )
            function_calls = response_function_calls(response)
            if not function_calls:
                parsed = parse_json_object(response_text(response))
                return (parsed if isinstance(parsed, dict) else None), turn + 1, calls
            history.extend(response_output_items(response))
            for call in function_calls:
                if calls >= self.config.max_tool_calls:
                    raise RuntimeError("maximum tool calls exceeded")
                try:
                    result = self._dispatch(request_id, call["name"], json.loads(call["arguments"]))
                except Exception as exc:
                    result = {"ok": False, "error": type(exc).__name__}
                history.append({"type": "function_call_output", "call_id": call["call_id"],
                                "output": json.dumps(result, default=str)})
                calls += 1
        return None, self.config.max_turns, calls

    def _content(self, payload, request_id):
        content = [{"type": "input_text", "text": json.dumps(payload, default=str)}]
        for ref in self.application.repository.context(request_id).evidence_refs:
            if ref.image_path:
                content.append({"type": "input_image", "image_url": image_data_url(
                    ref.image_path, self.dataset_dir / "media" / "images"),
                    "detail": self.config.image_detail})
        return content

    def _dispatch(self, request_id, name, arguments):
        if name == "inspect_source":
            return self.application.inspect_source(request_id, arguments["source_type"], arguments["source_id"])
        if name == "inspect_forecast":
            return self.application.explain_timeline(request_id)
        if name == "calculate_payment_options":
            return self.application.payment_options_summary(request_id)
        if name == "verify_candidate":
            decision = self.application.decision_for_candidate(request_id, arguments["candidate_id"])
            if decision is None:
                return {"found": False}
            result = self.application.simulate_plan(request_id, decision.payment_plan, decision.spending_changes)
            return {"found": True, "safe": result.safe,
                    "minimum_projected_balance": str(result.minimum_projected_balance),
                    "first_violation_date": result.first_violation_date.isoformat() if result.first_violation_date else None}
        return {"ok": False, "error": "unknown_tool"}

    def _selected(self, request_id, response):
        return self.application.decision_for_candidate(
            request_id, response.get("candidate_id")) if isinstance(response, dict) else None

    @staticmethod
    def _valid_selection(response, request_id: str) -> bool:
        """Only retry a structurally malformed decision response.

        A well-formed but unknown candidate remains harmless: the deterministic
        fallback candidate is used, just as before. This prevents a model from
        turning a bad candidate ID into an unbounded retry loop.
        """
        return (isinstance(response, dict)
                and response.get("request_id") == request_id
                and isinstance(response.get("candidate_id"), str)
                and isinstance(response.get("explanation"), str)
                and isinstance(response.get("evidence_used"), list))

    @staticmethod
    def _valid_review(response) -> bool:
        return (isinstance(response, dict)
                and isinstance(response.get("approved"), bool)
                and isinstance(response.get("candidate_id"), str)
                and isinstance(response.get("explanation"), str))

    @staticmethod
    def _candidate_id(case, decision):
        plan = [{"date": item.date.isoformat(), "amount": str(item.amount)} for item in decision.payment_plan]
        for candidate in case["candidates"]:
            if candidate["recommended_payment_method"] == decision.recommended_payment_method.value and candidate["payment_plan"] == plan:
                return candidate["candidate_id"]
        return "candidate_001"

    @staticmethod
    def _tools():
        return [
            {"type": "function", "name": "inspect_source", "description": "Inspect one supplied event or evidence record.", "strict": True, "parameters": {"type": "object", "additionalProperties": False, "properties": {"source_type": {"type": "string", "enum": ["event", "evidence"]}, "source_id": {"type": "string"}}, "required": ["source_type", "source_id"]}},
            {"type": "function", "name": "inspect_forecast", "description": "Return the deterministic 90-day forecast.", "strict": True, "parameters": {"type": "object", "additionalProperties": False, "properties": {}, "required": []}},
            {"type": "function", "name": "calculate_payment_options", "description": "Calculate legal full, partial, wait, and installment payment terms from supplied data.", "strict": True, "parameters": {"type": "object", "additionalProperties": False, "properties": {}, "required": []}},
            {"type": "function", "name": "verify_candidate", "description": "Verify a supplied candidate is safe.", "strict": True, "parameters": {"type": "object", "additionalProperties": False, "properties": {"candidate_id": {"type": "string"}}, "required": ["candidate_id"]}},
        ]
