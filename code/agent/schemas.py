from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

from buywait.domain import DecisionCore, Payment, PaymentMethod, SpendingChange

FINAL_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "request_id": {"type": "string"},
        "selected_method": {"type": "string", "enum": [x.value for x in PaymentMethod]},
        "selected_payment_option_id": {"type": ["string", "null"]},
        "payments": {"type": "array", "items": {"type": "object", "properties": {"date": {"type": "string"}, "amount": {"type": "string"}}, "required": ["date", "amount"], "additionalProperties": False}},
        "spending_changes": {"type": "array", "items": {"type": "object", "properties": {"event_id": {"type": "string"}, "action": {"type": "string", "enum": ["stop", "reduce_to"]}, "new_amount": {"type": ["string", "null"]}}, "required": ["event_id", "action", "new_amount"], "additionalProperties": False}},
        "reasoning_summary": {"type": "string"},
        "evidence_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["request_id", "selected_method", "selected_payment_option_id", "payments", "spending_changes", "reasoning_summary", "evidence_used"],
    "additionalProperties": False,
}


def jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    return value


def response_output_items(response):
    return [jsonable(item) for item in getattr(response, "output", [])]


def response_function_calls(response):
    calls = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "function_call":
            calls.append({"name": item.name, "call_id": item.call_id, "arguments": item.arguments})
    return calls

