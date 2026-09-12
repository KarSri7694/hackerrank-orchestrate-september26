from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

from buywait.domain import PaymentMethod

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

# The model critiques state assumptions only. It has no fields with which to
# select a payment method, schedule, or spending change.
CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "agree": {"type": "boolean"},
        "issue_type": {"type": ["string", "null"], "enum": [
            "missing_event", "cancelled_event", "amended_event", "incorrect_recurrence",
            "wrong_stream_termination", "internal_transfer", "duplicate_lifecycle",
            "incorrect_evidence_interpretation", "one_time_vs_recurring", None,
        ]},
        "supporting_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "correction": {"type": ["object", "null"], "properties": {
            "evidence_id": {"type": "string"}, "effect": {"type": "string", "enum": ["cancel", "amend", "amount_amendment", "date_amendment", "delay", "settle", "image_amount", "stream_update", "aggregate_stream_update", "terminate", "internal_transfer", "one_time"]},
            "related_event_id": {"type": ["string", "null"]}, "amount": {"type": ["string", "null"]},
            "currency": {"type": ["string", "null"]}, "effective_date": {"type": ["string", "null"]},
            "status": {"type": ["string", "null"]}, "category": {"type": ["string", "null"]},
            "direction": {"type": ["string", "null"]}, "description": {"type": ["string", "null"]},
            "recurring": {"type": ["boolean", "null"]}, "recurrence_days": {"type": ["integer", "null"]},
            "flexibility": {"type": ["string", "null"]}, "minimum_allowed_amount": {"type": ["string", "null"]},
            "stream_source": {"type": ["string", "null"]},
        }, "required": ["evidence_id", "effect", "related_event_id", "amount", "currency", "effective_date", "status", "category", "direction", "description", "recurring", "recurrence_days", "flexibility", "minimum_allowed_amount", "stream_source"], "additionalProperties": False},
        "summary": {"type": "string"},
    },
    "required": ["agree", "issue_type", "supporting_evidence_ids", "correction", "summary"],
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
