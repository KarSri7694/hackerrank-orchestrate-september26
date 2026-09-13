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
    output = getattr(response, "output", None)
    if output is not None:
        return [jsonable(item) for item in output]
    choices = getattr(response, "choices", None) or []
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    if message is None:
        return []
    # Store a Chat Completions assistant message in the same history shape
    # accepted by _chat_messages on the next agent turn.
    return [{
        "role": "assistant",
        "content": getattr(message, "content", None),
        "tool_calls": [jsonable(call) for call in (getattr(message, "tool_calls", None) or [])],
    }]


def response_function_calls(response):
    calls = []
    output = getattr(response, "output", None)
    if output is None:
        choices = getattr(response, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        for tool_call in (getattr(message, "tool_calls", None) or []):
            function = getattr(tool_call, "function", None)
            if function is not None:
                calls.append({
                    "name": getattr(function, "name", None),
                    "call_id": getattr(tool_call, "id", None),
                    "arguments": getattr(function, "arguments", "{}"),
                })
        return calls
    for item in output:
        if getattr(item, "type", None) == "function_call":
            calls.append({"name": item.name, "call_id": item.call_id, "arguments": item.arguments})
    return calls


def response_text(response) -> str:
    """Read final text from Responses SDK and OpenAI-compatible response shapes."""
    direct = getattr(response, "output_text", None)
    if direct:
        return str(direct)
    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content:
            return str(content)
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            return str(reasoning)
    pieces = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) in {"output_text", "text"} and getattr(content, "text", None):
                pieces.append(str(content.text))
        if getattr(item, "type", None) in {"output_text", "text"} and getattr(item, "text", None):
            pieces.append(str(item.text))
    return "\n".join(pieces)


def parse_json_object(raw: str) -> dict | None:
    """Extract a JSON object after optional thinking/fence text."""
    decoder = json.JSONDecoder()
    found = None
    found_rank = (-1, -1)
    for index, character in enumerate(raw or ""):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            signature = 1 if {"agree", "correction", "facts"} & candidate.keys() else 0
            rank = (signature, len(raw[index:]))
            if rank > found_rank:
                found, found_rank = candidate, rank
    return found
