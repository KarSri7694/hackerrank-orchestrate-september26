from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

from buywait.domain import PaymentMethod

DECISION_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "request_id": {"type": "string"},
        "candidate_id": {"type": "string"},
        "explanation": {"type": "string"},
        "evidence_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["request_id", "candidate_id", "explanation", "evidence_used"],
    "additionalProperties": False,
}

CRITIC_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "candidate_id": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["approved", "candidate_id", "explanation"],
    "additionalProperties": False,
}

# Backward-compatible import names for downstream integrations.  New code
# must use the two explicit selection schemas above.
FINAL_DECISION_SCHEMA = DECISION_SELECTION_SCHEMA
CRITIQUE_SCHEMA = CRITIC_REVIEW_SCHEMA


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
