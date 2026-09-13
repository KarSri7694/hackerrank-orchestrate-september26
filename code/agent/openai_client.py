from __future__ import annotations

from .config import AgentConfig


def _chat_tools(tools):
    converted = []
    for tool in tools or ():
        if tool.get("type") != "function":
            converted.append(tool)
            continue
        function = {key: tool[key] for key in ("name", "description", "parameters", "strict") if key in tool}
        converted.append({"type": "function", "function": function})
    return converted


def _chat_content(content):
    if isinstance(content, str) or content is None:
        return content
    result = []
    for item in content:
        if not isinstance(item, dict):
            result.append(item)
        elif item.get("type") in {"input_text", "text"}:
            result.append({"type": "text", "text": item.get("text", "")})
        elif item.get("type") == "input_image":
            result.append({"type": "image_url", "image_url": {"url": item.get("image_url"), "detail": item.get("detail", "auto")}})
        else:
            result.append(item)
    return result


def _chat_messages(history, instructions=None):
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    for item in history or ():
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": item.get("output", "")})
            continue
        role = item.get("role")
        if role in {"user", "system", "developer"}:
            messages.append({"role": role, "content": _chat_content(item.get("content"))})
            continue
        # Responses output items are converted to an assistant message so a
        # subsequent Chat Completions turn receives the complete tool history.
        if item.get("type") == "function_call":
            messages.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": item.get("call_id"), "type": "function",
                "function": {"name": item.get("name"), "arguments": item.get("arguments", "{}")},
            }]})
        elif role == "assistant":
            messages.append(item)
    return messages


class OpenAIResponsesClient:
    def __init__(self, config: AgentConfig, usage_tracker=None, trace_writer=None):
        from openai import OpenAI
        if not config.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        kwargs = {"api_key": config.api_key, "timeout": config.timeout, "base_url": config.base_url}
        self.model = config.model
        self.reasoning_effort = config.reasoning_effort
        self.agent_api = config.agent_api
        self.use_chat_for_agent = config.agent_api == "chat" or (
            config.agent_api == "auto" and "cerebras.ai" in config.base_url.casefold())
        self.client = OpenAI(**kwargs)
        self.usage_tracker = usage_tracker
        self.trace_writer = trace_writer
        self._trace_context = {}

    def set_trace_context(self, **context):
        self._trace_context = {key: value for key, value in context.items() if value is not None}

    def _record(self, kind, kwargs, response):
        if self.trace_writer is not None:
            self.trace_writer.record(kind=kind, model=self.model, request=kwargs,
                                     response=response, context=self._trace_context)

    def create(self, **kwargs):
        usage_kind = kwargs.pop("usage_kind", "unknown")
        if getattr(self, "use_chat_for_agent", False):
            return self._create_chat_agent(usage_kind, kwargs)
        if self.reasoning_effort and "reasoning" not in kwargs:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        try:
            response = self.client.responses.create(**kwargs)
        except Exception as exc:
            self._record(usage_kind, kwargs, {"error": f"{type(exc).__name__}: provider call failed"})
            raise
        self._record(usage_kind, kwargs, response)
        if self.usage_tracker is not None:
            self.usage_tracker.record(usage_kind, self.model, response)
        return response

    def _create_chat_agent(self, usage_kind, kwargs):
        request = dict(kwargs)
        instructions = request.pop("instructions", None)
        history = request.pop("input", ())
        request.pop("store", None)
        request.pop("text", None)
        if "max_output_tokens" in request:
            request["max_tokens"] = request.pop("max_output_tokens")
        if "reasoning" in request:
            request["reasoning_effort"] = request.pop("reasoning").get("effort")
        elif self.reasoning_effort and "reasoning_effort" not in request:
            request["reasoning_effort"] = self.reasoning_effort
        request["messages"] = _chat_messages(history, instructions)
        if "tools" in request:
            request["tools"] = _chat_tools(request["tools"])
        # The Responses JSON-schema format maps to Chat Completions' nested
        # json_schema response_format.
        response_format = kwargs.get("text", {}).get("format") if isinstance(kwargs.get("text"), dict) else None
        if response_format:
            request["response_format"] = {"type": "json_schema", "json_schema": {
                "name": response_format.get("name", "structured_output"),
                "strict": response_format.get("strict", True),
                "schema": response_format.get("schema", {}),
            }}
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            self._record(usage_kind, request, {"error": f"{type(exc).__name__}: provider call failed"})
            raise
        self._record(usage_kind, request, response)
        if self.usage_tracker is not None:
            self.usage_tracker.record(usage_kind, self.model, response)
        return response

    def chat_create(self, **kwargs):
        """Use the OpenAI-compatible chat endpoint for structured extraction."""
        usage_kind = kwargs.pop("usage_kind", "unknown")
        if self.reasoning_effort and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            self._record(usage_kind, kwargs, {"error": f"{type(exc).__name__}: provider call failed"})
            raise
        self._record(usage_kind, kwargs, response)
        if self.usage_tracker is not None:
            self.usage_tracker.record(usage_kind, self.model, response)
        return response
