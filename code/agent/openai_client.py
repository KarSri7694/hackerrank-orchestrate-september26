from __future__ import annotations

from .config import AgentConfig


class OpenAIResponsesClient:
    def __init__(self, config: AgentConfig, usage_tracker=None, trace_writer=None):
        from openai import OpenAI
        if not config.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        kwargs = {"api_key": config.api_key, "timeout": config.timeout, "base_url": config.base_url}
        self.model = config.model
        self.reasoning_effort = config.reasoning_effort
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
