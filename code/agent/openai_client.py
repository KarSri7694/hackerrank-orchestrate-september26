from __future__ import annotations

from .config import AgentConfig


class OpenAIResponsesClient:
    def __init__(self, config: AgentConfig):
        from openai import OpenAI
        if not config.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        kwargs = {"api_key": config.api_key, "timeout": config.timeout, "base_url": config.base_url}
        self.model = config.model
        self.client = OpenAI(**kwargs)

    def create(self, **kwargs):
        return self.client.responses.create(**kwargs)
