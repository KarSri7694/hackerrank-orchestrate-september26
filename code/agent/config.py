from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from decimal import Decimal

from dotenv import dotenv_values


REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
AGENT_APIS = frozenset({"auto", "responses", "chat"})


@dataclass(frozen=True)
class AgentConfig:
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5"
    reasoning_effort: str | None = "medium"
    agent_api: str = "auto"
    ai_mode: str = "auto"
    timeout: float = 60.0
    evidence_timeout: float = 180.0
    max_turns: int = 2
    max_tool_calls: int = 8
    max_decision_retries: int = 2
    request_poll_seconds: float = 5.0
    max_evidence_calls: int = 4
    max_critique_rounds: int = 2
    max_output_tokens: int = 8192
    image_detail: str = "high"
    vision_enabled: bool = True
    input_cost_per_million: Decimal | None = None
    output_cost_per_million: Decimal | None = None

    @classmethod
    def from_dotenv(cls, path: str | Path) -> "AgentConfig":
        values = dotenv_values(Path(path))
        def value(name: str, default: str | None = None) -> str | None:
            raw = values.get(name, default)
            return str(raw).strip() if raw is not None and str(raw).strip() else default
        reasoning_effort = value("OPENAI_REASONING_EFFORT", "medium")
        if reasoning_effort is not None:
            reasoning_effort = reasoning_effort.lower()
            if reasoning_effort not in REASONING_EFFORTS:
                raise ValueError(
                    f"OPENAI_REASONING_EFFORT must be one of {sorted(REASONING_EFFORTS)}"
                )
        agent_api = (value("OPENAI_AGENT_API", "auto") or "auto").lower()
        if agent_api not in AGENT_APIS:
            raise ValueError(f"OPENAI_AGENT_API must be one of {sorted(AGENT_APIS)}")
        return cls(
            api_key=value("OPENAI_API_KEY"),
            base_url=value("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=value("OPENAI_MODEL", "gpt-5"),
            reasoning_effort=reasoning_effort,
            agent_api=agent_api,
            ai_mode=(value("AI_MODE", "auto") or "auto").lower(),
            timeout=float(value("OPENAI_TIMEOUT", "60")),
            evidence_timeout=float(value("OPENAI_EVIDENCE_TIMEOUT", "180")),
            max_turns=min(int(value("AGENT_MAX_TURNS", "2")), 6),
            max_tool_calls=min(int(value("AGENT_MAX_TOOL_CALLS", "8")), 12),
            max_decision_retries=min(max(0, int(value("AGENT_MAX_DECISION_RETRIES", "2"))), 5),
            request_poll_seconds=min(max(0.1, float(value("REQUESTS_POLL_SECONDS", "5"))), 300.0),
            max_evidence_calls=min(int(value("AGENT_MAX_EVIDENCE_CALLS", "4")), 8),
            max_critique_rounds=min(max(1, int(value("AGENT_MAX_CRITIQUE_ROUNDS", "2"))), 2),
            max_output_tokens=min(max(1024, int(value("AGENT_MAX_OUTPUT_TOKENS", "8192"))), 32768),
            image_detail=value("OPENAI_IMAGE_DETAIL", "high") or "high",
            vision_enabled=(value("OPENAI_VISION_ENABLED", "true") or "true").lower() in {"true", "1", "yes", "y"},
            input_cost_per_million=Decimal(value("OPENAI_INPUT_COST_PER_MILLION")) if value("OPENAI_INPUT_COST_PER_MILLION") else None,
            output_cost_per_million=Decimal(value("OPENAI_OUTPUT_COST_PER_MILLION")) if value("OPENAI_OUTPUT_COST_PER_MILLION") else None,
        )
