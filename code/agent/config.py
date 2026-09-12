from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class AgentConfig:
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5"
    ai_mode: str = "auto"
    timeout: float = 60.0
    max_turns: int = 4
    max_tool_calls: int = 8
    max_evidence_calls: int = 4
    max_critique_rounds: int = 2
    image_detail: str = "high"

    @classmethod
    def from_dotenv(cls, path: str | Path) -> "AgentConfig":
        values = dotenv_values(Path(path))
        def value(name: str, default: str | None = None) -> str | None:
            raw = values.get(name, default)
            return str(raw).strip() if raw is not None and str(raw).strip() else default
        return cls(
            api_key=value("OPENAI_API_KEY"),
            base_url=value("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=value("OPENAI_MODEL", "gpt-5"),
            ai_mode=(value("AI_MODE", "auto") or "auto").lower(),
            timeout=float(value("OPENAI_TIMEOUT", "60")),
            max_turns=min(int(value("AGENT_MAX_TURNS", "4")), 6),
            max_tool_calls=min(int(value("AGENT_MAX_TOOL_CALLS", "8")), 12),
            max_evidence_calls=min(int(value("AGENT_MAX_EVIDENCE_CALLS", "4")), 8),
            max_critique_rounds=min(max(1, int(value("AGENT_MAX_CRITIQUE_ROUNDS", "2"))), 2),
            image_detail=value("OPENAI_IMAGE_DETAIL", "high") or "high",
        )
