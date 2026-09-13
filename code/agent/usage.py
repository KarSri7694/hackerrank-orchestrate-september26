from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class UsageRecord:
    kind: str
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


class UsageTracker:
    """Records actual provider calls only; cache hits are tracked separately."""
    def __init__(self, provider: str, input_cost_per_million: Decimal | None = None,
                 output_cost_per_million: Decimal | None = None):
        self.provider = provider
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.records: list[UsageRecord] = []
        self.cache_hits = 0

    def record(self, kind: str, model: str, response) -> None:
        usage = getattr(response, "usage", None)
        def field(name):
            if usage is None:
                return None
            aliases = {
                "input_tokens": ("input_tokens", "prompt_tokens"),
                "output_tokens": ("output_tokens", "completion_tokens"),
                "total_tokens": ("total_tokens",),
            }
            return next((getattr(usage, alias, None) for alias in aliases[name] if getattr(usage, alias, None) is not None), None)
        self.records.append(UsageRecord(kind, self.provider, model, field("input_tokens"),
                                        field("output_tokens"), field("total_tokens")))

    def report(self, path: str | Path, request_count: int, cache_hits: int = 0) -> None:
        self.cache_hits = cache_hits
        grouped: dict[tuple[str, str, str], list[UsageRecord]] = defaultdict(list)
        for record in self.records:
            grouped[(record.kind, record.provider, record.model)].append(record)
        input_total = sum(record.input_tokens or 0 for record in self.records)
        output_total = sum(record.output_tokens or 0 for record in self.records)
        token_total = sum(record.total_tokens or ((record.input_tokens or 0) + (record.output_tokens or 0)) for record in self.records)
        cost = None
        if self.input_cost_per_million is not None and self.output_cost_per_million is not None:
            cost = (Decimal(input_total) * self.input_cost_per_million + Decimal(output_total) * self.output_cost_per_million) / Decimal("1000000")
        lines = ["# Model usage report", "", "This report describes actual model calls made by the most recent successful production run.", "", "## Calls", "", "| Kind | Provider | Model | Calls | Input tokens | Output tokens | Total tokens |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
        for (kind, provider, model), records in sorted(grouped.items()):
            inputs = sum(record.input_tokens or 0 for record in records)
            outputs = sum(record.output_tokens or 0 for record in records)
            totals = sum(record.total_tokens or ((record.input_tokens or 0) + (record.output_tokens or 0)) for record in records)
            lines.append(f"| {kind} | {provider} | {model} | {len(records)} | {inputs} | {outputs} | {totals} |")
        lines += ["", "## Totals", "", f"- Requests processed: {request_count}", f"- Actual model calls: {len(self.records)}", f"- Evidence cache hits: {cache_hits}", f"- Input tokens: {input_total}", f"- Output tokens: {output_total}", f"- Total tokens: {token_total}", f"- Average tokens per request: {Decimal(token_total) / Decimal(request_count) if request_count else Decimal('0')}"]
        if cost is None:
            lines.append("- Estimated cost: unavailable (configure input/output cost per million tokens to calculate it)")
        else:
            lines += [f"- Estimated total cost: {format(cost, 'f')}", f"- Estimated cost per request: {format(cost / Decimal(request_count), 'f') if request_count else '0'}"]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
