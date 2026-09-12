from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

from .domain import EventStatus, FinancialEvent


def stream_key(category: str, direction: str, description: str | None) -> str:
    """Stable, deterministic stream identity; category alone is insufficient."""
    source = re.sub(r"\b(final|last|ended|terminated)\b", "", (description or "").lower())
    source = re.sub(r"[^a-z0-9]+", " ", source).strip()
    return "|".join((category.strip().lower(), direction.strip().lower(), source))


def event_stream_key(event: FinancialEvent) -> str:
    return stream_key(event.category, event.direction, event.description)


@dataclass(frozen=True)
class RecurringStream:
    representative: FinancialEvent
    cadence_days: int
    starts_on: date | None = None
    ends_before: date | None = None

    def next_date(self, value: date) -> date:
        if 25 <= self.cadence_days <= 35:
            month_index = value.year * 12 + value.month
            year, zero_month = divmod(month_index, 12)
            month = zero_month + 1
            return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))
        return value + timedelta(days=self.cadence_days)


def _stable_stream(values: list[FinancialEvent], minimum: int) -> RecurringStream | None:
    ordered = sorted(values, key=lambda event: event.event_date)
    if len(ordered) < minimum:
        return None
    gaps = [(right.event_date - left.event_date).days for left, right in zip(ordered, ordered[1:])]
    median = sorted(gaps)[len(gaps) // 2]
    if not 5 <= median <= 35 or any(abs(gap - median) > 3 for gap in gaps):
        return None
    return RecurringStream(ordered[-1], median)


def detect_recurrences(events) -> tuple[RecurringStream, ...]:
    eligible = [event for event in events if event.status == EventStatus.SETTLED and event.amount is not None]
    terminal = {event_stream_key(event) for event in events if event.status == EventStatus.CANCELLED or any(word in event.description.lower() for word in ("final", "last", "ended", "end of", "terminated", "terminal stream"))}
    exact: dict[tuple[str, str, str], list[FinancialEvent]] = {}
    by_category: dict[tuple[str, str], list[FinancialEvent]] = {}
    for event in eligible:
        exact.setdefault((event.category, event.description, event.direction), []).append(event)
        by_category.setdefault((event.category, event.direction), []).append(event)
    streams: list[RecurringStream] = []
    covered_categories: set[tuple[str, str]] = set()
    for key, values in exact.items():
        if stream_key(key[0], key[2], key[1]) in terminal:
            continue
        stream = _stable_stream(values, 3)
        if stream:
            streams.append(stream)
            covered_categories.add((key[0], key[2]))
    for key, values in by_category.items():
        values = [event for event in values if event_stream_key(event) not in terminal]
        if key in covered_categories or len({event.description for event in values}) < 2:
            continue
        stream = _stable_stream(values, 4)
        if stream:
            streams.append(stream)
    aggregates = [event for event in events if event.event_type == "aggregate_stream_update" and event.recurrence_days != 0 and event.status in {EventStatus.SCHEDULED, EventStatus.SETTLED} and event.amount is not None]
    for aggregate in aggregates:
        # Historical streams keep their history but stop forecasting once the
        # aggregate total becomes authoritative for that category/direction.
        streams = [RecurringStream(stream.representative, stream.cadence_days, stream.starts_on, aggregate.event_date)
                   if (stream.representative.category, stream.representative.direction) == (aggregate.category, aggregate.direction)
                   else stream for stream in streams]
        streams.append(RecurringStream(aggregate, aggregate.recurrence_days or 30, aggregate.event_date))
    overrides = [event for event in events if event.event_id.startswith("evidence:") and event.event_type != "aggregate_stream_update" and event.recurrence_days != 0 and event.status in {EventStatus.SCHEDULED, EventStatus.SETTLED} and event.amount is not None]
    for override in overrides:
        for index, stream in enumerate(streams):
            representative = stream.representative
            if event_stream_key(representative) == event_stream_key(override):
                streams[index] = RecurringStream(override, override.recurrence_days or stream.cadence_days)
                break
        else:
            # A stream explicitly marked recurring by evidence needs no
            # historical rows. One-time evidence uses recurrence_days=0 and
            # is deliberately excluded above.
            streams.append(RecurringStream(override, override.recurrence_days or 30))
    return tuple(streams)


def recurring_event_ids(events) -> frozenset[str]:
    return frozenset(stream.representative.event_id for stream in detect_recurrences(events))
