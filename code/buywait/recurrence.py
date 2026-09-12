from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta

from .domain import EventStatus, FinancialEvent


@dataclass(frozen=True)
class RecurringStream:
    representative: FinancialEvent
    cadence_days: int

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
    terminal = {(event.category, event.direction) for event in events if event.status == EventStatus.CANCELLED or any(word in event.description.lower() for word in ("final", "last", "ended", "end of", "terminated", "terminal stream"))}
    exact: dict[tuple[str, str, str], list[FinancialEvent]] = {}
    by_category: dict[tuple[str, str], list[FinancialEvent]] = {}
    for event in eligible:
        exact.setdefault((event.category, event.description, event.direction), []).append(event)
        by_category.setdefault((event.category, event.direction), []).append(event)
    streams: list[RecurringStream] = []
    covered_categories: set[tuple[str, str]] = set()
    for key, values in exact.items():
        if (key[0], key[2]) in terminal:
            continue
        stream = _stable_stream(values, 3)
        if stream:
            streams.append(stream)
            covered_categories.add((key[0], key[2]))
    for key, values in by_category.items():
        if key in terminal or key in covered_categories or len({event.description for event in values}) < 2:
            continue
        stream = _stable_stream(values, 4)
        if stream:
            streams.append(stream)
    overrides = [event for event in events if event.event_id.startswith("evidence:") and event.recurrence_days != 0 and event.status in {EventStatus.SCHEDULED, EventStatus.SETTLED} and event.amount is not None]
    for override in overrides:
        for index, stream in enumerate(streams):
            representative = stream.representative
            if (representative.category, representative.direction) == (override.category, override.direction):
                streams[index] = RecurringStream(override, override.recurrence_days or stream.cadence_days)
                break
    return tuple(streams)


def recurring_event_ids(events) -> frozenset[str]:
    return frozenset(stream.representative.event_id for stream in detect_recurrences(events))
