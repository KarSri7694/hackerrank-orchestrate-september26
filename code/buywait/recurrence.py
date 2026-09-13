from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, replace
from datetime import date, timedelta

from .domain import EventStatus, FinancialEvent


def stream_key(category: str, direction: str, description: str | None) -> str:
    """Stable, deterministic stream identity; category alone is insufficient."""
    source = re.sub(r"[^a-z0-9]+", " ", (description or "").lower())
    # These words describe the lifecycle or transaction representation, not
    # the employer/merchant/service that owns the stream. Removing them lets a
    # final payroll row close the preceding payroll stream while retaining
    # distinguishing names such as "Acme" vs "Beta".
    boilerplate = {
        "final", "last", "ended", "terminated", "terminal", "current",
        "next", "confirmed", "scheduled", "upcoming", "regular", "monthly",
        "weekly", "daily", "employer", "payroll", "salary", "income",
        "credit", "deposit", "payment", "transfer", "debit", "expense",
        "purchase", "order", "charge", "service", "subscription", "plan",
    }
    source = " ".join(word for word in source.split() if word not in boilerplate)
    return "|".join((category.strip().lower(), direction.strip().lower(), source))


def event_stream_key(event: FinancialEvent) -> str:
    return stream_key(event.category, event.direction, event.description)


def cash_date(event: FinancialEvent) -> date:
    """Date on which an event affects cash, falling back to event date."""
    return event.settlement_date or event.event_date


def _confirmed_salary_label(event: FinancialEvent) -> bool:
    """Recognize payroll-like historical credits conservatively."""
    if event.direction != "credit" or event.category != "salary":
        return False
    text = event.description.casefold()
    return any(token in text for token in ("salary", "payroll", "wage", "employer", "monthly pay"))


@dataclass(frozen=True)
class RecurringStream:
    representative: FinancialEvent
    cadence_days: int
    starts_on: date | None = None
    ends_before: date | None = None
    anchor_day: int | None = None

    def next_date(self, value: date) -> date:
        if 25 <= self.cadence_days <= 35:
            month_index = value.year * 12 + value.month
            year, zero_month = divmod(month_index, 12)
            month = zero_month + 1
            anchor = self.anchor_day or value.day
            return date(year, month, min(anchor, calendar.monthrange(year, month)[1]))
        return value + timedelta(days=self.cadence_days)


def _stable_stream(values: list[FinancialEvent], minimum: int) -> RecurringStream | None:
    ordered = sorted(values, key=cash_date)
    if len(ordered) < minimum:
        return None
    gaps = [(cash_date(right) - cash_date(left)).days for left, right in zip(ordered, ordered[1:])]
    median = sorted(gaps)[len(gaps) // 2]
    if not 5 <= median <= 35 or any(abs(gap - median) > 3 for gap in gaps):
        return None
    representative = ordered[-1]
    amounts = [event.amount for event in ordered if event.amount is not None]
    if amounts:
        # Forecast conservatively when a supported stream varies: the largest
        # recurring debit or smallest recurring credit is the safer bound.
        forecast_amount = max(amounts) if representative.direction == "debit" else min(amounts)
        representative = replace(representative, amount=forecast_amount)
    return RecurringStream(representative, median, anchor_day=cash_date(ordered[-1]).day)


def detect_recurrences(events) -> tuple[RecurringStream, ...]:
    eligible = [event for event in events if event.status == EventStatus.SETTLED and event.amount is not None
                and event.event_type != "internal_transfer"
                and (event.direction != "credit" or event.category != "salary"
                     or _confirmed_salary_label(event))]
    # A cancelled transaction occurrence is not, by itself, proof that the
    # underlying recurring stream ended. Reconciliation tags an explicit
    # stream termination as ``stream_status``; free-form terminal wording is
    # also accepted for legacy/imported rows.
    terminal = {event_stream_key(event) for event in events
                if ((event.status == EventStatus.CANCELLED and event.event_type == "stream_status")
                    or any(word in event.description.lower() for word in
                           ("final", "last", "ended", "end of", "terminated", "terminal stream")))}
    pending_markers = [event for event in events if event.event_type == "stream_status"
                       and event.status == EventStatus.PENDING]
    pending_keys = {
        event_stream_key(marker) for marker in pending_markers
        if not any(event_stream_key(settled) == event_stream_key(marker)
                   and cash_date(settled) >= cash_date(marker)
                   for settled in eligible)
    }
    # Currency is part of the cash-flow stream for forecasting. Two rows can
    # legitimately share a merchant/category label while representing
    # separate currency accounts; combining them would select an amount from
    # one currency and apply it to the other.
    exact: dict[tuple[str, str, str, str], list[FinancialEvent]] = {}
    for event in eligible:
        exact.setdefault((event.category, event.description, event.direction, event.currency), []).append(event)
    streams: list[RecurringStream] = []
    covered_event_ids: set[str] = set()
    for key, values in exact.items():
        if (stream_key(key[0], key[2], key[1]) in terminal
                or event_stream_key(values[0]) in pending_keys):
            continue
        stream = _stable_stream(values, 3)
        if stream:
            streams.append(stream)
            covered_event_ids.update(event.event_id for event in values)
    # Do not pool different descriptions by category. Category-only pooling
    # collapses unrelated merchants/employers and forecasts the last observed
    # amount as though it belonged to one stable stream. Variable spending is
    # forecast only when the same stream has a supported recurrence; an
    # explicit aggregate_stream_update remains the opt-in category-level path.
    aggregates = [event for event in events if event.event_type == "aggregate_stream_update" and event.recurrence_days and event.recurrence_days > 0 and event.status in {EventStatus.SCHEDULED, EventStatus.SETTLED} and event.amount is not None]
    for aggregate in aggregates:
        # Historical streams keep their history but stop forecasting once the
        # aggregate total becomes authoritative for that category/direction.
        streams = [RecurringStream(stream.representative, stream.cadence_days, stream.starts_on, aggregate.event_date, stream.anchor_day)
                   if (stream.representative.category, stream.representative.direction) == (aggregate.category, aggregate.direction)
                   else stream for stream in streams]
        streams.append(RecurringStream(aggregate, aggregate.recurrence_days or 30, aggregate.event_date,
                                       anchor_day=aggregate.event_date.day))
    overrides = [event for event in events if event.event_id.startswith("evidence:") and event.event_type != "aggregate_stream_update" and event.recurrence_days and event.recurrence_days > 0 and event.status in {EventStatus.SCHEDULED, EventStatus.SETTLED} and event.amount is not None]
    for override in overrides:
        for index, stream in enumerate(streams):
            representative = stream.representative
            if event_stream_key(representative) == event_stream_key(override):
                streams[index] = RecurringStream(override, override.recurrence_days or stream.cadence_days,
                                                 anchor_day=override.event_date.day)
                break
        else:
            # A stream explicitly marked recurring by evidence needs no
            # historical rows. One-time evidence uses recurrence_days=0 and
            # is deliberately excluded above.
            streams.append(RecurringStream(override, override.recurrence_days or 30,
                                           anchor_day=override.event_date.day))
    return tuple(streams)


def recurring_event_ids(events) -> frozenset[str]:
    return frozenset(stream.representative.event_id for stream in detect_recurrences(events))
