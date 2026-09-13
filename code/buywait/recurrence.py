from __future__ import annotations

import calendar
import re
from collections import defaultdict
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
        "next", "confirmed", "scheduled", "upcoming", "regular", "recurring", "monthly",
        "weekly", "daily", "employer", "payroll", "salary", "income",
        "credit", "deposit", "payment", "transfer", "debit", "expense",
        "purchase", "order", "charge", "service", "subscription", "plan",
        "from", "for", "by", "via", "received", "receipt",
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
    # Source-specific streams can be considered for user-authorized spending
    # changes.  Category aggregates are forecast-only reserves and must never
    # accidentally turn an arbitrary grocery purchase into a reducible stream.
    kind: str = "named"
    covered_event_ids: frozenset[str] = frozenset()

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
    anchor_date = cash_date(ordered[-1])
    starts_on = None
    if not 5 <= median <= 35 or any(abs(gap - median) > 3 for gap in gaps):
        # Settlement can be delayed for a single salary occurrence even when
        # the underlying dated payroll stream remains regular.  Recover only
        # when the source event dates themselves form a stable cadence; this
        # preserves settlement-date cadence for ordinary cash streams.
        event_gaps = [(right.event_date - left.event_date).days for left, right in zip(ordered, ordered[1:])]
        event_median = sorted(event_gaps)[len(event_gaps) // 2]
        if not 5 <= event_median <= 35 or any(abs(gap - event_median) > 3 for gap in event_gaps):
            return None
        median = event_median
        anchor_date = ordered[-1].event_date
        starts_on = anchor_date
    representative = ordered[-1]
    amounts = [event.amount for event in ordered if event.amount is not None]
    if amounts:
        # Forecast conservatively when a supported stream varies: the largest
        # recurring debit or smallest recurring credit is the safer bound.
        forecast_amount = max(amounts) if representative.direction == "debit" else min(amounts)
        representative = replace(representative, amount=forecast_amount)
    return RecurringStream(representative, median, starts_on=starts_on, anchor_day=anchor_date.day,
                           covered_event_ids=frozenset(event.event_id for event in ordered))


# These categories are inherently variable, essential debits.  They are
# deliberately narrow: income and discretionary merchant categories must not
# acquire a recurrence just because several rows share a broad category.
VARIABLE_ESSENTIAL_DEBIT_CATEGORIES = frozenset({
    # These purchase categories normally rotate merchants, so source-name
    # matching cannot see their regular commitment. Other categories retain
    # source-specific detection to avoid turning occasional bills into a
    # fabricated recurring obligation.
    "groceries", "grocery", "transport", "transportation",
})


def _variable_category_streams(events: list[FinancialEvent], covered: set[str], as_of: date | None) -> list[RecurringStream]:
    """Build conservative monthly debit reserves from *uncovered* merchants.

    A named subscription/rent stream owns its own history.  Other essential
    debit activity is grouped by calendar month so changing merchants does not
    erase an ordinary grocery or transport commitment.  Three completed
    months are required and the high observed monthly total is reserved.
    """
    if as_of is None:
        return []
    by_category_currency: dict[tuple[str, str], dict[tuple[int, int], list[FinancialEvent]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        if (event.event_id in covered or event.status != EventStatus.SETTLED
                or event.amount is None or event.direction != "debit"
                or event.category.strip().lower() not in VARIABLE_ESSENTIAL_DEBIT_CATEGORIES
                # A current partial month is not a completed historical
                # period and cannot justify reserving a full monthly total.
                or cash_date(event) >= date(as_of.year, as_of.month, 1)):
            continue
        by_category_currency[(event.category, event.currency)][(cash_date(event).year, cash_date(event).month)].append(event)
    streams: list[RecurringStream] = []
    for (_, _), months in by_category_currency.items():
        all_events = sorted((event for values in months.values() for event in values), key=cash_date)
        # Grocery and transport records in the supplied data commonly rotate
        # merchants every week. A calendar-month total is both late and too
        # coarse for a payment-safety timeline. When the last six purchases
        # support a stable 5–14 day cadence, reserve the conservative largest
        # observed amount at that actual cadence instead.
        recent = all_events[-6:]
        if len(recent) >= 6:
            gaps = [(cash_date(right) - cash_date(left)).days for left, right in zip(recent, recent[1:])]
            median_gap = sorted(gaps)[len(gaps) // 2]
            if 5 <= median_gap <= 14 and all(abs(gap - median_gap) <= 2 for gap in gaps):
                representative = replace(recent[-1], amount=max((event.amount or 0) for event in recent),
                                         flexibility="fixed")
                streams.append(RecurringStream(representative, median_gap, kind="variable_category",
                                               covered_event_ids=frozenset(event.event_id for event in all_events)))
                continue
        if len(months) < 3:
            continue
        # Do not invent a monthly reserve from sparse/seasonal history.
        ordered_months = sorted(months)[-3:]
        if len(ordered_months) < 3 or any((right[0] * 12 + right[1]) - (left[0] * 12 + left[1]) != 1
               for left, right in zip(ordered_months, ordered_months[1:])):
            continue
        monthly_totals = [sum((event.amount or 0) for event in months[key]) for key in ordered_months]
        latest_events = months[ordered_months[-1]]
        latest = max(latest_events, key=cash_date)
        # Use the first of the month for a reserve: it is intentionally
        # conservative and avoids assuming a later merchant purchase date.
        representative = replace(latest, amount=max(monthly_totals), flexibility="fixed")
        streams.append(RecurringStream(representative, 30,
                                       anchor_day=1, kind="variable_category",
                                       covered_event_ids=frozenset(event.event_id for values in months.values() for event in values)))
    return streams


def detect_recurrences(events, as_of: date | None = None, *, include_variable_aggregates: bool = False) -> tuple[RecurringStream, ...]:
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
    # Raw descriptions often include lifecycle words ("Acme payroll",
    # "salary from Acme", "confirmed Acme payroll").  Identity is the
    # normalized named source, never the raw description.  Empty source keys
    # are intentionally not pooled: that would merge unrelated salary credits.
    exact: dict[tuple[str, str, str], list[FinancialEvent]] = {}
    for event in eligible:
        key = event_stream_key(event)
        source = key.rsplit("|", 1)[-1]
        # A source-less label may still recur when it is exactly the same
        # label ("Payroll" every month), but never pool different generic
        # salary descriptions into fabricated income.
        generic_label = "" if source else re.sub(r"[^a-z0-9]+", " ", event.description.casefold()).strip()
        exact.setdefault((key, event.currency, generic_label), []).append(event)
    streams: list[RecurringStream] = []
    covered_event_ids: set[str] = set()
    for key, values in exact.items():
        if (key[0] in terminal
                or event_stream_key(values[0]) in pending_keys):
            continue
        stream = _stable_stream(values, 3)
        if stream:
            streams.append(stream)
            covered_event_ids.update(event.event_id for event in values)
    # Category aggregation is intentionally opt-in. The contest data model
    # does not mark which profiles authorize a monthly category budget or its
    # intra-month timing; callers with such a policy can use the conservative
    # detector without silently converting merchant history into obligations.
    if include_variable_aggregates:
        streams.extend(_variable_category_streams(eligible, covered_event_ids, as_of))
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
                                                 anchor_day=override.event_date.day, kind=stream.kind,
                                                 covered_event_ids=stream.covered_event_ids | frozenset({override.event_id}))
                break
        else:
            # A stream explicitly marked recurring by evidence needs no
            # historical rows. One-time evidence uses recurrence_days=0 and
            # is deliberately excluded above.
            streams.append(RecurringStream(override, override.recurrence_days or 30,
                                           anchor_day=override.event_date.day,
                                           covered_event_ids=frozenset({override.event_id})))
    return tuple(streams)


def recurring_event_ids(events) -> frozenset[str]:
    return frozenset(stream.representative.event_id for stream in detect_recurrences(events)
                     if stream.kind == "named")
