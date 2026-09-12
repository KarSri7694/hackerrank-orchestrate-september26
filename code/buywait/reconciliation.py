from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal

from .domain import EventStatus, EvidenceFact, FinancialEvent
from .recurrence import event_stream_key, stream_key

_AMENDMENT_EFFECTS = {"amend", "amendment", "amount_amendment", "date_amendment", "delay", "delayed", "confirm", "confirmed", "add_fact", "settle", "settled", "image_amount", "stream_update", "aggregate_stream_update"}
_CANCELLATION_EFFECTS = {"cancel", "cancelled", "cancellation", "terminate", "terminated"}
_STATUS_RANK = {EventStatus.SETTLED: 3, EventStatus.SCHEDULED: 2, EventStatus.PENDING: 1, None: 0}


def _timestamp(fact: EvidenceFact) -> tuple[int, int, int, int, int, int, int]:
    value = fact.sent_at or datetime.min
    return value.year, value.month, value.day, value.hour, value.minute, value.second, value.microsecond


def _rank(fact: EvidenceFact, event: FinancialEvent, field: str, position: int):
    explicit = 3 if fact.effect in _CANCELLATION_EFFECTS else 2 if fact.effect in _AMENDMENT_EFFECTS else 1
    settled = _STATUS_RANK.get(fact.status, 0)
    safer = Decimal("0")
    if field == "amount" and fact.amount is not None:
        safer = fact.amount if event.direction == "debit" else -fact.amount
    if field == "date" and fact.effective_date is not None:
        ordinal = Decimal(fact.effective_date.toordinal())
        safer = -ordinal if event.direction == "debit" else ordinal
    return explicit, settled, safer, fact.confidence, position


def _pick(items, event: FinancialEvent, field: str):
    if not items:
        return None
    newest_by_source: dict[str, tuple[int, EvidenceFact]] = {}
    for item in items:
        source = item[1].source_type or item[1].evidence_id
        current = newest_by_source.get(source)
        if current is None or (_timestamp(item[1]), item[0]) > (_timestamp(current[1]), current[0]):
            newest_by_source[source] = item
    return max(newest_by_source.values(), key=lambda item: _rank(item[1], event, field, item[0]))[1]


def _apply_event_facts(event: FinancialEvent, related: list[tuple[int, EvidenceFact]]) -> FinancialEvent:
    if not related:
        return event
    if any(f.effect in _CANCELLATION_EFFECTS or f.status == EventStatus.CANCELLED for _, f in related):
        return replace(event, status=EventStatus.CANCELLED)
    amount_fact = _pick([item for item in related if item[1].amount is not None], event, "amount")
    date_fact = _pick([item for item in related if item[1].effective_date is not None], event, "date")
    status_fact = _pick([item for item in related if item[1].status is not None], event, "status")
    updated = event
    if amount_fact:
        updated = replace(updated, amount=amount_fact.amount, currency=amount_fact.currency or updated.currency)
    if date_fact:
        updated = replace(updated, event_date=date_fact.effective_date, settlement_date=date_fact.effective_date)
    if status_fact:
        updated = replace(updated, status=status_fact.status)
    recurrence_fact = _pick([item for item in related if item[1].recurring is not None or item[1].recurrence_days is not None], event, "date")
    if recurrence_fact:
        updated = replace(updated, recurrence_days=(0 if recurrence_fact.recurring is False else recurrence_fact.recurrence_days))
    return updated


def _stream_event(fact: EvidenceFact, user_id: str, position: int) -> FinancialEvent | None:
    if not fact.category or not fact.direction or not fact.effective_date:
        return None
    terminal = fact.effect in _CANCELLATION_EFFECTS
    return FinancialEvent(
        event_id=f"evidence:{fact.evidence_id}:{position}", user_id=user_id,
        event_type="aggregate_stream_update" if fact.effect == "aggregate_stream_update" else ("income" if fact.direction == "credit" else "expense"),
        description=fact.stream_source or fact.description or ("Terminal stream evidence" if terminal else "Evidence stream update"),
        category=fact.category, direction=fact.direction, amount=fact.amount,
        currency=fact.currency or "", event_date=fact.effective_date,
        settlement_date=fact.effective_date if fact.status == EventStatus.SETTLED else None,
        status=EventStatus.CANCELLED if terminal else (fact.status or EventStatus.SCHEDULED),
        linked_event_id=None, flexibility=fact.flexibility or "fixed",
        minimum_allowed_amount=fact.minimum_allowed_amount,
        recurrence_days=0 if fact.recurring is False else fact.recurrence_days,
    )


def _deduplicate_lifecycles(events: list[FinancialEvent]) -> tuple[FinancialEvent, ...]:
    unique: dict[tuple, FinancialEvent] = {}
    for event in events:
        key = (event.user_id, event.event_type, event.category, event.direction, event.amount, event.currency,
               event.event_date, event.settlement_date, event.status, event.linked_event_id)
        unique.setdefault(key, event)
    rows = list(unique.values())
    by_id = {event.event_id: event for event in rows}
    discarded: set[str] = set()
    for event in rows:
        prior = by_id.get(event.linked_event_id or "")
        if not prior or prior.direction != event.direction or prior.currency != event.currency:
            continue
        if prior.amount != event.amount and event.status != EventStatus.SETTLED:
            continue
        prior_rank = (_STATUS_RANK.get(prior.status, 0), prior.settlement_date or prior.event_date)
        event_rank = (_STATUS_RANK.get(event.status, 0), event.settlement_date or event.event_date)
        discarded.add(prior.event_id if event_rank >= prior_rank else event.event_id)
    return tuple(event for event in rows if event.event_id not in discarded)


def _lifecycle_component(events: list[FinancialEvent], event_id: str) -> set[str]:
    """Follow both directions of linked_event_id to retain a transfer pair."""
    by_id = {event.event_id: event for event in events}
    component, pending = set(), [event_id]
    while pending:
        current = pending.pop()
        if current in component or current not in by_id:
            continue
        component.add(current)
        linked = by_id[current].linked_event_id
        if linked:
            pending.append(linked)
        pending.extend(event.event_id for event in events if event.linked_event_id == current)
    return component


def _mark_internal_transfer_pair(events: list[FinancialEvent], event_id: str) -> list[FinancialEvent]:
    component = _lifecycle_component(events, event_id)
    directions = {event.direction for event in events if event.event_id in component}
    # A one-sided assertion is not enough: leave it untouched rather than
    # inventing a debit or credit by suppressing only one side.
    if not {"debit", "credit"}.issubset(directions):
        return events
    return [replace(event, event_type="internal_transfer") if event.event_id in component else event for event in events]


def _event_day(event: FinancialEvent):
    return event.settlement_date or event.event_date


def _unique_transfer_pair(events: list[FinancialEvent], fact: EvidenceFact) -> tuple[str, str] | None:
    """Find one strong debit/credit match for request-level transfer evidence."""
    candidates: list[tuple[tuple[int, int], str, str]] = []
    debits = [event for event in events if event.direction == "debit" and event.amount is not None]
    credits = [event for event in events if event.direction == "credit" and event.amount is not None]
    for debit in debits:
        for credit in credits:
            if debit.amount != credit.amount or debit.currency != credit.currency:
                continue
            if fact.amount is not None and debit.amount != fact.amount:
                continue
            if fact.currency is not None and debit.currency != fact.currency:
                continue
            distance = abs((_event_day(debit) - _event_day(credit)).days)
            if distance > 3:
                continue
            if fact.effective_date and min(abs((_event_day(debit) - fact.effective_date).days), abs((_event_day(credit) - fact.effective_date).days)) > 3:
                continue
            linked = credit.linked_event_id == debit.event_id or debit.linked_event_id == credit.event_id
            # A linked lifecycle is strongest; otherwise same amount/currency
            # and an exact/near date can qualify only when unique.
            candidates.append(((1 if linked else 0, -distance), debit.event_id, credit.event_id))
    if not candidates:
        return None
    best_score = max(candidate[0] for candidate in candidates)
    best = [(debit, credit) for score, debit, credit in candidates if score == best_score]
    return best[0] if len(best) == 1 else None


def reconcile_events(events, facts) -> tuple[FinancialEvent, ...]:
    facts = tuple(facts)
    by_event: dict[str, list[tuple[int, EvidenceFact]]] = {}
    for position, fact in enumerate(facts):
        if fact.related_event_id:
            by_event.setdefault(fact.related_event_id, []).append((position, fact))
    resolved = [_apply_event_facts(event, by_event.get(event.event_id, [])) for event in events]
    for fact in facts:
        if fact.effect != "internal_transfer":
            continue
        if fact.related_event_id:
            resolved = _mark_internal_transfer_pair(resolved, fact.related_event_id)
        else:
            pair = _unique_transfer_pair(resolved, fact)
            if pair:
                # Both IDs are in the same inferred two-event lifecycle for
                # this fact. Marking the explicit pair avoids any category-
                # wide or one-sided suppression.
                resolved = [replace(event, event_type="internal_transfer") if event.event_id in pair else event for event in resolved]
    # An aggregate update is authoritative only prospectively. Do not alter
    # settled history, but prevent supplied future rows from being counted in
    # addition to the new aggregate recurrence.
    for fact in facts:
        if fact.effect != "aggregate_stream_update" or not (fact.category and fact.direction and fact.effective_date):
            continue
        resolved = [replace(event, event_type="aggregate_superseded")
                    if event.category == fact.category and event.direction == fact.direction
                    and event.status != EventStatus.SETTLED and _event_day(event) >= fact.effective_date
                    else event for event in resolved]
    # Employment, rent, and similar stream evidence often has no one-to-one
    # event row. A terminal stream fact must stop inferred recurrence as well
    # as supplied future occurrences; the supplied opening balance already
    # accounts for cash settled before the request horizon.
    for fact in facts:
        if fact.related_event_id is not None or fact.effect not in _CANCELLATION_EFFECTS:
            continue
        fact_key = _fact_stream_key(fact)
        if not fact.category or not fact.direction or not fact_key:
            continue
        resolved = [replace(event, status=EventStatus.CANCELLED)
                    if event_stream_key(event) == fact_key
                    else event for event in resolved]
    user_id = resolved[0].user_id if resolved else ""
    for position, fact in enumerate(facts):
        if fact.related_event_id is None:
            stream = _stream_event(fact, user_id, position)
            if stream:
                resolved.append(stream)
    return _deduplicate_lifecycles(resolved)


def reconcile_context(context):
    return replace(context, events=reconcile_events(context.events, context.evidence_facts))


def _fact_stream_key(fact: EvidenceFact) -> str | None:
    if fact.stream_source and fact.category and fact.direction:
        return stream_key(fact.category, fact.direction, fact.stream_source)
    # Read legacy cached facts written before stream_source existed.
    if fact.stream_key:
        return fact.stream_key
    if fact.description and fact.category and fact.direction:
        return stream_key(fact.category, fact.direction, fact.description)
    return None
