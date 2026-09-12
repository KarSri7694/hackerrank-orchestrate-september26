from __future__ import annotations

from dataclasses import replace

from .domain import EventStatus, EvidenceFact, FinancialEvent


_AMENDMENT_EFFECTS = {"amend", "amendment", "amount_amendment", "date_amendment", "delay", "delayed", "confirm", "confirmed", "add_fact", "settle", "settled", "image_amount"}
_CANCELLATION_EFFECTS = {"cancel", "cancelled", "cancellation"}


def _fact_rank(fact: EvidenceFact, position: int) -> tuple[int, float, int]:
    """Rank explicit evidence deterministically; position breaks same-confidence ties."""
    explicit = 3 if fact.effect in _CANCELLATION_EFFECTS else 2 if fact.effect in _AMENDMENT_EFFECTS else 1
    return explicit, fact.confidence, position


def reconcile_events(events: tuple[FinancialEvent, ...] | list[FinancialEvent], facts: tuple[EvidenceFact, ...] | list[EvidenceFact]) -> tuple[FinancialEvent, ...]:
    """Apply structured evidence to raw events without doing financial arithmetic.

    Evidence is scoped to an existing event. Explicit cancellation wins over
    all other facts for that event. Amount/date/status amendments are selected
    by explicit-effect priority, confidence, then stable input order.
    """
    by_event: dict[str, list[tuple[int, EvidenceFact]]] = {}
    for position, fact in enumerate(facts):
        if fact.related_event_id:
            by_event.setdefault(fact.related_event_id, []).append((position, fact))

    resolved: list[FinancialEvent] = []
    for event in events:
        related = by_event.get(event.event_id, [])
        if not related:
            resolved.append(event)
            continue
        cancellation = [item for item in related if item[1].effect in _CANCELLATION_EFFECTS or item[1].status == EventStatus.CANCELLED]
        if cancellation:
            resolved.append(replace(event, status=EventStatus.CANCELLED))
            continue

        chosen = max(related, key=lambda item: _fact_rank(item[1], item[0]))[1]
        amount_facts = [item for item in related if item[1].amount is not None and item[1].effect in _AMENDMENT_EFFECTS]
        date_facts = [item for item in related if item[1].effective_date is not None and item[1].effect in _AMENDMENT_EFFECTS]
        status_facts = [item for item in related if item[1].status is not None and item[1].effect in _AMENDMENT_EFFECTS]
        amount_fact = max(amount_facts, key=lambda item: _fact_rank(item[1], item[0]))[1] if amount_facts else None
        date_fact = max(date_facts, key=lambda item: _fact_rank(item[1], item[0]))[1] if date_facts else None
        status_fact = max(status_facts, key=lambda item: _fact_rank(item[1], item[0]))[1] if status_facts else None

        updated = event
        if amount_fact is not None:
            updated = replace(updated, amount=amount_fact.amount, currency=amount_fact.currency or updated.currency)
        if date_fact is not None:
            updated = replace(updated, event_date=date_fact.effective_date)
            if status_fact and status_fact.status in {EventStatus.SETTLED, EventStatus.PENDING, EventStatus.SCHEDULED}:
                updated = replace(updated, settlement_date=date_fact.effective_date)
        if status_fact is not None:
            updated = replace(updated, status=status_fact.status)
            if status_fact.status == EventStatus.SETTLED and date_fact is not None:
                updated = replace(updated, settlement_date=date_fact.effective_date)
        resolved.append(updated)
    return tuple(resolved)


def reconcile_context(context):
    return replace(context, events=reconcile_events(context.events, context.evidence_facts))
