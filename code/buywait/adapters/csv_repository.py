from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from ..domain import (EventStatus, EvidenceReference, FinancialEvent, FinancialProfile,
                      FinancialRequest, PaymentMethod, PaymentOption, RequestContext)


def _text(value: str | None) -> str:
    return (value or "").strip()


def _date(value: str | None) -> date | None:
    value = _text(value)
    return date.fromisoformat(value) if value else None


def _decimal(value: str | None) -> Decimal | None:
    value = _text(value)
    return Decimal(value) if value else None


def _datetime(value: str | None) -> datetime | None:
    value = _text(value)
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _bool(value: str | None) -> bool:
    return _text(value).lower() in {"true", "1", "yes", "y"}


def _pipe(value: str | None) -> tuple[str, ...]:
    return tuple(x.strip() for x in _text(value).split("|") if x.strip())


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _assert_unique(rows: list[dict[str, str]], field: str, label: str) -> None:
    """Fail at ingestion instead of silently replacing a duplicate row."""
    values = [_text(row.get(field)) for row in rows]
    if any(not value for value in values):
        raise ValueError(f"{label} contains a blank {field}")
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicate {field}: {', '.join(duplicates)}")


def _assert_event_links(events: dict[str, FinancialEvent]) -> None:
    """Validate lifecycle links before reconciliation relies on them."""
    for event in events.values():
        if not event.linked_event_id:
            continue
        linked = events.get(event.linked_event_id)
        if linked is None:
            raise ValueError(
                f"financial_events.csv event {event.event_id} references unknown linked_event_id "
                f"{event.linked_event_id}"
            )
        if linked.user_id != event.user_id:
            raise ValueError(
                f"financial_events.csv event {event.event_id} links across users to {event.linked_event_id}"
            )
        if linked.event_id == event.event_id:
            raise ValueError(f"financial_events.csv event {event.event_id} cannot link to itself")


def _assert_evidence_event_links(rows: list[dict[str, str]], events: dict[str, FinancialEvent], label: str) -> None:
    """Ensure evidence cannot point at an absent or unrelated ledger row."""
    for row in rows:
        related_event_id = _text(row.get("related_event_id"))
        if not related_event_id:
            continue
        event = events.get(related_event_id)
        if event is None:
            raise ValueError(f"{label} references unknown related_event_id: {related_event_id}")
        user_id = _text(row.get("user_id"))
        if user_id and event.user_id != user_id:
            raise ValueError(f"{label} related_event_id crosses users: {related_event_id}")


class CsvRepository:
    def __init__(self, dataset_dir: str | Path):
        root = Path(dataset_dir)
        requests = _read(root / "requests.csv")
        profiles = _read(root / "financial_profiles.csv")
        events = _read(root / "financial_events.csv")
        options = _read(root / "request_payment_options.csv")
        messages = _read(root / "messages.csv")
        images = _read(root / "images.csv")

        _assert_unique(requests, "request_id", "requests.csv")
        _assert_unique(profiles, "user_id", "financial_profiles.csv")
        _assert_unique(events, "event_id", "financial_events.csv")
        _assert_unique(options, "payment_option_id", "request_payment_options.csv")
        _assert_unique(messages, "message_id", "messages.csv")
        _assert_unique(images, "image_id", "images.csv")

        self.requests = {
            r["request_id"]: FinancialRequest(
                request_id=_text(r["request_id"]), user_id=_text(r["user_id"]),
                request_date=_date(r["request_date"]), request_type=_text(r["request_type"]),
                requested_amount=Decimal(_text(r["requested_amount"])),
                desired_completion_date=_date(r["desired_completion_date"]),
                allows_partial_payment=_bool(r["allows_partial_payment"]), request_text=_text(r["request_text"])
            ) for r in requests
        }
        self.profiles = {
            r["user_id"]: FinancialProfile(
                user_id=_text(r["user_id"]), home_currency=_text(r["home_currency"]),
                current_available_balance=Decimal(_text(r["current_available_balance"])),
                minimum_balance_to_keep=Decimal(_text(r["minimum_balance_to_keep"])),
                financial_priorities=_pipe(r["financial_priorities"]),
                protected_categories=frozenset(_pipe(r["expense_categories_to_protect"])),
                reducible_categories=frozenset(_pipe(r["expense_categories_user_is_willing_to_reduce"])),
                stoppable_categories=frozenset(_pipe(r["expense_categories_user_is_willing_to_stop"])),
                accepted_payment_methods=frozenset(PaymentMethod(x) for x in _pipe(r["payment_methods_user_will_consider"])),
                max_installment_months=int(r["max_installment_months"]) if _text(r["max_installment_months"]) else None,
            ) for r in profiles
        }
        missing_profiles = sorted({r["user_id"].strip() for r in requests if r["user_id"].strip() not in self.profiles})
        if missing_profiles:
            raise ValueError(f"requests.csv references missing profiles: {', '.join(missing_profiles)}")
        self.events = {}
        self.events_by_user: dict[str, list[FinancialEvent]] = defaultdict(list)
        for r in events:
            if _text(r["user_id"]) not in self.profiles:
                raise ValueError(f"financial_events.csv references unknown user_id: {_text(r['user_id'])}")
            event = FinancialEvent(
                event_id=_text(r["event_id"]), user_id=_text(r["user_id"]), event_type=_text(r["event_type"]),
                description=_text(r["description"]), category=_text(r["category"]), direction=_text(r["direction"]),
                amount=_decimal(r["amount"]), currency=_text(r["currency"]), event_date=_date(r["event_date"]),
                settlement_date=_date(r["settlement_date"]), status=EventStatus(_text(r["status"])),
                linked_event_id=_text(r["linked_event_id"]) or None, flexibility=_text(r["flexibility"]),
                minimum_allowed_amount=_decimal(r["minimum_allowed_amount"]),
            )
            self.events[event.event_id] = event
            self.events_by_user[event.user_id].append(event)
        _assert_event_links(self.events)
        self.options_by_request: dict[str, list[PaymentOption]] = defaultdict(list)
        for r in options:
            option = PaymentOption(
                payment_option_id=_text(r["payment_option_id"]), request_id=_text(r["request_id"]),
                payment_method=PaymentMethod(_text(r["payment_method"])), payment_amount=Decimal(_text(r["payment_amount"])),
                number_of_payments=int(r["number_of_payments"]), first_payment_date=_date(r["first_payment_date"]),
                payment_frequency_days=int(r["payment_frequency_days"]) if _text(r["payment_frequency_days"]) else None,
                financing_fee=Decimal(_text(r["financing_fee"])), total_payable_amount=Decimal(_text(r["total_payable_amount"])),
            )
            self.options_by_request[option.request_id].append(option)
        self.evidence_refs: dict[str, EvidenceReference] = {}
        for r in messages:
            if _text(r["user_id"]) not in self.profiles:
                raise ValueError(f"messages.csv references unknown user_id: {_text(r['user_id'])}")
            ref = EvidenceReference(
                evidence_id=_text(r["message_id"]), source_type=_text(r["source_type"]), user_id=_text(r["user_id"]),
                request_id=_text(r["request_id"]) or None, related_event_id=_text(r["related_event_id"]) or None,
                text=_text(r["message_text"]) or None, sent_at=_datetime(r.get("sent_at")),
            )
            self.evidence_refs[ref.evidence_id] = ref
        _assert_evidence_event_links(messages, self.events, "messages.csv")
        for r in images:
            if _text(r["user_id"]) not in self.profiles:
                raise ValueError(f"images.csv references unknown user_id: {_text(r['user_id'])}")
            image_path = root / "media" / "images" / f"{_text(r['image_id'])}.png"
            ref = EvidenceReference(
                evidence_id=_text(r["image_id"]), source_type="image", user_id=_text(r["user_id"]),
                request_id=_text(r["request_id"]) or None, related_event_id=_text(r["related_event_id"]) or None,
                image_path=str(image_path),
            )
            self.evidence_refs[ref.evidence_id] = ref
        _assert_evidence_event_links(images, self.events, "images.csv")
        self.evidence_by_user: dict[str, list[EvidenceReference]] = defaultdict(list)
        for ref in self.evidence_refs.values():
            self.evidence_by_user[ref.user_id].append(ref)

    def context(self, request_id: str) -> RequestContext:
        if request_id not in self.requests:
            raise KeyError(f"unknown request_id: {request_id}")
        request = self.requests[request_id]
        profile = self.profiles[request.user_id]
        event_ids = {event.event_id for event in self.events_by_user[request.user_id]}
        # Evidence is known only once it has been sent.  Requests are
        # date-granular, so a message on the request date is admissible.
        candidates = [
            ref for ref in self.evidence_by_user[request.user_id]
            if (ref.sent_at is None or ref.sent_at.date() <= request.request_date)
            and (ref.request_id in {None, request_id} or ref.related_event_id in event_ids)
        ]
        def relevance(ref: EvidenceReference):
            # Request-specific evidence is strongest, then evidence tied to a
            # financial event, then broad user-level stream evidence.
            tier = 0 if ref.request_id == request_id else 1 if ref.related_event_id in event_ids else 2
            sent = ref.sent_at or datetime.min
            return (tier, -sent.toordinal() if hasattr(sent, "toordinal") else 0, ref.evidence_id)
        refs = tuple(sorted(candidates, key=relevance))
        return RequestContext(request, profile, tuple(self.events_by_user[request.user_id]), tuple(self.options_by_request[request_id]), refs)

    def event(self, event_id: str) -> FinancialEvent:
        return self.events[event_id]

    def evidence(self, evidence_id: str) -> EvidenceReference:
        return self.evidence_refs[evidence_id]

    def request_ids(self) -> tuple[str, ...]:
        return tuple(self.requests)
