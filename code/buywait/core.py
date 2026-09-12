from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations

from .domain import (AffordabilityStatus, DecisionCore, EventStatus, FinancialEvent,
                     Payment, PaymentMethod, PlanCandidate, RequestContext,
                     SimulationResult, SpendingChange)
from .reconciliation import reconcile_context

ZERO = Decimal("0")


def _cash(event: FinancialEvent, home_currency: str, fx, on_date: date) -> Decimal:
    if event.amount is None:
        return ZERO
    value = fx.convert(event.amount, event.currency, home_currency, on_date)
    return value if event.direction == "credit" else -value


def _effective_events(ctx: RequestContext, fx) -> list[tuple[date, Decimal, str, FinancialEvent]]:
    request_date = ctx.request.request_date
    end = request_date + timedelta(days=89)
    rows: list[tuple[date, Decimal, str, FinancialEvent]] = []
    for event in ctx.events:
        if event.status in {EventStatus.FAILED, EventStatus.CANCELLED, EventStatus.UNREALIZED} or event.amount is None:
            continue
        if event.status == EventStatus.PENDING and event.direction == "credit":
            continue
        event_day = event.settlement_date or event.event_date
        if event_day >= request_date and event_day <= end:
            rows.append((event_day, _cash(event, ctx.profile.home_currency, fx, event_day), event.event_id, event))
    # Infer recurrence only from repeated same-user/category/direction records.
    groups: dict[tuple[str, str, str], list[FinancialEvent]] = defaultdict(list)
    for event in ctx.events:
        if event.status == EventStatus.SETTLED and event.amount is not None:
            groups[(event.category, event.description, event.direction)].append(event)
    for group in groups.values():
        ordered = sorted(group, key=lambda e: e.event_date)
        if len(ordered) < 2:
            continue
        gaps = [(b.event_date - a.event_date).days for a, b in zip(ordered, ordered[1:])]
        cadence = round(sum(gaps) / len(gaps))
        if cadence < 20 or cadence > 45:
            continue
        last = ordered[-1]
        next_day = last.event_date
        while next_day < request_date:
            next_day += timedelta(days=cadence)
        while next_day <= end:
            if not any(x[2] == last.event_id and x[0] == next_day for x in rows):
                rows.append((next_day, _cash(last, ctx.profile.home_currency, fx, next_day), last.event_id, last))
            next_day += timedelta(days=cadence)
    return rows


def _change_amount(event: FinancialEvent, change: SpendingChange, fx, home: str, on_date: date) -> Decimal:
    original = abs(_cash(event, home, fx, on_date))
    if change.action == "stop":
        return ZERO
    return change.new_amount if change.new_amount is not None else original


def simulate(ctx: RequestContext, fx, payments: tuple[Payment, ...] = (), changes: tuple[SpendingChange, ...] = ()) -> SimulationResult:
    ctx = reconcile_context(ctx)
    changes_by_id = {c.event_id: c for c in changes}
    timeline = _effective_events(ctx, fx)
    balances = ctx.profile.current_available_balance
    by_day: dict[date, list[tuple[Decimal, str, FinancialEvent | None]]] = defaultdict(list)
    for day, amount, event_id, event in timeline:
        by_day[day].append((amount, event_id, event))
    for payment in payments:
        by_day[payment.date].append((-payment.amount, "request_payment", None))
    end = ctx.request.request_date + timedelta(days=89)
    minimum = ctx.profile.minimum_balance_to_keep
    lowest = balances
    violation = None
    for day in sorted(d for d in by_day if ctx.request.request_date <= d <= end):
        for amount, event_id, event in by_day[day]:
            if event is not None and event_id in changes_by_id and event.direction == "debit" and event.flexibility != "fixed":
                amount = -_change_amount(event, changes_by_id[event_id], fx, ctx.profile.home_currency, day)
            balances += amount
            lowest = min(lowest, balances)
            if balances < minimum and violation is None:
                violation = day
    shortfall = max(ZERO, minimum - lowest)
    return SimulationResult(violation is None, lowest, minimum, violation, shortfall, balances)


def baseline(ctx: RequestContext, fx) -> tuple[Decimal, date | None]:
    ctx = reconcile_context(ctx)
    base = simulate(ctx, fx)
    # The minimum projected balance already includes all required future cash flows.
    safe_today = max(ZERO, base.minimum_projected_balance - ctx.profile.minimum_balance_to_keep)
    safe_today = min(ctx.request.requested_amount, safe_today)
    earliest = None
    for offset in range(90):
        day = ctx.request.request_date + timedelta(days=offset)
        result = simulate(ctx, fx, (Payment(day, ctx.request.requested_amount),))
        if result.safe:
            earliest = day
            break
    return safe_today, earliest


def expand_option(option) -> tuple[Payment, ...]:
    frequency = option.payment_frequency_days or 0
    return tuple(Payment(option.first_payment_date + timedelta(days=frequency * i), option.payment_amount) for i in range(option.number_of_payments))


def legal_changes(ctx: RequestContext, events: list[FinancialEvent], maximum: int = 3) -> list[tuple[SpendingChange, ...]]:
    ctx = reconcile_context(ctx)
    events = list(ctx.events)
    candidates: list[tuple[SpendingChange, ...]] = [()]
    for event in events:
        if event.status != EventStatus.SETTLED or not event.flexibility or event.flexibility == "fixed":
            continue
        options: list[SpendingChange] = []
        if event.category in ctx.profile.stoppable_categories and event.flexibility in {"stoppable", "reducible_or_stoppable"}:
            options.append(SpendingChange(event.event_id, "stop"))
        if event.category in ctx.profile.reducible_categories and event.flexibility in {"reducible", "reducible_or_stoppable"}:
            current = abs(event.amount or ZERO)
            minimum = event.minimum_allowed_amount or ZERO
            if current > minimum:
                options.append(SpendingChange(event.event_id, "reduce_to", minimum))
        for change in options:
            candidates.append((change,))
    for size in (2, 3):
        for combo in combinations([c for group in candidates if len(group) == 1 for c in group], size):
            if len({c.event_id for c in combo}) == size:
                candidates.append(tuple(combo))
    return [c for c in candidates if len(c) <= maximum]


def solve(ctx: RequestContext, fx) -> DecisionCore:
    ctx = reconcile_context(ctx)
    safe_today, earliest = baseline(ctx, fx)
    req = ctx.request
    candidates: list[PlanCandidate] = []
    if PaymentMethod.FULL in ctx.profile.accepted_payment_methods:
        candidates.append(PlanCandidate(PaymentMethod.FULL, (Payment(req.request_date, req.requested_amount),), req.requested_amount))
    if PaymentMethod.PARTIAL in ctx.profile.accepted_payment_methods and req.allows_partial_payment and safe_today > ZERO and safe_today < req.requested_amount and earliest and earliest <= req.desired_completion_date:
        candidates.append(PlanCandidate(PaymentMethod.PARTIAL, (Payment(req.request_date, safe_today), Payment(earliest, req.requested_amount - safe_today)), req.requested_amount))
    if PaymentMethod.INSTALLMENTS in ctx.profile.accepted_payment_methods:
        for option in ctx.payment_options:
            if option.payment_method != PaymentMethod.INSTALLMENTS:
                continue
            payments = expand_option(option)
            months = (payments[-1].date - payments[0].date).days / 30 if payments else 0
            if (ctx.profile.max_installment_months is None or months <= ctx.profile.max_installment_months) and payments[-1].date <= req.desired_completion_date:
                candidates.append(PlanCandidate(PaymentMethod.INSTALLMENTS, payments, option.total_payable_amount, payment_option_id=option.payment_option_id))
    if PaymentMethod.FULL in ctx.profile.accepted_payment_methods and earliest and earliest > req.request_date and earliest <= req.desired_completion_date:
        candidates.append(PlanCandidate(PaymentMethod.WAIT, (Payment(earliest, req.requested_amount),), req.requested_amount))
    safe_candidates = [c for c in candidates if simulate(ctx, fx, c.payments).safe]
    if not safe_candidates:
        for candidate in candidates:
            for changes in legal_changes(ctx, list(ctx.events)):
                if changes and simulate(ctx, fx, candidate.payments, changes).safe:
                    safe_candidates.append(PlanCandidate(candidate.method, candidate.payments, candidate.total_payable, changes, candidate.payment_option_id))
    if not safe_candidates:
        return DecisionCore(req.request_id, safe_today, AffordabilityStatus.NOT_AFFORDABLE, PaymentMethod.NOT_RECOMMENDED, (), earliest, (), {"minimum_balance": str(ctx.profile.minimum_balance_to_keep)})
    def rank(c: PlanCandidate):
        return (0 if c.payments[-1].date <= req.desired_completion_date else 1, 0 if not c.spending_changes else 1, c.total_payable, c.payments[0].date, len(c.payments), c.payment_option_id or "")
    winner = sorted(safe_candidates, key=rank)[0]
    status = AffordabilityStatus.NOW if winner.method == PaymentMethod.FULL and winner.payments[0].date == req.request_date and not winner.spending_changes else AffordabilityStatus.LATER if winner.method == PaymentMethod.WAIT else AffordabilityStatus.WITH_PLAN
    return DecisionCore(req.request_id, safe_today, status, winner.method, winner.payments, earliest, winner.spending_changes, {"minimum_balance": str(ctx.profile.minimum_balance_to_keep), "winner_total": str(winner.total_payable)})
