from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from itertools import combinations

from .domain import (AffordabilityStatus, DecisionCore, EventStatus, FinancialEvent,
                     Payment, PaymentMethod, PlanCandidate, RequestContext,
                     SimulationResult, SpendingChange)
from .reconciliation import reconcile_context
from .recurrence import detect_recurrences, recurring_event_ids

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
        if event.event_type in {"internal_transfer", "aggregate_superseded"} or event.status in {EventStatus.FAILED, EventStatus.CANCELLED, EventStatus.UNREALIZED} or event.amount is None:
            continue
        if event.status == EventStatus.PENDING and event.direction == "credit":
            continue
        event_day = event.settlement_date or event.event_date
        if event.status == EventStatus.PENDING and event.direction == "debit" and event.settlement_date is None and event_day < request_date:
            event_day = request_date
        if event_day >= request_date and event_day <= end:
            rows.append((event_day, _cash(event, ctx.profile.home_currency, fx, event_day), event.event_id, event))
    for stream in detect_recurrences(ctx.events):
        last = stream.representative
        next_day = stream.starts_on or last.settlement_date or last.event_date
        while next_day < request_date:
            next_day = stream.next_date(next_day)
        while next_day <= end and (stream.ends_before is None or next_day < stream.ends_before):
            # An explicit event on the same date wins over an inferred one;
            # descriptions can differ when a bank/payroll system changes its
            # wording, so category/direction/date are the stable identity.
            same_stream_exists = any(x[0] == next_day and x[3].category == last.category and x[3].direction == last.direction for x in rows)
            if not same_stream_exists:
                rows.append((next_day, _cash(last, ctx.profile.home_currency, fx, next_day), last.event_id, last))
            next_day = stream.next_date(next_day)
    return rows


def _change_amount(event: FinancialEvent, change: SpendingChange, fx, home: str, on_date: date) -> Decimal:
    original = abs(_cash(event, home, fx, on_date))
    if change.action == "stop":
        return ZERO
    if change.new_amount is None:
        return original
    return fx.convert(change.new_amount, event.currency, home, on_date)


def validate_changes(ctx: RequestContext, changes: tuple[SpendingChange, ...]) -> None:
    if len(changes) > 3 or len({change.event_id for change in changes}) != len(changes):
        raise ValueError("at most three unique spending changes are allowed")
    events = {event.event_id: event for event in ctx.events}
    recurring = recurring_event_ids(ctx.events)
    for change in changes:
        event = events.get(change.event_id)
        if event is None or event.event_id not in recurring or event.direction != "debit":
            raise ValueError(f"event is not a recurring debit: {change.event_id}")
        if event.flexibility == "fixed" or event.category in ctx.profile.protected_categories:
            raise ValueError(f"event is not a legal flexible target: {change.event_id}")
        if change.action == "stop":
            if event.category not in ctx.profile.stoppable_categories or event.flexibility not in {"stoppable", "reducible_or_stoppable"} or change.new_amount is not None:
                raise ValueError(f"stop is not allowed: {change.event_id}")
        elif change.action == "reduce_to":
            minimum = event.minimum_allowed_amount or ZERO
            current = abs(event.amount or ZERO)
            if event.category not in ctx.profile.reducible_categories or event.flexibility not in {"reducible", "reducible_or_stoppable"}:
                raise ValueError(f"reduction is not allowed: {change.event_id}")
            if change.new_amount is None or not minimum <= change.new_amount <= current:
                raise ValueError(f"reduced amount is outside legal bounds: {change.event_id}")
        else:
            raise ValueError(f"unsupported spending change action: {change.action}")


def simulate(ctx: RequestContext, fx, payments: tuple[Payment, ...] = (), changes: tuple[SpendingChange, ...] = ()) -> SimulationResult:
    ctx = reconcile_context(ctx)
    validate_changes(ctx, changes)
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
    if option.number_of_payments < 1 or (option.number_of_payments > 1 and frequency <= 0):
        return ()
    return tuple(Payment(option.first_payment_date + timedelta(days=frequency * i), option.payment_amount) for i in range(option.number_of_payments))


def _calendar_months_between(start: date, end: date) -> int:
    """Return elapsed calendar months, charging a partial month as a month."""
    months = (end.year - start.year) * 12 + end.month - start.month
    return months + (1 if end.day > start.day else 0)


def _option_order(option_id: str | None) -> tuple[int, str]:
    if not option_id:
        return 10**12, ""
    match = re.search(r"(\d+)$", option_id)
    return (int(match.group(1)) if match else 10**12, option_id)


def legal_changes(ctx: RequestContext, events: list[FinancialEvent], maximum: int = 3) -> list[tuple[SpendingChange, ...]]:
    ctx = reconcile_context(ctx)
    event_by_id = {event.event_id: event for event in ctx.events}
    recurring_ids = recurring_event_ids(ctx.events)
    candidates: list[tuple[SpendingChange, ...]] = [()]
    for event_id in sorted(recurring_ids):
        event = event_by_id[event_id]
        if event.direction != "debit" or event.flexibility == "fixed":
            continue
        if event.category in ctx.profile.protected_categories:
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


def _optimized_changes(ctx: RequestContext, fx, payments: tuple[Payment, ...], maximum: int = 3) -> list[SpendingChange] | None:
    if simulate(ctx, fx, payments).safe:
        return []
    options = [change for group in legal_changes(ctx, list(ctx.events), maximum=1) for change in group]
    refined: list[SpendingChange] = []
    for change in options:
        if change.action != "reduce_to":
            refined.append(change)
            continue
        event = next(e for e in ctx.events if e.event_id == change.event_id)
        minimum = change.new_amount or ZERO
        current = abs(event.amount or ZERO)
        if not simulate(ctx, fx, payments, (change,)).safe:
            refined.append(change)
            continue
        # Find the largest safe amount (therefore the smallest reduction) by
        # repeatedly re-simulating the candidate. Values are expressed in the
        # event currency and are never allowed below its supplied floor.
        low, high = minimum, current
        for _ in range(40):
            if high - low <= Decimal("0.01"):
                break
            mid = (low + high) / 2
            probe = SpendingChange(change.event_id, "reduce_to", mid)
            if simulate(ctx, fx, payments, (probe,)).safe:
                low = mid
            else:
                high = mid
        refined.append(SpendingChange(change.event_id, "reduce_to", low.quantize(Decimal("0.01"), rounding=ROUND_DOWN)))
    options = refined
    safe: list[tuple[tuple[Decimal, int, tuple[str, ...]], tuple[SpendingChange, ...]]] = []
    for size in range(1, maximum + 1):
        for group in combinations(options, size):
            if len({change.event_id for change in group}) != size:
                continue
            adjusted = list(group)
            if not simulate(ctx, fx, payments, tuple(adjusted)).safe:
                continue
            # A reduction may need a companion change to become safe. Once a
            # safe combination exists, relax every reduction upward while the
            # complete combination remains safe.
            for index, change in enumerate(adjusted):
                if change.action != "reduce_to":
                    continue
                event = next(e for e in ctx.events if e.event_id == change.event_id)
                low, high = change.new_amount or ZERO, abs(event.amount or ZERO)
                for _ in range(40):
                    if high - low <= Decimal("0.01"):
                        break
                    mid = (low + high) / 2
                    probe = list(adjusted)
                    probe[index] = SpendingChange(change.event_id, "reduce_to", mid)
                    if simulate(ctx, fx, payments, tuple(probe)).safe:
                        low = mid
                    else:
                        high = mid
                adjusted[index] = SpendingChange(change.event_id, "reduce_to", low.quantize(Decimal("0.01"), rounding=ROUND_DOWN))
            adjusted_tuple = tuple(adjusted)
            if not simulate(ctx, fx, payments, adjusted_tuple).safe:
                continue
            reduction = ZERO
            for change in adjusted_tuple:
                event = next(e for e in ctx.events if e.event_id == change.event_id)
                reduced = ZERO if change.action == "stop" else (change.new_amount or ZERO)
                reduction += fx.convert(abs(event.amount or ZERO) - reduced, event.currency, ctx.profile.home_currency, ctx.request.request_date)
            safe.append(((reduction, size, tuple(c.event_id for c in adjusted_tuple)), adjusted_tuple))
    return list(sorted(safe, key=lambda item: item[0])[0][1]) if safe else None


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
            if not payments or payments[0].date < req.request_date or payments[-1].date > req.desired_completion_date:
                continue
            months = _calendar_months_between(payments[0].date, payments[-1].date)
            if ctx.profile.max_installment_months is None or months <= ctx.profile.max_installment_months:
                candidates.append(PlanCandidate(PaymentMethod.INSTALLMENTS, payments, option.total_payable_amount, payment_option_id=option.payment_option_id))
    if PaymentMethod.FULL in ctx.profile.accepted_payment_methods and earliest and earliest > req.request_date and earliest <= req.desired_completion_date:
        candidates.append(PlanCandidate(PaymentMethod.WAIT, (Payment(earliest, req.requested_amount),), req.requested_amount))
    # Candidate generation is permissive; only simulator-safe schedules may
    # enter ranking, including full, partial, wait, and installments.
    safe_candidates = [c for c in candidates if simulate(ctx, fx, c.payments).safe]
    if not safe_candidates:
        for candidate in candidates:
            changes = _optimized_changes(ctx, fx, candidate.payments)
            if changes:
                safe_candidates.append(PlanCandidate(candidate.method, candidate.payments, candidate.total_payable, tuple(changes), candidate.payment_option_id))
    if not safe_candidates:
        return DecisionCore(req.request_id, safe_today, AffordabilityStatus.NOT_AFFORDABLE, PaymentMethod.NOT_RECOMMENDED, (), earliest, (), {"minimum_balance": str(ctx.profile.minimum_balance_to_keep)})
    def rank(c: PlanCandidate):
        return (0 if c.payments[-1].date <= req.desired_completion_date else 1, 0 if not c.spending_changes else 1, c.total_payable, c.payments[0].date, len(c.payments), _option_order(c.payment_option_id))
    winner = sorted(safe_candidates, key=rank)[0]
    status = AffordabilityStatus.NOW if winner.method == PaymentMethod.FULL and winner.payments[0].date == req.request_date and not winner.spending_changes else AffordabilityStatus.LATER if winner.method == PaymentMethod.WAIT else AffordabilityStatus.WITH_PLAN
    return DecisionCore(req.request_id, safe_today, status, winner.method, winner.payments, earliest, winner.spending_changes, {"minimum_balance": str(ctx.profile.minimum_balance_to_keep), "winner_total": str(winner.total_payable)})
