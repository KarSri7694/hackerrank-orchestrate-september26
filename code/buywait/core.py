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
from .recurrence import detect_recurrences, event_stream_key, recurring_event_ids

ZERO = Decimal("0")


def _forecast_streams(ctx: RequestContext):
    """Use protected-category reserves only for deterministic cash safety."""
    return detect_recurrences(
        ctx.events, ctx.request.request_date, include_variable_aggregates=True,
        protected_categories=ctx.profile.protected_categories,
    )


def _cash(event: FinancialEvent, home_currency: str, fx, on_date: date) -> Decimal:
    if event.amount is None:
        return ZERO
    value = fx.convert(event.amount, event.currency, home_currency, on_date)
    return value if event.direction == "credit" else -value


def _effective_events(ctx: RequestContext, fx, duplicate_suppressions: list[dict[str, str]] | None = None) -> list[tuple[date, Decimal, str, FinancialEvent]]:
    request_date = ctx.request.request_date
    end = request_date + timedelta(days=89)
    rows: list[tuple[date, Decimal, str, FinancialEvent]] = []
    historical_keys_by_pair: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    historical_events_by_pair: dict[tuple[str, str, str], list[FinancialEvent]] = defaultdict(list)
    for historical in ctx.events:
        if historical.status == EventStatus.SETTLED and historical.amount is not None:
            pair = (historical.category, historical.direction, historical.currency)
            historical_keys_by_pair[pair].add(event_stream_key(historical))
            historical_events_by_pair[pair].append(historical)

    def is_same_stream(candidate: FinancialEvent, representative: FinancialEvent) -> bool:
        if candidate.currency and representative.currency and candidate.currency != representative.currency:
            return False
        if event_stream_key(candidate) == event_stream_key(representative):
            return True
        # A bank may label a confirmed future occurrence generically (for
        # example, "next confirmed salary") while history carries the source
        # name. Treat that label as the one historical stream only when the
        # category/direction pair is unambiguous; never guess between two
        # simultaneous employers or services.
        candidate_source = event_stream_key(candidate).rsplit("|", 1)[-1]
        if candidate_source:
            return False
        pair = (candidate.category, candidate.direction, candidate.currency)
        keys = historical_keys_by_pair.get(pair, set())
        if len(keys) == 1:
            return event_stream_key(representative) in keys
        # With several streams, an exact amount/currency match may still make
        # a generic record unambiguous. Do not use approximate amounts: a
        # false match would silently drop income or double-count it.
        matching_keys = {
            event_stream_key(item) for item in historical_events_by_pair.get(pair, ())
            if item.amount == candidate.amount and item.currency == candidate.currency
        }
        return len(matching_keys) == 1 and event_stream_key(representative) in matching_keys
    for event in ctx.events:
        if event.event_type in {"internal_transfer", "aggregate_superseded"} or event.status in {EventStatus.FAILED, EventStatus.CANCELLED, EventStatus.UNREALIZED} or event.amount is None:
            continue
        if event.status == EventStatus.PENDING and event.direction == "credit":
            continue
        event_day = event.settlement_date or event.event_date
        if (event.status == EventStatus.PENDING and event.direction == "debit"
                and event_day < request_date):
            # A pending debit is already reserved even when its source row is
            # stale. Only a settlement date in the future gives us a more
            # precise date; otherwise reserve it immediately at the request
            # boundary.
            event_day = request_date
        if event_day >= request_date and event_day <= end:
            rows.append((event_day, _cash(event, ctx.profile.home_currency, fx, event_day), event.event_id, event))
    for stream in _forecast_streams(ctx):
        last = stream.representative
        next_day = stream.starts_on or last.settlement_date or last.event_date
        while next_day < request_date:
            next_day = stream.next_date(next_day)
        while next_day <= end and (stream.ends_before is None or next_day < stream.ends_before):
            # An explicit event on the same date wins over an inferred one;
            # descriptions can differ when a bank/payroll system changes its
            # wording, so category/direction/date are the stable identity.
            # Two simultaneous streams may share a category and direction
            # (for example two employers). Only an explicit event from this
            # stream suppresses its inferred occurrence.
            same_stream_exists = any(x[0] == next_day
                                     and ((stream.kind == "variable_category"
                                           and x[3].category == last.category
                                           and x[3].direction == last.direction
                                           and x[3].currency == last.currency)
                                          or is_same_stream(x[3], last)
                                          or last.event_type == "aggregate_stream_update")
                                     for x in rows)
            if not same_stream_exists:
                rows.append((next_day, _cash(last, ctx.profile.home_currency, fx, next_day), last.event_id, last))
            elif duplicate_suppressions is not None:
                explicit = next(x[3] for x in rows if x[0] == next_day
                                and ((stream.kind == "variable_category"
                                      and x[3].category == last.category
                                      and x[3].direction == last.direction
                                      and x[3].currency == last.currency)
                                     or is_same_stream(x[3], last)
                                     or last.event_type == "aggregate_stream_update"))
                duplicate_suppressions.append({"inferred_event_id": last.event_id,
                                               "explicit_event_id": explicit.event_id,
                                               "date": next_day.isoformat()})
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


def _validate_payments(ctx: RequestContext, payments: tuple[Payment, ...]) -> None:
    """Reject schedules that the bounded forecast would otherwise ignore."""
    end = ctx.request.request_date + timedelta(days=89)
    previous = None
    for payment in payments:
        if payment.amount <= ZERO:
            raise ValueError("payment amounts must be positive")
        if not ctx.request.request_date <= payment.date <= end:
            raise ValueError("payment date must be within the 90-day forecast")
        if previous is not None and payment.date < previous:
            raise ValueError("payment plan must be chronological")
        previous = payment.date


def simulate(ctx: RequestContext, fx, payments: tuple[Payment, ...] = (), changes: tuple[SpendingChange, ...] = ()) -> SimulationResult:
    ctx = reconcile_context(ctx)
    _validate_payments(ctx, payments)
    validate_changes(ctx, changes)
    changes_by_id = {c.event_id: c for c in changes}
    timeline = _effective_events(ctx, fx)
    balances = ctx.profile.current_available_balance
    # Confirmed cash events settle on their stated day and are available to a
    # payment on that date. This ordering also matches the dated data contract.
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


def _maximum_safe_payment(ctx: RequestContext, fx, on_date: date, ceiling: Decimal) -> Decimal:
    """Find the largest cent-precise payment accepted by the same simulator."""
    if ceiling <= ZERO or not simulate(ctx, fx).safe:
        return ZERO
    if simulate(ctx, fx, (Payment(on_date, ceiling),)).safe:
        return ceiling
    cents = Decimal("0.01")
    low, high = 0, int((ceiling / cents).to_integral_value(rounding=ROUND_DOWN))
    while low < high:
        middle = (low + high + 1) // 2
        amount = cents * middle
        if simulate(ctx, fx, (Payment(on_date, amount),)).safe:
            low = middle
        else:
            high = middle - 1
    return (cents * low).quantize(cents, rounding=ROUND_DOWN)


def baseline(ctx: RequestContext, fx) -> tuple[Decimal, date | None]:
    ctx = reconcile_context(ctx)
    # Do not derive the amount from a baseline minimum. The payment may occur
    # before same-day cash events, and the simulator is the single authority
    # for all payment methods and dates.
    safe_today = _maximum_safe_payment(ctx, fx, ctx.request.request_date, ctx.request.requested_amount)
    earliest = None
    for offset in range(90):
        day = ctx.request.request_date + timedelta(days=offset)
        result = simulate(ctx, fx, (Payment(day, ctx.request.requested_amount),))
        if result.safe:
            earliest = day
            break
    return safe_today, earliest


def explain_timeline(ctx: RequestContext, fx) -> dict[str, object]:
    """Return an auditable deterministic forecast without changing decisions.

    This is intentionally data-only so the regression harness and MCP/debug
    clients can expose the assumptions behind a safe amount.  It includes
    excluded source events and the post-event balance at every projected row.
    """
    ctx = reconcile_context(ctx)
    request_date = ctx.request.request_date
    end = request_date + timedelta(days=89)
    streams = _forecast_streams(ctx)
    suppressions: list[dict[str, str]] = []
    rows = _effective_events(ctx, fx, suppressions)
    explicit_ids = {
        event.event_id for event in ctx.events
        if event.amount is not None and event.status not in {EventStatus.FAILED, EventStatus.CANCELLED, EventStatus.UNREALIZED}
        and (event.settlement_date or event.event_date) >= request_date
    }
    balance = ctx.profile.current_available_balance
    minimum_balance = balance
    minimum_date = request_date
    projected = []
    for day, amount, event_id, event in sorted(rows, key=lambda item: (item[0], item[1], item[2])):
        if not request_date <= day <= end:
            continue
        balance += amount
        if balance < minimum_balance:
            minimum_balance = balance
            minimum_date = day
        inferred = event_id not in explicit_ids
        projected.append({
            "date": day.isoformat(), "event_id": event_id,
            "amount": str(amount), "category": event.category,
            "direction": event.direction, "currency": event.currency,
            "origin": "inferred_recurring" if inferred else "explicit",
            "balance_after": str(balance),
        })
    safe, earliest = baseline(ctx, fx)
    ignored = []
    for event in ctx.events:
        if event.amount is None:
            reason = "missing_amount"
        elif event.event_type == "internal_transfer":
            reason = "internal_transfer"
        elif event.status in {EventStatus.FAILED, EventStatus.CANCELLED, EventStatus.UNREALIZED}:
            reason = f"status_{event.status.value}"
        elif event.status == EventStatus.PENDING and event.direction == "credit":
            reason = "pending_credit"
        else:
            continue
        ignored.append({"event_id": event.event_id, "reason": reason})
    return {
        "opening_balance": str(ctx.profile.current_available_balance),
        "minimum_balance_to_keep": str(ctx.profile.minimum_balance_to_keep),
        "streams": [{"event_id": stream.representative.event_id, "kind": stream.kind,
                     "category": stream.representative.category, "direction": stream.representative.direction,
                     "currency": stream.representative.currency, "amount": str(stream.representative.amount),
                     "cadence_days": stream.cadence_days,
                     "covered_event_ids": sorted(stream.covered_event_ids)} for stream in streams],
        "ignored_events": ignored,
        "duplicate_suppressed_events": suppressions,
        "projected_events": projected,
        "minimum_projected_balance": str(minimum_balance),
        "minimum_projected_balance_date": minimum_date.isoformat(),
        "amount_safe_to_pay": str(safe),
        "earliest_date_for_full_payment": earliest.isoformat() if earliest else None,
    }


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


def candidate_decisions(ctx: RequestContext, fx) -> tuple[DecisionCore, ...]:
    """Return every deterministic, legal, safe decision in preference order.

    Models may select only from this list.  Keeping candidate construction in
    the finance core means no model response can invent arithmetic, dates, or
    a payment schedule.
    """
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
            forecast_end = req.request_date + timedelta(days=89)
            if (not payments or payments[0].date < req.request_date
                    or payments[-1].date > req.desired_completion_date
                    or payments[-1].date > forecast_end):
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
        return (DecisionCore(req.request_id, safe_today, AffordabilityStatus.NOT_AFFORDABLE,
                             PaymentMethod.NOT_RECOMMENDED, (), earliest, (),
                             {"minimum_balance": str(ctx.profile.minimum_balance_to_keep)}),)
    def rank(c: PlanCandidate):
        return (0 if c.payments[-1].date <= req.desired_completion_date else 1, 0 if not c.spending_changes else 1, c.total_payable, c.payments[0].date, len(c.payments), _option_order(c.payment_option_id))
    decisions = []
    for candidate in sorted(safe_candidates, key=rank):
        status = (AffordabilityStatus.NOW if candidate.method == PaymentMethod.FULL
                  and candidate.payments[0].date == req.request_date and not candidate.spending_changes
                  else AffordabilityStatus.LATER if candidate.method == PaymentMethod.WAIT
                  else AffordabilityStatus.WITH_PLAN)
        decisions.append(DecisionCore(
            req.request_id, safe_today, status, candidate.method, candidate.payments,
            earliest, candidate.spending_changes,
            {"minimum_balance": str(ctx.profile.minimum_balance_to_keep),
             "winner_total": str(candidate.total_payable),
             "payment_option_id": candidate.payment_option_id or ""},
        ))
    return tuple(decisions)


def solve(ctx: RequestContext, fx) -> DecisionCore:
    """Return the deterministic first-choice candidate for non-AI operation."""
    return candidate_decisions(ctx, fx)[0]
