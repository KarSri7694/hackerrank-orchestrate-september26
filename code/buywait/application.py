from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from .adapters.csv_repository import CsvRepository
from .adapters.exchange_rates import CsvExchangeRates
from .core import baseline, simulate, solve
from .domain import EvidenceFact, Payment, SpendingChange
from .evidence import EvidenceService


class Application:
    """Inbound application services used by CLI and MCP adapters."""

    def __init__(self, dataset_dir: str | Path):
        self.repository = CsvRepository(dataset_dir)
        self.fx = CsvExchangeRates(Path(dataset_dir) / "exchange_rates.csv")
        self.evidence_service = EvidenceService(self.repository)

    def _context(self, request_id: str):
        return self.repository.context(request_id)

    def decision(self, request_id: str):
        return solve(self._context(request_id), self.fx)

    def case(self, request_id: str) -> dict[str, object]:
        ctx = self._context(request_id)
        safe, earliest = baseline(ctx, self.fx)
        return {
            "request_id": ctx.request.request_id,
            "user_id": ctx.request.user_id,
            "request_date": ctx.request.request_date.isoformat(),
            "request_type": ctx.request.request_type,
            "requested_amount": str(ctx.request.requested_amount),
            "desired_completion_date": ctx.request.desired_completion_date.isoformat(),
            "allows_partial_payment": ctx.request.allows_partial_payment,
            "home_currency": ctx.profile.home_currency,
            "current_available_balance": str(ctx.profile.current_available_balance),
            "minimum_balance_to_keep": str(ctx.profile.minimum_balance_to_keep),
            "accepted_payment_methods": sorted(x.value for x in ctx.profile.accepted_payment_methods),
            "max_installment_months": ctx.profile.max_installment_months,
            "amount_safe_to_pay": str(safe),
            "earliest_date_for_full_payment": earliest.isoformat() if earliest else None,
            "payment_options": [{
                "payment_option_id": o.payment_option_id,
                "payment_method": o.payment_method.value,
                "payment_amount": str(o.payment_amount),
                "number_of_payments": o.number_of_payments,
                "first_payment_date": o.first_payment_date.isoformat(),
                "payment_frequency_days": o.payment_frequency_days,
                "financing_fee": str(o.financing_fee),
                "total_payable_amount": str(o.total_payable_amount),
            } for o in ctx.payment_options],
            "evidence_refs": [r.__dict__ for r in ctx.evidence_refs],
        }

    def simulate_plan(self, request_id: str, payments: tuple[Payment, ...], changes: tuple[SpendingChange, ...]):
        ctx = self._context(request_id)
        self._validate_changes(ctx, changes)
        return simulate(ctx, self.fx, payments, changes)

    def optimize_spending(self, request_id: str, payments: tuple[Payment, ...]) -> dict[str, object]:
        from .core import legal_changes
        ctx = self._context(request_id)
        for changes in legal_changes(ctx, list(ctx.events)):
            result = simulate(ctx, self.fx, payments, changes)
            if result.safe:
                return {"possible": True, "spending_changes": [c.__dict__ for c in changes], "minimum_projected_balance": str(result.minimum_projected_balance)}
        return {"possible": False, "spending_changes": [], "minimum_projected_balance": str(simulate(ctx, self.fx, payments).minimum_projected_balance)}

    def _validate_changes(self, ctx, changes: tuple[SpendingChange, ...]):
        if len(changes) > 3 or len({c.event_id for c in changes}) != len(changes):
            raise ValueError("at most three spending changes are allowed and event IDs must be unique")
        events = {e.event_id: e for e in ctx.events}
        for change in changes:
            event = events.get(change.event_id)
            if event is None or event.flexibility == "fixed" or event.category in ctx.profile.protected_categories:
                raise ValueError(f"event is not a legal flexible spending target: {change.event_id}")
