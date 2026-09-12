from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

from .adapters.csv_repository import CsvRepository
from .adapters.exchange_rates import CsvExchangeRates
from .core import _optimized_changes, baseline, simulate, solve, validate_changes
from .domain import EvidenceFact, Payment, SpendingChange
from .evidence import EvidenceService
from .reconciliation import reconcile_context


class Application:
    """Inbound application services used by CLI and MCP adapters."""

    def __init__(self, dataset_dir: str | Path, evidence_extractor=None):
        self.repository = CsvRepository(dataset_dir)
        self.fx = CsvExchangeRates(Path(dataset_dir) / "exchange_rates.csv")
        self.evidence_service = EvidenceService(self.repository, evidence_extractor)

    def _context(self, request_id: str):
        raw = self.repository.context(request_id)
        facts = self.evidence_service.facts_for(raw.evidence_refs)
        return reconcile_context(replace(raw, evidence_facts=facts))

    def context_with_evidence(self, request_id: str, facts: tuple[EvidenceFact, ...]):
        self.evidence_service.register(facts)
        return self._context(request_id)

    def decision_with_evidence(self, request_id: str, facts: tuple[EvidenceFact, ...]):
        """Run the complete deterministic decision flow on resolved events."""
        return solve(self.context_with_evidence(request_id, facts), self.fx)

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
            "request_text": ctx.request.request_text,
            "requested_amount": str(ctx.request.requested_amount),
            "desired_completion_date": ctx.request.desired_completion_date.isoformat(),
            "allows_partial_payment": ctx.request.allows_partial_payment,
            "home_currency": ctx.profile.home_currency,
            "current_available_balance": str(ctx.profile.current_available_balance),
            "minimum_balance_to_keep": str(ctx.profile.minimum_balance_to_keep),
            "financial_priorities": list(ctx.profile.financial_priorities),
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
            "evidence_refs": [{
                "evidence_id": r.evidence_id, "source_type": r.source_type,
                "request_id": r.request_id, "related_event_id": r.related_event_id,
                "sent_at": r.sent_at.isoformat() if r.sent_at else None,
                "has_text": bool(r.text), "image_available": bool(r.image_path),
            } for r in ctx.evidence_refs],
        }

    def simulate_plan(self, request_id: str, payments: tuple[Payment, ...], changes: tuple[SpendingChange, ...] = ()):
        ctx = self._context(request_id)
        self._validate_changes(ctx, changes)
        return simulate(ctx, self.fx, payments, changes)

    def optimize_spending(self, request_id: str, payments: tuple[Payment, ...]) -> dict[str, object]:
        ctx = self._context(request_id)
        changes = _optimized_changes(ctx, self.fx, payments)
        result = simulate(ctx, self.fx, payments, tuple(changes or ()))
        return {"possible": bool(changes is not None and result.safe), "spending_changes": [c.__dict__ for c in changes or ()], "minimum_projected_balance": str(result.minimum_projected_balance)}

    def _validate_changes(self, ctx, changes: tuple[SpendingChange, ...]):
        validate_changes(ctx, changes)
