from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

from .adapters.csv_repository import CsvRepository
from .adapters.evidence_cache import JsonEvidenceCache
from .adapters.exchange_rates import CsvExchangeRates
from .core import _optimized_changes, baseline, simulate, solve, validate_changes
from .domain import CorrectionEvaluation, EvidenceFact, EventStatus, Payment, SpendingChange
from .evidence import EvidenceService
from .reconciliation import _fact_stream_key, _unique_transfer_pair, reconcile_context


class Application:
    """Inbound application services used by CLI and MCP adapters."""

    def __init__(self, dataset_dir: str | Path, evidence_extractor=None, evidence_cache_path: str | Path | None = None):
        self.repository = CsvRepository(dataset_dir)
        self.fx = CsvExchangeRates(Path(dataset_dir) / "exchange_rates.csv")
        cache_path = Path(evidence_cache_path) if evidence_cache_path else Path(dataset_dir) / ".evidence_cache.json"
        self.evidence_service = EvidenceService(self.repository, evidence_extractor, JsonEvidenceCache(cache_path))

    def _context(self, request_id: str):
        raw = self.repository.context(request_id)
        facts = self.evidence_service.facts_for(raw.evidence_refs)
        return self._reconciled(raw, facts)

    @staticmethod
    def _reconciled(raw, facts):
        return reconcile_context(replace(raw, evidence_facts=tuple(facts)))

    def context_with_evidence(self, request_id: str, facts: tuple[EvidenceFact, ...]):
        self.evidence_service.register(facts)
        return self._context(request_id)

    def decision_with_evidence(self, request_id: str, facts: tuple[EvidenceFact, ...]):
        """Run the complete deterministic decision flow on resolved events."""
        return solve(self.context_with_evidence(request_id, facts), self.fx)

    def evaluate_correction(self, request_id: str, facts: tuple[EvidenceFact, ...],
                            supporting_evidence_ids: tuple[str, ...]) -> CorrectionEvaluation:
        """Evaluate a proposed fact change without allowing a plan override.

        A correction is admissible only when it is explicitly tied to evidence
        attached to this request and can be reconstructed and solved by the
        same deterministic path used in production.
        """
        raw = self.repository.context(request_id)
        baseline_decision = self.decision(request_id)
        reference_ids = {ref.evidence_id for ref in raw.evidence_refs}
        support = set(supporting_evidence_ids)
        if not facts or not support or not support.issubset(reference_ids):
            return CorrectionEvaluation(False, "missing or unrelated supporting evidence", baseline_decision)
        event_ids = {event.event_id for event in raw.events}
        allowed_effects = {"cancel", "cancelled", "cancellation", "terminate", "terminated",
                           "amend", "amendment", "amount_amendment", "date_amendment",
                           "delay", "delayed", "settle", "settled", "image_amount", "stream_update", "aggregate_stream_update",
                           "internal_transfer", "one_time"}
        for fact in facts:
            if fact.evidence_id not in support or fact.evidence_id not in reference_ids:
                return CorrectionEvaluation(False, "fact is not tied to supplied evidence", baseline_decision)
            if fact.effect not in allowed_effects:
                return CorrectionEvaluation(False, "unsupported correction effect", baseline_decision)
            if fact.amount is not None and fact.amount < 0:
                return CorrectionEvaluation(False, "negative correction amount", baseline_decision)
            if fact.related_event_id and fact.related_event_id not in event_ids:
                return CorrectionEvaluation(False, "correction references an unknown event", baseline_decision)
            if fact.effect == "internal_transfer" and not fact.related_event_id:
                if _unique_transfer_pair(list(raw.events), fact) is None:
                    return CorrectionEvaluation(False, "internal transfer has no unambiguous debit/credit pair", baseline_decision)
            elif not fact.related_event_id and not (fact.category and fact.direction and fact.effective_date):
                return CorrectionEvaluation(False, "stream correction lacks category, direction, or effective date", baseline_decision)
            if fact.effect == "aggregate_stream_update" and not (fact.category and fact.direction and fact.amount is not None and fact.currency and fact.effective_date and fact.recurring is True and fact.recurrence_days):
                return CorrectionEvaluation(False, "aggregate stream update lacks required recurring total", baseline_decision)
            if fact.effect in {"terminate", "terminated"} and not fact.related_event_id and not _fact_stream_key(fact):
                return CorrectionEvaluation(False, "stream termination lacks a stable stream identity", baseline_decision)
            if fact.direction and fact.direction not in {"credit", "debit"}:
                return CorrectionEvaluation(False, "invalid stream direction", baseline_decision)
            if fact.status and not isinstance(fact.status, EventStatus):
                return CorrectionEvaluation(False, "invalid event status", baseline_decision)
            if not self._fact_is_supported(raw, fact):
                return CorrectionEvaluation(False, "correction cannot be verified from supplied evidence", baseline_decision)
        base_facts = self.evidence_service.facts_for(raw.evidence_refs)
        candidate_context = self._reconciled(raw, base_facts + tuple(facts))
        candidate = solve(candidate_context, self.fx)
        if candidate.payment_plan and not simulate(candidate_context, self.fx, candidate.payment_plan, candidate.spending_changes).safe:
            return CorrectionEvaluation(False, "correction produced an unsafe plan", baseline_decision)
        if candidate == baseline_decision:
            return CorrectionEvaluation(False, "correction does not change the deterministic result", baseline_decision)
        return CorrectionEvaluation(True, "verified evidence correction", candidate, tuple(facts))

    def apply_correction(self, facts: tuple[EvidenceFact, ...]) -> None:
        """Persist only an already evaluated correction for subsequent rounds."""
        self.evidence_service.add_facts(facts)

    def _fact_is_supported(self, raw, fact: EvidenceFact) -> bool:
        """Require an extractor-confirmed fact or explicit textual support.

        This deliberately favors rejection: an evidence identifier by itself is
        not proof that a model's proposed cash-flow change appears in that
        message or image.
        """
        reference = next(ref for ref in raw.evidence_refs if ref.evidence_id == fact.evidence_id)
        extracted = self.evidence_service.inspect(fact.evidence_id)
        fields = ("effect", "related_event_id", "amount", "currency", "effective_date", "status",
                  "category", "direction", "recurring", "recurrence_days")
        for known in extracted:
            exact = all(getattr(known, field) == getattr(fact, field)
                        for field in fields if getattr(fact, field) is not None)
            same_stream = not fact.stream_source or _fact_stream_key(known) == _fact_stream_key(fact)
            if exact and same_stream:
                # effect is mandatory, so this branch never treats an omitted
                # extractor field as proof for an arbitrary critic value.
                return True
        # Image evidence has no deterministic textual representation. It must
        # be supported by the VLM's structured extraction above.
        if reference.source_type == "image":
            return False
        text = (reference.text or "").casefold()
        if not text:
            return False
        # The constrained critic is the semantic interpreter for effect,
        # category, direction, status, and recurrence.  Raw text is only used
        # for deterministic checks of literal source identifiers and numeric
        # claims; this works for multilingual text without English keywords.
        scalar_fields = {
            "related_event_id": None if fact.related_event_id == reference.related_event_id else fact.related_event_id,
            "amount": str(fact.amount) if fact.amount is not None else None,
            "currency": fact.currency, "effective_date": fact.effective_date.isoformat() if fact.effective_date else None,
            "stream_source": fact.stream_source,
        }
        if any(str(value).casefold() not in text for value in scalar_fields.values() if value is not None):
            return False
        if fact.effect == "internal_transfer" and not fact.related_event_id:
            return True
        if not any(value is not None for value in scalar_fields.values()) and not reference.related_event_id:
            return False
        return True

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
