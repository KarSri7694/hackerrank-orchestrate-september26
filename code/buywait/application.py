from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import re

from .adapters.csv_repository import CsvRepository
from .adapters.evidence_cache import JsonEvidenceCache
from .adapters.exchange_rates import CsvExchangeRates
from .core import (_optimized_changes, baseline, candidate_decisions, expand_option,
                   explain_timeline, simulate, solve, validate_changes)
from .domain import CorrectionEvaluation, EvidenceFact, EventStatus, Payment, SpendingChange
from .evidence import EvidenceService
from .presentation import decision_explanation
from .reconciliation import _fact_stream_key, _unique_transfer_pair, reconcile_context
from .recurrence import detect_recurrences, event_stream_key


class Application:
    """Inbound application services used by CLI and MCP adapters."""

    def __init__(self, dataset_dir: str | Path, evidence_extractor=None, evidence_cache_path: str | Path | None = None,
                 allow_legacy_evidence_cache: bool = True):
        self.repository = CsvRepository(dataset_dir)
        self.fx = CsvExchangeRates(Path(dataset_dir) / "exchange_rates.csv")
        cache_path = Path(evidence_cache_path) if evidence_cache_path else Path(dataset_dir) / ".evidence_cache.json"
        self.evidence_service = EvidenceService(
            self.repository, evidence_extractor, JsonEvidenceCache(cache_path),
            stream_context_provider=self._stream_context_for_evidence,
            allow_legacy_cache=allow_legacy_evidence_cache,
        )

    def _stream_context_for_evidence(self, reference):
        """Give the extractor a small ledger view rather than the whole CSV."""
        events = self.repository.events_by_user.get(reference.user_id, ())
        related = self.repository.events.get(reference.related_event_id or "")
        selected = []
        if related:
            selected.append(related)
            selected.extend(event for event in events if event.event_id != related.event_id
                            and event_stream_key(event) == event_stream_key(related))
        else:
            words = {word.casefold() for word in (reference.text or "").replace("/", " ").split() if len(word) > 2}
            selected = [event for event in events if words & {word.casefold() for word in event.description.replace("/", " ").split()}]
            if not selected:
                selected = sorted(events, key=lambda event: event.settlement_date or event.event_date, reverse=True)[:12]
        selected = sorted(selected, key=lambda event: event.settlement_date or event.event_date, reverse=True)[:12]
        return tuple(self._event_payload(event) for event in selected)

    def _context(self, request_id: str):
        raw = self.repository.context(request_id)
        facts = self._validated_facts(raw, self.evidence_service.facts_for(raw.evidence_refs, request_id=request_id))
        return self._reconciled(raw, facts)

    def _validated_facts(self, raw, facts):
        """Reject malformed extractor facts before reconciliation can use them.

        This is deliberately deterministic: model confidence is not authority.
        Critic corrections receive the stricter evidence-support checks in
        ``evaluate_correction`` as well.
        """
        refs = {ref.evidence_id: ref for ref in raw.evidence_refs}
        events = {event.event_id: event for event in raw.events}
        currencies = {raw.profile.home_currency} | {event.currency for event in raw.events if event.currency}
        # A stream explicitly introduced by evidence may have no historical
        # event in its currency yet. Admit only currencies represented by the
        # fixed offline exchange-rate table; simulation will still enforce
        # that a usable dated conversion exists.
        rates = getattr(getattr(self, "fx", None), "rates", {})
        for rate_key in rates:
            if isinstance(rate_key, tuple) and len(rate_key) >= 3:
                currencies.update(rate_key[1:3])
        effects = {"cancel", "amend", "amount_amendment", "date_amendment", "delay", "settle",
                   "image_amount", "stream_update", "aggregate_stream_update", "terminate",
                   "internal_transfer", "one_time"}
        accepted = []
        for fact in facts:
            ref = refs.get(fact.evidence_id)
            # A missing image is not evidence.  Do not let a stale cache entry
            # or a model response produced without the image mutate the ledger.
            if (ref is not None and ref.source_type == "image"
                    and (not ref.image_path or not Path(ref.image_path).is_file())):
                continue
            # Older persistent extraction caches can contain a model
            # placeholder amount. Re-apply this safety boundary when loading
            # a legacy cache, before any fact can reach reconciliation.
            if (ref is not None and ref.text and fact.amount == Decimal("0")
                    and not re.search(r"(?<![A-Za-z0-9])0(?![A-Za-z0-9])", ref.text)):
                continue
            if ref is None or fact.effect not in effects:
                continue
            if (fact.amount is not None
                    and (not fact.amount.is_finite() or fact.amount < 0)):
                continue
            if (fact.minimum_allowed_amount is not None
                    and (not fact.minimum_allowed_amount.is_finite() or fact.minimum_allowed_amount < 0)):
                continue
            if fact.currency and fact.currency not in currencies:
                continue
            if fact.direction and fact.direction not in {"credit", "debit"}:
                continue
            event = events.get(fact.related_event_id or "")
            if fact.related_event_id and event is None:
                continue
            if (ref.source_type == "image" and event is not None and event.amount is None
                    and fact.amount is not None and fact.currency):
                fact = replace(fact, effect="image_amount")
            if event is not None:
                if fact.direction and fact.direction != event.direction:
                    continue
                if fact.category and fact.category != event.category:
                    continue
                if fact.effect == "image_amount" and (ref.source_type != "image" or ref.related_event_id != event.event_id or event.amount is not None):
                    continue
            if fact.effect == "internal_transfer" and not fact.related_event_id and _unique_transfer_pair(list(raw.events), fact) is None:
                continue
            stream_level = fact.related_event_id is None and fact.effect != "internal_transfer"
            pending_stream_status = fact.effect == "stream_update" and fact.status == EventStatus.PENDING
            if stream_level and not (fact.category and fact.direction
                                     and (fact.effective_date or pending_stream_status)):
                continue
            if fact.recurrence_days is not None and fact.recurrence_days <= 0:
                continue
            if fact.recurring is False and fact.recurrence_days is not None:
                continue
            if fact.effect == "aggregate_stream_update" and not (fact.amount is not None and fact.currency and fact.recurring is True and fact.recurrence_days):
                continue
            if fact.effect == "stream_update" and fact.recurring is True and not fact.recurrence_days:
                continue
            if fact.effect == "terminate" and stream_level and not _fact_stream_key(fact):
                continue
            accepted.append(fact)
        return tuple(accepted)

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
            if (fact.amount is not None
                    and (not fact.amount.is_finite() or fact.amount < 0)):
                return CorrectionEvaluation(False, "negative correction amount", baseline_decision)
            if (fact.minimum_allowed_amount is not None
                    and (not fact.minimum_allowed_amount.is_finite() or fact.minimum_allowed_amount < 0)):
                return CorrectionEvaluation(False, "invalid minimum correction amount", baseline_decision)
            if fact.recurrence_days is not None and fact.recurrence_days <= 0:
                return CorrectionEvaluation(False, "invalid recurrence cadence", baseline_decision)
            if fact.recurring is False and fact.recurrence_days is not None:
                return CorrectionEvaluation(False, "one-time correction cannot have recurrence cadence", baseline_decision)
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
        base_facts = self._validated_facts(raw, self.evidence_service.facts_for(raw.evidence_refs))
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

    def candidate_decisions(self, request_id: str):
        """Stable, code-generated choices exposed to the two model stages."""
        return candidate_decisions(self._context(request_id), self.fx)

    def candidate_options(self, request_id: str) -> list[dict[str, object]]:
        ctx = self._context(request_id)
        options = []
        for index, decision in enumerate(candidate_decisions(ctx, self.fx), 1):
            simulation = simulate(ctx, self.fx, decision.payment_plan, decision.spending_changes)
            payment_option_id = decision.trace.get("payment_option_id") or None
            provider_option = next((item for item in ctx.payment_options
                                    if item.payment_option_id == payment_option_id), None)
            options.append({
                "candidate_id": f"candidate_{index:03d}",
                "affordability_status": decision.affordability_status.value,
                "recommended_payment_method": decision.recommended_payment_method.value,
                "amount_safe_to_pay": str(decision.amount_safe_to_pay),
                "earliest_date_for_full_payment": (decision.earliest_date_for_full_payment.isoformat()
                                                    if decision.earliest_date_for_full_payment else None),
                "payment_plan": [{"date": payment.date.isoformat(), "amount": str(payment.amount)}
                                 for payment in decision.payment_plan],
                "spending_changes": [{"event_id": change.event_id, "action": change.action,
                                      "new_amount": str(change.new_amount) if change.new_amount is not None else None}
                                     for change in decision.spending_changes],
                "payment_option_id": payment_option_id,
                "total_payable": decision.trace.get("winner_total"),
                "financing_fee": str(provider_option.financing_fee) if provider_option else "0",
                "safe": simulation.safe,
                "minimum_projected_balance": str(simulation.minimum_projected_balance),
            })
        return options

    def payment_options_summary(self, request_id: str) -> dict[str, object]:
        """Return code-calculated payment terms for model comparison.

        This accepts no model-supplied amount, duration, or schedule. Every
        returned option is already legal, generated from provider data, and
        simulator-safe.
        """
        options = self.candidate_options(request_id)
        case = self.case(request_id)
        summary: dict[str, object] = {
            "amount_safe_to_pay": case["amount_safe_to_pay"],
            "earliest_date_for_full_payment": case["earliest_date_for_full_payment"],
            "full_payment": None,
            "partial_payment": None,
            "wait": None,
            "installments": [],
        }
        for option in options:
            method = option["recommended_payment_method"]
            if method == "installments":
                payments = option["payment_plan"]
                summary["installments"].append({
                    "candidate_id": option["candidate_id"],
                    "payment_option_id": option["payment_option_id"],
                    "payment_amount": payments[0]["amount"] if payments else None,
                    "number_of_payments": len(payments),
                    "months": len(payments),
                    "first_payment_date": payments[0]["date"] if payments else None,
                    "last_payment_date": payments[-1]["date"] if payments else None,
                    "payment_plan": payments,
                    "total_payable": option["total_payable"],
                    "financing_fee": option["financing_fee"],
                    "minimum_projected_balance": option["minimum_projected_balance"],
                })
            elif method == "full_payment":
                summary["full_payment"] = option
            elif method == "partial_payment":
                summary["partial_payment"] = option
            elif method == "wait":
                summary["wait"] = option
        return summary

    def decision_for_candidate(self, request_id: str, candidate_id: str):
        if not isinstance(candidate_id, str):
            return None
        try:
            index = int(candidate_id.removeprefix("candidate_")) - 1
        except ValueError:
            return None
        decisions = self.candidate_decisions(request_id)
        return decisions[index] if 0 <= index < len(decisions) and candidate_id == f"candidate_{index + 1:03d}" else None

    def inspect_source(self, request_id: str, source_type: str, source_id: str) -> dict[str, object]:
        """Read-only direct-tool implementation for source records."""
        raw = self.repository.context(request_id)
        if source_type == "event":
            event = next((item for item in raw.events if item.event_id == source_id), None)
            return {"found": bool(event), "source_type": "event",
                    "record": self._event_payload(event) if event else None}
        reference = next((item for item in raw.evidence_refs if item.evidence_id == source_id), None)
        return {"found": bool(reference), "source_type": "evidence", "record": {
            "evidence_id": reference.evidence_id, "source_type": reference.source_type,
            "related_event_id": reference.related_event_id,
            "sent_at": reference.sent_at.isoformat() if reference.sent_at else None,
            "text": reference.text, "image_available": bool(reference.image_path),
        } if reference else None}

    def agent_case(self, request_id: str) -> dict[str, object]:
        """Complete read-only context for decision and critic model calls."""
        payload = self.case(request_id)
        raw = self.repository.context(request_id)
        ctx = self._context(request_id)
        payload["raw_financial_events"] = [self._event_payload(event) for event in raw.events]
        payload["resolved_financial_events"] = [self._event_payload(event) for event in ctx.events]
        payload["evidence"] = [{
            "evidence_id": ref.evidence_id, "source_type": ref.source_type,
            "related_event_id": ref.related_event_id,
            "sent_at": ref.sent_at.isoformat() if ref.sent_at else None,
            "text": ref.text, "image_available": bool(ref.image_path),
        } for ref in raw.evidence_refs]
        payload["forecast"] = explain_timeline(ctx, self.fx)
        payload["candidates"] = self.candidate_options(request_id)
        return payload

    def explain_timeline(self, request_id: str) -> dict[str, object]:
        """Expose the same reconciled deterministic timeline used by solve."""
        return explain_timeline(self._context(request_id), self.fx)

    def finalized_decision(self, request_id: str, decision=None):
        """Attach the submission explanation from the same reconciled state used to solve."""
        ctx = self._context(request_id)
        decision = decision or solve(ctx, self.fx)
        simulation = simulate(ctx, self.fx, decision.payment_plan, decision.spending_changes)
        financing_fee = None
        if decision.recommended_payment_method.value == "installments":
            for option in ctx.payment_options:
                if tuple((p.date, p.amount) for p in expand_option(option)) == tuple((p.date, p.amount) for p in decision.payment_plan):
                    financing_fee = option.financing_fee
                    break
        explanation = decision_explanation(
            decision, currency=ctx.profile.home_currency,
            minimum_balance=ctx.profile.minimum_balance_to_keep,
            projected_minimum=simulation.minimum_projected_balance if decision.payment_plan else None,
            financing_fee=financing_fee,
        )
        return replace(decision, decision_explanation=explanation)

    def case(self, request_id: str) -> dict[str, object]:
        raw = self.repository.context(request_id)
        facts = self._validated_facts(raw, self.evidence_service.facts_for(raw.evidence_refs, request_id=request_id))
        ctx = self._reconciled(raw, facts)
        safe, earliest = baseline(ctx, self.fx)
        selected = solve(ctx, self.fx)
        baseline_result = simulate(ctx, self.fx)
        selected_result = simulate(ctx, self.fx, selected.payment_plan, selected.spending_changes)
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
            "raw_financial_events": [self._event_payload(event) for event in self._compact_events(raw.events, raw.request.request_date)],
            "resolved_financial_events": [self._event_payload(event) for event in self._compact_events(ctx.events, ctx.request.request_date)],
            "resolved_recurring_streams": [self._stream_payload(stream) for stream in detect_recurrences(ctx.events)],
            "extracted_evidence": [{
                "evidence_id": ref.evidence_id, "source_type": ref.source_type,
                "related_event_id": ref.related_event_id,
                "sent_at": ref.sent_at.isoformat() if ref.sent_at else None,
                "facts": [self._fact_payload(fact) for fact in facts if fact.evidence_id == ref.evidence_id],
            } for ref in ctx.evidence_refs],
            "forecast_diagnostics": {
                "amount_safe_to_pay": str(safe),
                "earliest_date_for_full_payment": earliest.isoformat() if earliest else None,
                "required_minimum_balance": str(ctx.profile.minimum_balance_to_keep),
                "baseline_minimum_projected_balance": str(baseline_result.minimum_projected_balance),
                "recommended_plan_minimum_projected_balance": str(selected_result.minimum_projected_balance),
                "recommended_plan_ending_balance": str(selected_result.ending_balance),
                "first_violation_date": selected_result.first_violation_date.isoformat() if selected_result.first_violation_date else None,
            },
            "evidence_refs": [{
                "evidence_id": r.evidence_id, "source_type": r.source_type,
                "request_id": r.request_id, "related_event_id": r.related_event_id,
                "sent_at": r.sent_at.isoformat() if r.sent_at else None,
                "has_text": bool(r.text), "image_available": bool(r.image_path),
            } for r in ctx.evidence_refs],
        }

    @staticmethod
    def _event_payload(event):
        return {
            "event_id": event.event_id, "event_type": event.event_type,
            "description": event.description, "category": event.category,
            "direction": event.direction, "amount": str(event.amount) if event.amount is not None else None,
            "currency": event.currency, "event_date": event.event_date.isoformat(),
            "settlement_date": event.settlement_date.isoformat() if event.settlement_date else None,
            "status": event.status.value, "linked_event_id": event.linked_event_id,
            "flexibility": event.flexibility,
            "minimum_allowed_amount": str(event.minimum_allowed_amount) if event.minimum_allowed_amount is not None else None,
            "recurrence_days": event.recurrence_days,
        }

    @staticmethod
    def _fact_payload(fact):
        return {
            "effect": fact.effect, "related_event_id": fact.related_event_id,
            "amount": str(fact.amount) if fact.amount is not None else None,
            "currency": fact.currency,
            "effective_date": fact.effective_date.isoformat() if fact.effective_date else None,
            "status": fact.status.value if fact.status else None,
            "category": fact.category, "direction": fact.direction,
            "description": fact.description, "recurring": fact.recurring,
            "recurrence_days": fact.recurrence_days, "stream_source": fact.stream_source,
        }

    @staticmethod
    def _stream_payload(stream):
        event = stream.representative
        return {
            "description": event.description,
            "category": event.category,
            "direction": event.direction,
            "amount": str(event.amount) if event.amount is not None else None,
            "currency": event.currency,
            "cadence_days": stream.cadence_days,
            "starts_on": stream.starts_on.isoformat() if stream.starts_on else None,
            "ends_before": stream.ends_before.isoformat() if stream.ends_before else None,
        }

    @staticmethod
    def _compact_events(events, request_date: date):
        """Keep forecast rows plus enough per-stream history to audit recurrence."""
        lower, upper = request_date - timedelta(days=120), request_date + timedelta(days=89)
        selected = {event.event_id for event in events if request_date <= (event.settlement_date or event.event_date) <= upper}
        history_by_stream: dict[tuple[str, str, str, str], list] = {}
        for event in sorted(events, key=lambda item: item.settlement_date or item.event_date, reverse=True):
            day = event.settlement_date or event.event_date
            if not lower <= day < request_date:
                continue
            key = (event.description, event.category, event.direction, event.currency)
            bucket = history_by_stream.setdefault(key, [])
            if len(bucket) < 3:
                bucket.append(event)
                selected.add(event.event_id)
        changed = True
        while changed:
            changed = False
            for event in events:
                if event.event_id in selected or event.linked_event_id in selected:
                    if event.event_id not in selected:
                        selected.add(event.event_id)
                        changed = True
                    if event.linked_event_id and event.linked_event_id not in selected:
                        selected.add(event.linked_event_id)
                        changed = True
        return tuple(event for event in events if event.event_id in selected)

    def simulate_plan(self, request_id: str, payments: tuple[Payment, ...], changes: tuple[SpendingChange, ...] = ()):
        ctx = self._context(request_id)
        self._validate_changes(ctx, changes)
        return simulate(ctx, self.fx, payments, changes)

    def optimize_spending(self, request_id: str, payments: tuple[Payment, ...]) -> dict[str, object]:
        ctx = self._context(request_id)
        changes = _optimized_changes(ctx, self.fx, payments)
        result = simulate(ctx, self.fx, payments, tuple(changes or ()))
        return {"possible": bool(changes is not None and result.safe), "spending_changes": [
            {"event_id": change.event_id, "action": change.action,
             "new_amount": str(change.new_amount) if change.new_amount is not None else None}
            for change in changes or ()
        ], "minimum_projected_balance": str(result.minimum_projected_balance)}

    def _validate_changes(self, ctx, changes: tuple[SpendingChange, ...]):
        validate_changes(ctx, changes)
