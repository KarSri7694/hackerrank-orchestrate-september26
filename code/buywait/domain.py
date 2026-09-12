from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum


class AffordabilityStatus(str, Enum):
    NOW = "affordable_now"
    WITH_PLAN = "affordable_with_plan"
    LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"


class PaymentMethod(str, Enum):
    FULL = "full_payment"
    PARTIAL = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"


class EventStatus(str, Enum):
    SETTLED = "settled"
    SCHEDULED = "scheduled"
    PENDING = "pending"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNREALIZED = "unrealized"


@dataclass(frozen=True)
class FinancialRequest:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    protected_categories: frozenset[str]
    reducible_categories: frozenset[str]
    stoppable_categories: frozenset[str]
    accepted_payment_methods: frozenset[PaymentMethod]
    max_installment_months: int | None


@dataclass(frozen=True)
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Decimal | None
    currency: str
    event_date: date
    settlement_date: date | None
    status: EventStatus
    linked_event_id: str | None
    flexibility: str
    minimum_allowed_amount: Decimal | None


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: PaymentMethod
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal


@dataclass(frozen=True)
class Payment:
    date: date
    amount: Decimal


@dataclass(frozen=True)
class SpendingChange:
    event_id: str
    action: str
    new_amount: Decimal | None = None


@dataclass(frozen=True)
class EvidenceReference:
    evidence_id: str
    source_type: str
    user_id: str
    request_id: str | None = None
    related_event_id: str | None = None
    text: str | None = None
    image_path: str | None = None


@dataclass(frozen=True)
class EvidenceFact:
    evidence_id: str
    effect: str
    related_event_id: str | None = None
    amount: Decimal | None = None
    currency: str | None = None
    effective_date: date | None = None
    status: EventStatus | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class PlanCandidate:
    method: PaymentMethod
    payments: tuple[Payment, ...]
    total_payable: Decimal
    spending_changes: tuple[SpendingChange, ...] = ()
    payment_option_id: str | None = None


@dataclass(frozen=True)
class SimulationResult:
    safe: bool
    minimum_projected_balance: Decimal
    required_minimum_balance: Decimal
    first_violation_date: date | None
    shortfall: Decimal
    ending_balance: Decimal


@dataclass(frozen=True)
class DecisionCore:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: AffordabilityStatus
    recommended_payment_method: PaymentMethod
    payment_plan: tuple[Payment, ...]
    earliest_date_for_full_payment: date | None
    spending_changes: tuple[SpendingChange, ...]
    trace: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestContext:
    request: FinancialRequest
    profile: FinancialProfile
    events: tuple[FinancialEvent, ...]
    payment_options: tuple[PaymentOption, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    evidence_facts: tuple[EvidenceFact, ...] = ()
