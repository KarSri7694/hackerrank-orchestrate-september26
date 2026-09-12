from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Protocol

from .domain import (DecisionCore, EvidenceFact, EvidenceReference, FinancialEvent,
                     Payment, RequestContext, SimulationResult, SpendingChange)


class Repository(Protocol):
    def context(self, request_id: str) -> RequestContext: ...
    def event(self, event_id: str) -> FinancialEvent: ...
    def evidence(self, evidence_id: str) -> EvidenceReference: ...


class ExchangeRateProvider(Protocol):
    def convert(self, amount: Decimal, source: str, target: str, on_date: date) -> Decimal: ...


class EvidenceExtractor(Protocol):
    def extract(self, reference: EvidenceReference) -> tuple[EvidenceFact, ...]: ...


class EvidenceCache(Protocol):
    def get(self, key: str) -> tuple[EvidenceFact, ...] | None: ...
    def put(self, key: str, facts: tuple[EvidenceFact, ...]) -> None: ...


class UsageRecorder(Protocol):
    def record(self, **values: object) -> None: ...


class CaseService(Protocol):
    def get_case(self, request_id: str) -> dict[str, object]: ...


class EvidenceService(Protocol):
    def inspect(self, evidence_id: str) -> tuple[EvidenceFact, ...]: ...


class SimulationService(Protocol):
    def simulate(self, request_id: str, payments: tuple[Payment, ...], changes: tuple[SpendingChange, ...]) -> SimulationResult: ...

