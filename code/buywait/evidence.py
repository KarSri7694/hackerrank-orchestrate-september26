from __future__ import annotations

from dataclasses import replace

from .domain import EvidenceFact


class NullEvidenceExtractor:
    """Safe default: evidence is not interpreted without a configured provider."""
    def extract(self, reference):
        return ()


class EvidenceService:
    def __init__(self, repository, extractor=None):
        self.repository = repository
        self.extractor = extractor or NullEvidenceExtractor()
        self._facts: dict[str, tuple[EvidenceFact, ...]] = {}

    def inspect(self, evidence_id: str) -> tuple[EvidenceFact, ...]:
        if evidence_id not in self._facts:
            ref = self.repository.evidence(evidence_id)
            self._facts[evidence_id] = tuple(
                replace(fact, source_type=fact.source_type or ref.source_type, sent_at=fact.sent_at or ref.sent_at)
                for fact in self.extractor.extract(ref)
            )
        return self._facts[evidence_id]

    def register(self, facts: tuple[EvidenceFact, ...] | list[EvidenceFact]) -> None:
        grouped: dict[str, list[EvidenceFact]] = {}
        for fact in facts:
            grouped.setdefault(fact.evidence_id, []).append(fact)
        for evidence_id, values in grouped.items():
            ref = self.repository.evidence(evidence_id)
            self._facts[evidence_id] = tuple(
                replace(fact, source_type=fact.source_type or ref.source_type, sent_at=fact.sent_at or ref.sent_at)
                for fact in values
            )

    def add_facts(self, facts: tuple[EvidenceFact, ...] | list[EvidenceFact]) -> None:
        """Append accepted corrections without discarding extracted evidence."""
        grouped: dict[str, list[EvidenceFact]] = {}
        for fact in facts:
            grouped.setdefault(fact.evidence_id, []).append(fact)
        for evidence_id, values in grouped.items():
            ref = self.repository.evidence(evidence_id)
            normalized = tuple(
                replace(fact, source_type=fact.source_type or ref.source_type, sent_at=fact.sent_at or ref.sent_at)
                for fact in values
            )
            existing = self.inspect(evidence_id)
            combined = existing + tuple(fact for fact in normalized if fact not in existing)
            self._facts[evidence_id] = combined

    def facts_for(self, references, inspect: bool = True) -> tuple[EvidenceFact, ...]:
        facts: list[EvidenceFact] = []
        for ref in references:
            if inspect or ref.evidence_id in self._facts:
                facts.extend(self.inspect(ref.evidence_id))
        return tuple(facts)
