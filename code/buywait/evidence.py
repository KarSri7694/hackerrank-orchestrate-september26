from __future__ import annotations

from .domain import EvidenceFact


class NullEvidenceExtractor:
    """Safe default: evidence is not interpreted without a configured provider."""
    def extract(self, reference):
        return ()


class EvidenceService:
    def __init__(self, repository, extractor=None):
        self.repository = repository
        self.extractor = extractor or NullEvidenceExtractor()

    def inspect(self, evidence_id: str) -> tuple[EvidenceFact, ...]:
        return tuple(self.extractor.extract(self.repository.evidence(evidence_id)))

