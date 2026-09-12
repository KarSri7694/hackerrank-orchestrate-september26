from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from .adapters.evidence_cache import JsonEvidenceCache
from .domain import EvidenceFact


class NullEvidenceExtractor:
    """Safe default: evidence is not interpreted without a configured provider."""
    def extract(self, reference):
        return ()


class EvidenceService:
    CACHE_SCHEMA_VERSION = "evidence-facts-v3"

    def __init__(self, repository, extractor=None, persistent_cache: JsonEvidenceCache | None = None):
        self.repository = repository
        self.extractor = extractor or NullEvidenceExtractor()
        self._facts: dict[str, tuple[EvidenceFact, ...]] = {}
        self.persistent_cache = persistent_cache

    def _cache_key(self, reference) -> str:
        content = (reference.text or "").encode("utf-8")
        if reference.image_path and Path(reference.image_path).is_file():
            content += Path(reference.image_path).read_bytes()
        model = getattr(self.extractor, "model", self.extractor.__class__.__name__)
        return f"{reference.evidence_id}:{model}:{self.CACHE_SCHEMA_VERSION}:{hashlib.sha256(content).hexdigest()}"

    @staticmethod
    def _normalized(facts, reference) -> tuple[EvidenceFact, ...]:
        return tuple(replace(fact, source_type=fact.source_type or reference.source_type,
                             sent_at=fact.sent_at or reference.sent_at) for fact in facts)

    def inspect(self, evidence_id: str) -> tuple[EvidenceFact, ...]:
        if evidence_id not in self._facts:
            ref = self.repository.evidence(evidence_id)
            key = self._cache_key(ref)
            cached = self.persistent_cache.get(key) if self.persistent_cache else None
            if cached is None:
                cached = self._normalized(self.extractor.extract(ref), ref)
                if self.persistent_cache:
                    self.persistent_cache.put(key, cached)
            self._facts[evidence_id] = self._normalized(cached, ref)
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
