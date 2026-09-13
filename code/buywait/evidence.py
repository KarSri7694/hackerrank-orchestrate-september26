from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

from .adapters.evidence_cache import JsonEvidenceCache
from .domain import EvidenceFact
from .evidence_validation import text_supports_amount, text_supports_date


class NullEvidenceExtractor:
    """Safe default: evidence is not interpreted without a configured provider."""
    def extract(self, reference):
        return ()


class EvidenceService:
    # Bump when extraction instructions/normalization change so malformed or
    # incomplete facts from an older local-model run are not reused.
    CACHE_SCHEMA_VERSION = "evidence-facts-v22"

    def __init__(self, repository, extractor=None, persistent_cache: JsonEvidenceCache | None = None,
                 stream_context_provider=None):
        self.repository = repository
        self.extractor = extractor or NullEvidenceExtractor()
        self._facts: dict[str, tuple[EvidenceFact, ...]] = {}
        self.persistent_cache = persistent_cache
        # Deterministic/no-AI mode must not consume facts produced by a prior
        # model run, including entries whose key used NullEvidenceExtractor.
        # Otherwise --no-ai would silently mutate the financial state from
        # stale external interpretation and usage accounting would be false.
        self._cache_enabled = not isinstance(self.extractor, NullEvidenceExtractor)
        self.stream_context_provider = stream_context_provider
        self.cache_hits = 0
        self.extraction_calls = 0

    def _cache_key(self, reference) -> str:
        content = (reference.text or "").encode("utf-8")
        if reference.image_path and Path(reference.image_path).is_file():
            content += Path(reference.image_path).read_bytes()
        model = getattr(self.extractor, "model", self.extractor.__class__.__name__)
        return f"{reference.evidence_id}:{model}:{self.CACHE_SCHEMA_VERSION}:{hashlib.sha256(content).hexdigest()}"

    def _cache_empty_results(self) -> bool:
        return bool(getattr(self.extractor, "cache_empty_results", True))

    def _legacy_cached(self, key: str):
        """Find a same-content fact from an older schema cache entry.

        Schema bumps should invalidate malformed entries, but retaining a
        non-empty same-model/same-content result avoids throwing away a valid
        expensive VLM extraction. The application validation boundary still
        decides whether the legacy fact is admissible.
        """
        if not self.persistent_cache:
            return None
        prefix, content_hash = key.rsplit(":", 1)
        current_schema = self.CACHE_SCHEMA_VERSION
        candidates = []
        for candidate_key in self.persistent_cache.data:
            if not candidate_key.startswith(prefix.rsplit(":", 1)[0] + ":evidence-facts-v"):
                continue
            if not candidate_key.endswith(":" + content_hash) or candidate_key.endswith(":" + current_schema + ":" + content_hash):
                continue
            marker = candidate_key.split(":evidence-facts-v", 1)[-1].split(":", 1)[0]
            try:
                candidates.append((int(marker), candidate_key))
            except ValueError:
                continue
        for _, candidate_key in sorted(candidates, reverse=True):
            facts = self.persistent_cache.get(candidate_key)
            if facts:
                return facts
        return None

    @staticmethod
    def _normalized(facts, reference, *, validate_cached_dates: bool = False) -> tuple[EvidenceFact, ...]:
        normalized = []
        for fact in facts:
            # Cache entries are untrusted just like fresh model output. Keep
            # dates only when textual evidence explicitly contains the same
            # numeric date; VLM/image facts remain governed by structured
            # image extraction and are not text-filtered.
            if (validate_cached_dates and reference.source_type != "image" and reference.text
                    and not text_supports_date(reference.text, fact.effective_date)):
                continue
            if (validate_cached_dates and reference.source_type != "image" and reference.text
                    and not text_supports_amount(reference.text, fact.amount, fact.currency)):
                continue
            # Source metadata is authoritative in messages.csv/images.csv;
            # never let model output alter the source bucket or timestamp used
            # by deterministic precedence rules.
            value = replace(fact, source_type=reference.source_type,
                            sent_at=reference.sent_at)
            # Keep the persisted representation canonical even when it came
            # from a pre-normalization cache entry. Linked blank image
            # amounts use the dedicated effect in the evidence contract.
            if (reference.source_type == "image" and reference.related_event_id
                    and value.amount is not None):
                value = replace(value, effect="image_amount",
                                related_event_id=value.related_event_id or reference.related_event_id)
            normalized.append(value)
        return tuple(normalized)

    def inspect(self, evidence_id: str) -> tuple[EvidenceFact, ...]:
        if evidence_id in self._facts:
            if self._cache_enabled:
                self.cache_hits += 1
            return self._facts[evidence_id]
        if not self._cache_enabled:
            self._facts[evidence_id] = ()
            return ()
        ref = self.repository.evidence(evidence_id)
        key = self._cache_key(ref)
        cached = self.persistent_cache.get(key) if self.persistent_cache else None
        if cached is not None:
            persisted = cached
            cached = self._normalized(persisted, ref, validate_cached_dates=True)
            # A non-empty persisted result that normalizes to empty was
            # invalid evidence (for example a stale unsupported date/amount),
            # not a legitimate negative extraction. Re-run the extractor so
            # the invalid cache entry cannot suppress a corrected fact.
            if persisted and not cached:
                cached = None
        if cached is None:
            cached = self._legacy_cached(key)
            if cached:
                cached = self._normalized(cached, ref, validate_cached_dates=True)
                if self.persistent_cache:
                    if cached:
                        self.persistent_cache.put(key, cached)
                    else:
                        cached = None
        if cached == () and not self._cache_empty_results():
            cached = None
        if cached is None:
            context = self.stream_context_provider(ref) if self.stream_context_provider else ()
            parameters = inspect.signature(self.extractor.extract).parameters
            if "stream_context" in parameters:
                produced = self.extractor.extract(ref, stream_context=context)
            else:
                produced = self.extractor.extract(ref)
            self.extraction_calls += 1
            cached = self._normalized(produced, ref)
            if self.persistent_cache and (cached or self._cache_empty_results()):
                self.persistent_cache.put(key, cached)
        else:
            self.cache_hits += 1
        self._facts[evidence_id] = self._normalized(cached, ref)
        return self._facts[evidence_id]

    def register(self, facts: tuple[EvidenceFact, ...] | list[EvidenceFact]) -> None:
        grouped: dict[str, list[EvidenceFact]] = {}
        for fact in facts:
            grouped.setdefault(fact.evidence_id, []).append(fact)
        for evidence_id, values in grouped.items():
            ref = self.repository.evidence(evidence_id)
            self._facts[evidence_id] = tuple(
                replace(fact, source_type=ref.source_type, sent_at=ref.sent_at)
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
                replace(fact, source_type=ref.source_type, sent_at=ref.sent_at)
                for fact in values
            )
            existing = self.inspect(evidence_id)
            combined = existing + tuple(fact for fact in normalized if fact not in existing)
            self._facts[evidence_id] = combined

    def facts_for(self, references, inspect: bool = True, request_id: str | None = None) -> tuple[EvidenceFact, ...]:
        facts: list[EvidenceFact] = []
        setter = getattr(self.extractor, "set_trace_request", None)
        if callable(setter):
            setter(request_id)
        for ref in references:
            if inspect or ref.evidence_id in self._facts:
                facts.extend(self.inspect(ref.evidence_id))
        return tuple(facts)
