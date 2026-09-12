from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from ..domain import EvidenceFact, EventStatus


class JsonEvidenceCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def get(self, key: str):
        rows = self.data.get(key)
        if rows is None:
            return None
        return tuple(EvidenceFact(
            evidence_id=r["evidence_id"], effect=r["effect"], related_event_id=r.get("related_event_id"),
            amount=Decimal(r["amount"]) if r.get("amount") is not None else None,
            currency=r.get("currency"), effective_date=date.fromisoformat(r["effective_date"]) if r.get("effective_date") else None,
            status=EventStatus(r["status"]) if r.get("status") else None, confidence=float(r.get("confidence", 1.0))
        ) for r in rows)

    def put(self, key: str, facts: tuple[EvidenceFact, ...]) -> None:
        self.data[key] = [{
            "evidence_id": f.evidence_id, "effect": f.effect, "related_event_id": f.related_event_id,
            "amount": str(f.amount) if f.amount is not None else None, "currency": f.currency,
            "effective_date": f.effective_date.isoformat() if f.effective_date else None,
            "status": f.status.value if f.status else None, "confidence": f.confidence,
        } for f in facts]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

