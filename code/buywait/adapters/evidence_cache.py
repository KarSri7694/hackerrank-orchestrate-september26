from __future__ import annotations

import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from ..domain import EvidenceFact, EventStatus


class JsonEvidenceCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            self.data = {}
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            loaded = {}
        self.data = loaded if isinstance(loaded, dict) else {}

    def get(self, key: str):
        rows = self.data.get(key)
        if rows is None:
            return None
        if not isinstance(rows, list):
            return None
        try:
            return tuple(EvidenceFact(
                evidence_id=r["evidence_id"], effect=r["effect"], related_event_id=r.get("related_event_id"),
                amount=Decimal(r["amount"]) if r.get("amount") is not None else None,
                currency=r.get("currency"), effective_date=date.fromisoformat(r["effective_date"]) if r.get("effective_date") else None,
                status=EventStatus(r["status"]) if r.get("status") else None, confidence=float(r.get("confidence", 1.0)),
                source_type=r.get("source_type"), sent_at=datetime.fromisoformat(r["sent_at"]) if r.get("sent_at") else None,
                category=r.get("category"), direction=r.get("direction"), description=r.get("description"),
                recurring=r.get("recurring"), recurrence_days=r.get("recurrence_days"),
                flexibility=r.get("flexibility", "fixed"),
                minimum_allowed_amount=Decimal(r["minimum_allowed_amount"]) if r.get("minimum_allowed_amount") is not None else None,
                stream_source=r.get("stream_source"),
                stream_key=r.get("stream_key"),
            ) for r in rows)
        except (KeyError, TypeError, ValueError, ArithmeticError):
            return None

    def put(self, key: str, facts: tuple[EvidenceFact, ...]) -> None:
        self.data[key] = [{
            "evidence_id": f.evidence_id, "effect": f.effect, "related_event_id": f.related_event_id,
            "amount": str(f.amount) if f.amount is not None else None, "currency": f.currency,
            "effective_date": f.effective_date.isoformat() if f.effective_date else None,
            "status": f.status.value if f.status else None, "confidence": f.confidence,
            "source_type": f.source_type, "sent_at": f.sent_at.isoformat() if f.sent_at else None,
            "category": f.category, "direction": f.direction, "description": f.description,
            "recurring": f.recurring, "recurrence_days": f.recurrence_days,
            "stream_key": f.stream_key,
            "stream_source": f.stream_source,
            "flexibility": f.flexibility,
            "minimum_allowed_amount": str(f.minimum_allowed_amount) if f.minimum_allowed_amount is not None else None,
        } for f in facts]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
