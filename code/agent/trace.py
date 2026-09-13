from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from threading import Lock


_SECRET_NAMES = {"api_key", "authorization", "cookie", "token", "password", "secret"}


def _safe(value, key: str | None = None):
    """Convert SDK objects to inspectable JSON without credentials or image blobs."""
    if key and any(part in key.casefold() for part in _SECRET_NAMES):
        return "[REDACTED]"
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and value.startswith("data:image/"):
            return "[IMAGE DATA URL REDACTED]"
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return "[BINARY DATA REDACTED]"
    if isinstance(value, dict):
        return {str(k): _safe(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    for method in ("model_dump", "to_dict"):
        converter = getattr(value, method, None)
        if callable(converter):
            try:
                return _safe(converter())
            except Exception:
                pass
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return _safe(attributes)
    return str(value)


class ModelTurnRecorder:
    """Append one durable JSONL record for every provider call.

    The file is flushed after each call so a long 25-sample run remains
    inspectable even if a later request times out or is interrupted.
    """

    def __init__(self, path: str | Path, run_id: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._sequence = 0
        self._lock = Lock()
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")

    def record(self, *, kind: str, model: str, request, response, context: dict | None = None) -> None:
        with self._lock:
            self._sequence += 1
            row = {
                "sequence": self._sequence,
                "run_id": self.run_id,
                "recorded_at": datetime.now().astimezone().isoformat(),
                "kind": kind,
                "model": model,
                "context": _safe(context or {}),
                "request": _safe(request),
                "response": _safe(response),
            }
            self._handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
