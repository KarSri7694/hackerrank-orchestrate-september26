from __future__ import annotations

import base64
from pathlib import Path


def image_data_url(path: str | Path, allowed_root: str | Path, max_bytes: int = 8_000_000) -> str:
    root = Path(allowed_root).resolve()
    candidate = Path(path).resolve()
    if root not in candidate.parents:
        raise ValueError("image path is outside the dataset image directory")
    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        raise ValueError("unsupported image type")
    payload = candidate.read_bytes()
    if len(payload) > max_bytes:
        raise ValueError("image exceeds configured size limit")
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}[candidate.suffix.lower()]
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"

