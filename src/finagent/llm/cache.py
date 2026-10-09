"""On-disk cache so repeated runs never pay twice for the same LLM call."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def make_key(**parts: Any) -> str:
    """Stable SHA-256 key for a call. ``bytes`` values are hashed, not embedded."""

    def _default(obj: Any) -> Any:
        if isinstance(obj, bytes):
            return {"__bytes_sha256__": hashlib.sha256(obj).hexdigest()}
        raise TypeError(f"Cannot hash value of type {type(obj).__name__}")

    payload = json.dumps(parts, sort_keys=True, default=_default, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    """A tiny JSON-file cache sharded by key prefix. Writes are atomic."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, key: str) -> Path:
        return self._root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["value"]
        except (json.JSONDecodeError, KeyError):
            return None  # corrupted entry: treat as a miss and let it be overwritten

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"value": value}, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
