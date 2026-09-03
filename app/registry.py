from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from .config import settings


class AssetRegistry:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def upsert(self, asset_id: str, payload: dict) -> None:
        with self._lock:
            data = self._read()
            data[asset_id] = payload
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get(self, asset_id: str) -> dict | None:
        with self._lock:
            return self._read().get(asset_id)

    def list(self) -> list[dict]:
        with self._lock:
            data = self._read()
            return [{"asset_id": k, **v} for k, v in data.items()]


registry = AssetRegistry(settings.registry_path)
