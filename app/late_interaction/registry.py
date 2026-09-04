from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from app.config import settings


class LateInteractionRegistry:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    def upsert(self, asset_id: str, payload: dict) -> None:
        with self._lock:
            data = self._read()
            item = dict(payload)
            item["updated_at"] = item.get("updated_at") or datetime.now(timezone.utc).isoformat()
            data[asset_id] = item
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def remove(self, asset_id: str) -> bool:
        with self._lock:
            data = self._read()
            existed = asset_id in data
            if existed:
                data.pop(asset_id, None)
                self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return existed

    def get(self, asset_id: str) -> dict | None:
        with self._lock:
            item = self._read().get(asset_id)
        return dict(item) if item else None

    def list(self) -> list[dict]:
        with self._lock:
            data = self._read()
        rows = [{"asset_id": asset_id, **dict(payload)} for asset_id, payload in data.items()]
        rows.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
        return rows

    @staticmethod
    def public_view(asset_id: str, payload: dict) -> dict:
        kind = str(payload.get("asset_type") or "")
        image_paths = [Path(p) for p in (payload.get("image_paths") or [])]
        text_paths = [Path(p) for p in (payload.get("text_paths") or [])]
        media_available = (
            any(path.is_file() for path in image_paths)
            if kind == "images"
            else any(path.is_file() for path in text_paths)
        )
        return {
            "asset_id": asset_id,
            "name": payload.get("name") or asset_id,
            "asset_type": kind,
            "updated_at": payload.get("updated_at"),
            "image_count": int(payload.get("image_count") or len(image_paths)),
            "text_file_count": int(payload.get("text_file_count") or len(text_paths)),
            "text_chunks": int(payload.get("text_chunks") or 0),
            "weaviate_objects": int(payload.get("weaviate_objects") or 0),
            "late_vectors": int(payload.get("late_vectors") or 0),
            "average_late_vectors": float(payload.get("average_late_vectors") or 0.0),
            "dense_dim": 2048,
            "late_dim": 128,
            "embedding_model": "jinaai/jina-embeddings-v4",
            "media_available": media_available,
            "preview_files": [p.name for p in text_paths[:24]],
        }

    def list_public(self) -> list[dict]:
        return [self.public_view(row["asset_id"], row) for row in self.list()]

    def get_public(self, asset_id: str) -> dict | None:
        payload = self.get(asset_id)
        return self.public_view(asset_id, payload) if payload else None


registry = LateInteractionRegistry(settings.li_registry_path)
