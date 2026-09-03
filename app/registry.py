from __future__ import annotations

import json
from datetime import datetime, timezone
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
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def asset_type(payload: dict) -> str:
        kind = payload.get("asset_type")
        if kind in {"video", "images"}:
            return kind
        if payload.get("image_paths"):
            return "images"
        return "video"

    def upsert(self, asset_id: str, payload: dict) -> None:
        with self._lock:
            data = self._read()
            item = dict(payload)
            item["asset_type"] = self.asset_type(item)
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
            if not item:
                return None
            item = dict(item)
            item["asset_type"] = self.asset_type(item)
            return item

    def list(self) -> list[dict]:
        with self._lock:
            data = self._read()
        rows = []
        for asset_id, raw in data.items():
            payload = dict(raw)
            payload["asset_type"] = self.asset_type(payload)
            rows.append({"asset_id": asset_id, **payload})
        rows.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
        return rows

    @staticmethod
    def public_view(asset_id: str, payload: dict) -> dict:
        kind = AssetRegistry.asset_type(payload)
        image_paths = [Path(p) for p in (payload.get("image_paths") or [])]
        if kind == "video":
            video_path = Path(payload.get("video_path") or "")
            media_available = bool(payload.get("video_path")) and video_path.is_file()
            image_count = 0
        else:
            media_available = any(p.is_file() for p in image_paths)
            image_count = int(payload.get("image_count") or len(image_paths))

        return {
            "asset_id": asset_id,
            "name": payload.get("name") or asset_id,
            "asset_type": kind,
            "updated_at": payload.get("updated_at"),
            "duration_sec": float(payload.get("duration_sec") or 0),
            "video_chunks": int(payload.get("video_chunks") or 0),
            "transcript_chunks": int(payload.get("transcript_chunks") or 0),
            "image_count": image_count,
            "weaviate_objects": int(payload.get("weaviate_objects") or 0),
            "media_available": media_available,
        }

    def list_public(self) -> list[dict]:
        return [
            self.public_view(row["asset_id"], row)
            for row in self.list()
        ]

    def get_public(self, asset_id: str) -> dict | None:
        payload = self.get(asset_id)
        if not payload:
            return None
        return self.public_view(asset_id, payload)


registry = AssetRegistry(settings.registry_path)
