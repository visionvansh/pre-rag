from __future__ import annotations

import hashlib
import traceback
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from app.config import settings
from app.job_store import jobs
from .jina_v4_client import embedder
from .registry import registry
from .text_ingestion import LI_INGEST_LOCK
from .weaviate_store import store


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"}


def discover_image_paths(source: str | Path) -> list[Path]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Image file/folder not found: {path}")
    files = sorted(
        (p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    if not files:
        raise ValueError(f"No supported images found under: {path}")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_id(paths: list[Path]) -> str:
    digest = hashlib.sha256(b"jina-v4-late-image-v1\0")
    for path in sorted(paths, key=lambda p: (p.name.lower(), str(p).lower())):
        digest.update(path.name.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def _set(job_id: str, pct: float, stage: str, detail: str, **counters) -> None:
    jobs.update(job_id, status="running", stage=stage, overall_pct=round(float(pct), 2), detail=detail, counters=counters)


def _run(job_id: str, image_paths: list[str], asset_name: str | None) -> None:
    try:
        paths = [Path(value).expanduser().resolve() for value in image_paths]
        if not paths:
            raise ValueError("No image files supplied")
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported or missing image: {path}")

        asset_id = _asset_id(paths)
        jobs.update(job_id, asset_id=asset_id)
        _set(job_id, 3, "Validate images", f"Validated {len(paths)} images")
        store.ensure_collection()
        _set(job_id, 8, "Jina v4 worker", "Starting the isolated Jina v4 worker")
        embedder.status()

        store.delete_asset(asset_id)
        asset_dir = settings.li_assets_dir / asset_id
        thumbs_dir = asset_dir / "thumbs"
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        late_total = 0
        total = len(paths)
        for index, path in enumerate(paths):
            # Decode locally before expensive model work so corrupt files fail clearly.
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                thumb = image.copy()
            encoded = embedder.encode_image(path)
            late_count = int(encoded["late_vector_count"])
            late_total += late_count

            thumb.thumbnail((720, 720))
            thumb_name = f"{index:05d}.jpg"
            thumb.save(thumbs_dir / thumb_name, format="JPEG", quality=86, optimize=True)
            store.insert(
                {
                    "asset_id": asset_id,
                    "chunk_id": f"image_{index:05d}",
                    "modality": "image",
                    "chunk_index": index,
                    "text": "",
                    "token_count": 0,
                    "source_name": path.name,
                    "thumbnail_relpath": f"thumbs/{thumb_name}",
                    "late_vector_count": late_count,
                    "embedding_model": "jinaai/jina-embeddings-v4",
                    "model_revision": str(settings.jina_v4_revision),
                    "dense_dim": 2048,
                    "late_dim": 128,
                },
                encoded["dense"],
                encoded["multi"],
            )
            done = index + 1
            _set(
                job_id,
                10 + 86 * (done / total),
                "Embed/index images",
                f"Image {done}/{total} indexed · {late_total:,} late vectors",
                image_total=total,
                image_done=done,
                late_vectors=late_total,
                weaviate_objects=done,
            )

        object_count = store.count_asset(asset_id)
        if object_count != total:
            raise RuntimeError(f"Verification failed: expected {total} objects, found {object_count}")
        default_name = paths[0].parent.name if len(paths) > 1 else paths[0].stem
        registry.upsert(
            asset_id,
            {
                "name": asset_name or default_name or f"li-images-{asset_id[:8]}",
                "asset_type": "images",
                "image_paths": [str(path) for path in paths],
                "image_count": total,
                "text_file_count": 0,
                "text_chunks": 0,
                "weaviate_objects": object_count,
                "late_vectors": late_total,
                "average_late_vectors": late_total / max(1, total),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        jobs.update(
            job_id,
            status="completed",
            stage="Complete",
            overall_pct=100.0,
            detail=f"Indexed {total} images with {late_total:,} late vectors",
            counters={"image_total": total, "image_done": total, "late_vectors": late_total, "weaviate_objects": object_count},
        )
    except Exception as exc:
        jobs.update(job_id, status="failed", stage="Failed", error=f"{type(exc).__name__}: {exc}", detail=traceback.format_exc())


def run_image_ingestion(job_id: str, image_paths: list[str], asset_name: str | None = None) -> None:
    if not LI_INGEST_LOCK.acquire(blocking=False):
        jobs.update(job_id, status="queued", stage="Waiting for LI ingestion slot", detail="Another Jina-v4 LI ingestion is active")
        LI_INGEST_LOCK.acquire()
    try:
        _run(job_id, image_paths, asset_name)
    finally:
        LI_INGEST_LOCK.release()
