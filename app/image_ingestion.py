from __future__ import annotations

import hashlib
import traceback
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from .config import settings
from .ingestion import INGEST_LOCK, MODEL_NAME, _set
from .jina_mlx import embedder
from .job_store import jobs
from .registry import registry
from .weaviate_store import store


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tif", ".tiff", ".gif", ".avif",
}


def discover_image_paths(source: str | Path) -> list[Path]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Image file/folder not found: {path}")
    images = sorted(
        (
            p.resolve()
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda p: str(p).lower(),
    )
    if not images:
        raise ValueError(f"No supported images found under: {path}")
    return images


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_asset_id(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: (p.name.lower(), str(p).lower())):
        digest.update(path.name.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as raw:
        return ImageOps.exif_transpose(raw).convert("RGB")


def _run_image_ingestion(job_id: str, image_paths: list[str], asset_name: str | None = None):
    try:
        paths = [Path(p).expanduser().resolve() for p in image_paths]
        if not paths:
            raise ValueError("No image files supplied")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image extension: {path.suffix}")

        _set(job_id, 3, "Validate images", f"Checking {len(paths)} image files")
        asset_id = _image_asset_id(paths)
        jobs.update(job_id, asset_id=asset_id)

        _set(job_id, 7, "Connect Weaviate", "Checking local Weaviate and schema")
        store.ensure_collection()

        _set(job_id, 11, "Load Jina MLX", "Loading the local Jina checkpoint")
        embedder.load()

        _set(job_id, 15, "Image preflight", "Testing processor + MLX vision tower")
        embedder.image_preflight()

        store.delete_asset(asset_id)

        asset_dir = settings.assets_dir / asset_id
        thumbs_dir = asset_dir / "thumbs"
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        total = len(paths)
        _set(
            job_id, 18, "Index images",
            f"Embedding {total} images one by one",
            image_total=total, image_done=0, weaviate_objects=0,
        )

        done = 0
        for index, path in enumerate(paths):
            image = _open_rgb(path)
            vector = embedder.embed_image(image)

            thumb_name = f"{index:05d}.jpg"
            thumb_path = thumbs_dir / thumb_name
            thumb = image.copy()
            thumb.thumbnail((720, 720))
            thumb.save(thumb_path, format="JPEG", quality=86, optimize=True)

            chunk_id = f"image_{index:05d}"
            store.insert(
                {
                    "asset_id": asset_id,
                    "chunk_id": chunk_id,
                    "modality": "image",
                    "chunk_index": index,
                    "start_sec": 0.0,
                    "end_sec": 0.0,
                    "duration_sec": 0.0,
                    "text": "",
                    "speaker_ids": [],
                    "token_count": 0,
                    "frame_count": 1,
                    "source_name": path.name,
                    "thumbnail_relpath": f"thumbs/{thumb_name}",
                    "embedding_model": MODEL_NAME,
                    "embedding_dim": 1024,
                },
                vector,
            )
            done += 1
            pct = 18 + 78 * (done / max(1, total))
            _set(
                job_id, pct, "Embed/index images",
                f"Image {done}/{total} indexed · {path.name}",
                image_done=done,
                weaviate_objects=done,
            )

        total_objects = store.count_asset(asset_id)
        default_name = paths[0].parent.name if len(paths) > 1 else paths[0].stem
        registry.upsert(
            asset_id,
            {
                "name": asset_name or default_name or f"image-set-{asset_id[:8]}",
                "asset_type": "images",
                "image_paths": [str(path) for path in paths],
                "image_count": len(paths),
                "duration_sec": 0.0,
                "video_chunks": 0,
                "transcript_chunks": 0,
                "weaviate_objects": total_objects,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        jobs.update(
            job_id,
            status="completed",
            stage="Complete",
            overall_pct=100.0,
            detail=f"Indexed {total_objects} images for asset {asset_id}",
            counters={
                "image_total": total,
                "image_done": total,
                "weaviate_objects": total_objects,
            },
        )
    except Exception as exc:
        jobs.update(
            job_id,
            status="failed",
            stage="Failed",
            error=f"{type(exc).__name__}: {exc}",
            detail=traceback.format_exc(),
        )


def run_image_ingestion(job_id: str, image_paths: list[str], asset_name: str | None = None):
    if not INGEST_LOCK.acquire(blocking=False):
        jobs.update(
            job_id,
            status="queued",
            stage="Waiting for ingestion slot",
            detail="Another local MLX ingestion job is active; this image job will start next.",
        )
        INGEST_LOCK.acquire()
    try:
        return _run_image_ingestion(job_id, image_paths, asset_name)
    finally:
        INGEST_LOCK.release()
