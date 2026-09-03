from __future__ import annotations

import asyncio
import shutil
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from .config import settings
from .image_ingestion import discover_image_paths, run_image_ingestion
from .ingestion import run_ingestion
from .jina_mlx import embedder
from .job_store import jobs
from .registry import registry
from .schemas import ImagePathIngestRequest, PathIngestRequest, SearchRequest
from .weaviate_store import store


app = FastAPI(title="Jina v5 Omni Multimodal RAG Lab")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    weaviate_ready = False
    weaviate_error = None
    try:
        weaviate_ready = bool(store.connect().is_ready())
    except Exception as exc:
        weaviate_error = str(exc)
    return {
        "ok": True,
        "model_path": str(settings.jina_model_path),
        "model_exists": settings.jina_model_path.exists(),
        "weaviate_ready": weaviate_ready,
        "weaviate_error": weaviate_error,
        "video_chunk_seconds": settings.video_chunk_seconds,
        "video_chunk_seconds_min": 1.0,
        "video_chunk_seconds_max": 120.0,
        "video_max_frames": settings.video_max_frames,
        "transcript_chunk_tokens": settings.transcript_chunk_tokens,
        "transcript_overlap_tokens": settings.transcript_overlap_tokens,
        "supported_modalities": ["transcript", "video", "image"],
    }


def _public_asset(asset_id: str) -> dict:
    public = registry.get_public(asset_id)
    if not public:
        raise HTTPException(404, "Asset not found")
    if public["asset_type"] == "video":
        public["media_url"] = f"/api/assets/{asset_id}/media" if public["media_available"] else None
        public["preview_urls"] = []
    else:
        count = public["image_count"]
        public["media_url"] = None
        public["preview_urls"] = [f"/api/assets/{asset_id}/thumb/{i}" for i in range(min(count, 24))]
    return public


@app.get("/api/assets")
def assets():
    rows = []
    for item in registry.list_public():
        try:
            rows.append(_public_asset(item["asset_id"]))
        except HTTPException:
            continue
    return rows


@app.get("/api/assets/{asset_id}")
def asset_detail(asset_id: str):
    return _public_asset(asset_id)


@app.delete("/api/assets/{asset_id}")
def remove_asset(asset_id: str):
    if not registry.get(asset_id):
        raise HTTPException(404, "Asset not found")
    store.delete_asset(asset_id)
    registry.remove(asset_id)
    generated_dir = settings.assets_dir / asset_id
    if generated_dir.exists():
        shutil.rmtree(generated_dir, ignore_errors=True)
    return {"ok": True, "asset_id": asset_id}


@app.post("/api/ingest/path")
async def ingest_path(request: PathIngestRequest):
    job = jobs.create()
    asyncio.create_task(
        asyncio.to_thread(
            run_ingestion,
            job.id,
            request.video_path,
            request.transcript_path,
            request.asset_name,
            request.video_chunk_seconds,
        )
    )
    return {"job_id": job.id}


async def _save_upload(upload: UploadFile, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out:
        while True:
            chunk = await upload.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    await upload.close()


@app.post("/api/ingest/upload")
async def ingest_upload(
    video: UploadFile = File(...),
    transcript: UploadFile = File(...),
    asset_name: str | None = Form(default=None),
    video_chunk_seconds: float | None = Form(default=None),
):
    if video_chunk_seconds is not None and not 1.0 <= video_chunk_seconds <= 120.0:
        raise HTTPException(400, "video_chunk_seconds must be between 1 and 120")

    upload_id = str(uuid4())
    upload_dir = settings.uploads_dir / upload_id
    video_path = upload_dir / Path(video.filename or "video.mp4").name
    transcript_path = upload_dir / Path(transcript.filename or "transcript.json").name
    await _save_upload(video, video_path)
    await _save_upload(transcript, transcript_path)

    job = jobs.create()
    asyncio.create_task(
        asyncio.to_thread(
            run_ingestion,
            job.id,
            str(video_path),
            str(transcript_path),
            asset_name,
            video_chunk_seconds,
        )
    )
    return {"job_id": job.id}


@app.post("/api/ingest/images/path")
async def ingest_images_path(request: ImagePathIngestRequest):
    paths = discover_image_paths(request.image_path)
    job = jobs.create()
    asyncio.create_task(
        asyncio.to_thread(
            run_image_ingestion,
            job.id,
            [str(path) for path in paths],
            request.asset_name,
        )
    )
    return {"job_id": job.id, "image_count": len(paths)}


@app.post("/api/ingest/images/upload")
async def ingest_images_upload(
    images: list[UploadFile] = File(...),
    asset_name: str | None = Form(default=None),
):
    if not images:
        raise HTTPException(400, "Upload at least one image")

    upload_id = str(uuid4())
    upload_dir = settings.uploads_dir / upload_id / "images"
    paths: list[Path] = []
    for index, upload in enumerate(images):
        safe_name = Path(upload.filename or f"image_{index:05d}.jpg").name
        destination = upload_dir / f"{index:05d}_{safe_name}"
        await _save_upload(upload, destination)
        paths.append(destination)

    job = jobs.create()
    asyncio.create_task(
        asyncio.to_thread(
            run_image_ingestion,
            job.id,
            [str(path) for path in paths],
            asset_name,
        )
    )
    return {"job_id": job.id, "image_count": len(paths)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


def _decorate_results(rows: list[dict]) -> list[dict]:
    for row in rows:
        asset_id = row.get("asset_id")
        payload = registry.get(asset_id) if asset_id else None
        public = registry.get_public(asset_id) if asset_id else None
        row["asset_name"] = (public or {}).get("name") or asset_id
        modality = row.get("modality")

        if modality in {"video", "image"} and asset_id:
            idx = int(row.get("chunk_index", 0))
            row["thumbnail_url"] = f"/api/assets/{asset_id}/thumb/{idx}"
        else:
            row["thumbnail_url"] = None

        if modality == "image" and asset_id:
            row["image_url"] = f"/api/assets/{asset_id}/image/{int(row.get('chunk_index', 0))}"
        else:
            row["image_url"] = None

        if payload and registry.asset_type(payload) == "video" and public and public["media_available"]:
            row["media_url"] = f"/api/assets/{asset_id}/media"
        else:
            row["media_url"] = None
    return rows


@app.post("/api/search")
def search(request: SearchRequest):
    query_vector = embedder.embed_query(request.query)
    rows = store.search(query_vector, request.asset_id, request.modality, request.limit)
    return {"query": request.query, "query_type": "text", "results": _decorate_results(rows)}


@app.post("/api/search/image")
async def search_by_image(
    query_image: UploadFile = File(...),
    asset_id: str | None = Form(default=None),
    modality: str = Form(default="all"),
    limit: int = Form(default=12),
):
    if modality not in {"all", "video", "transcript", "image"}:
        raise HTTPException(400, "Invalid modality filter")
    limit = max(1, min(50, int(limit)))
    data = await query_image.read()
    await query_image.close()
    if not data:
        raise HTTPException(400, "Query image is empty")
    try:
        with Image.open(BytesIO(data)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Could not decode query image: {exc}") from exc

    query_vector = embedder.embed_image(image)
    rows = store.search(query_vector, asset_id or None, modality, limit)
    return {
        "query": query_image.filename or "query image",
        "query_type": "image",
        "results": _decorate_results(rows),
    }


@app.get("/api/assets/{asset_id}/media")
def media(asset_id: str):
    asset = registry.get(asset_id)
    if not asset:
        raise HTTPException(404, "Asset not found")
    if registry.asset_type(asset) != "video":
        raise HTTPException(400, "This asset is not a video")
    path = Path(asset.get("video_path") or "")
    if not path.exists():
        raise HTTPException(404, "Original video file is no longer available")
    return FileResponse(path)


@app.get("/api/assets/{asset_id}/image/{image_index}")
def image_media(asset_id: str, image_index: int):
    asset = registry.get(asset_id)
    if not asset:
        raise HTTPException(404, "Asset not found")
    if registry.asset_type(asset) != "images":
        raise HTTPException(400, "This asset is not an image collection")
    paths = asset.get("image_paths") or []
    if image_index < 0 or image_index >= len(paths):
        raise HTTPException(404, "Image index out of range")
    path = Path(paths[image_index])
    if not path.is_file():
        raise HTTPException(404, "Original image file is no longer available")
    return FileResponse(path)


@app.get("/api/assets/{asset_id}/thumb/{chunk_index}")
def thumbnail(asset_id: str, chunk_index: int):
    path = settings.assets_dir / asset_id / "thumbs" / f"{chunk_index:05d}.jpg"
    if not path.exists():
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")


@app.on_event("shutdown")
def shutdown():
    store.close()
