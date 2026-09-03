from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .ingestion import run_ingestion
from .jina_mlx import embedder
from .job_store import jobs
from .registry import registry
from .schemas import PathIngestRequest, SearchRequest
from .weaviate_store import store


app = FastAPI(title="Jina v5 Omni Video RAG Lab")
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
        "transcript_chunk_tokens": settings.transcript_chunk_tokens,
        "transcript_overlap_tokens": settings.transcript_overlap_tokens,
    }


@app.get("/api/assets")
def assets():
    return registry.list()


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
):
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
        )
    )
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/api/search")
def search(request: SearchRequest):
    query_vector = embedder.embed_query(request.query)
    rows = store.search(query_vector, request.asset_id, request.modality, request.limit)
    for row in rows:
        if row.get("modality") == "video" and row.get("asset_id"):
            idx = int(row.get("chunk_index", 0))
            row["thumbnail_url"] = f"/api/assets/{row['asset_id']}/thumb/{idx}"
        else:
            row["thumbnail_url"] = None
    return {"query": request.query, "results": rows}


@app.get("/api/assets/{asset_id}/media")
def media(asset_id: str):
    asset = registry.get(asset_id)
    if not asset:
        raise HTTPException(404, "Asset not found")
    path = Path(asset["video_path"])
    if not path.exists():
        raise HTTPException(404, "Original video file is no longer available")
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
