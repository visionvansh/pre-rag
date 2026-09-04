from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from app.config import settings
from app.jina_reranker import reranker
from app.job_store import jobs
from .image_ingestion import discover_image_paths, run_image_ingestion
from .jina_v4_client import embedder
from .registry import registry
from .schemas import LIImagePathIngestRequest, LISearchRequest, LITextPathIngestRequest
from .search_service import search_image, search_text
from .text_ingestion import discover_text_paths, run_text_ingestion
from .weaviate_store import store


router = APIRouter(prefix="/api/li", tags=["late-interaction"])


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out:
        while True:
            chunk = await upload.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    await upload.close()


@router.get("/health")
def health():
    weaviate_ready = False
    weaviate_error = None
    collection_ready = False
    try:
        weaviate_ready = bool(store.connect().is_ready())
        store.ensure_collection()
        collection_ready = True
    except Exception as exc:
        weaviate_error = str(exc)
    return {
        "ok": True,
        "lab": "jina-v4-late-interaction",
        "model": embedder.status(),
        "weaviate_ready": weaviate_ready,
        "collection_ready": collection_ready,
        "weaviate_error": weaviate_error,
        "collection": settings.li_weaviate_collection,
        "reranker": reranker.status(),
        "supported_modalities": ["image", "text"],
        "text_chunk_tokens": settings.li_text_chunk_tokens,
        "text_overlap_tokens": settings.li_text_overlap_tokens,
    }


@router.get("/assets")
def assets():
    return registry.list_public()


@router.get("/assets/{asset_id}")
def asset_detail(asset_id: str):
    item = registry.get_public(asset_id)
    if not item:
        raise HTTPException(404, "Late-interaction asset not found")
    return item


@router.delete("/assets/{asset_id}")
def remove_asset(asset_id: str):
    if not registry.get(asset_id):
        raise HTTPException(404, "Late-interaction asset not found")
    store.delete_asset(asset_id)
    registry.remove(asset_id)
    generated = settings.li_assets_dir / asset_id
    if generated.exists():
        shutil.rmtree(generated, ignore_errors=True)
    return {"ok": True, "asset_id": asset_id}


@router.post("/ingest/images/path")
async def ingest_images_path(request: LIImagePathIngestRequest):
    paths = discover_image_paths(request.image_path)
    job = jobs.create()
    asyncio.create_task(asyncio.to_thread(run_image_ingestion, job.id, [str(p) for p in paths], request.asset_name))
    return {"job_id": job.id, "image_count": len(paths)}


@router.post("/ingest/images/upload")
async def ingest_images_upload(images: list[UploadFile] = File(...), asset_name: str | None = Form(default=None)):
    if not images:
        raise HTTPException(400, "Upload at least one image")
    upload_dir = settings.li_uploads_dir / uuid4().hex / "images"
    paths: list[Path] = []
    for index, upload in enumerate(images):
        destination = upload_dir / f"{index:05d}_{Path(upload.filename or f'image_{index}.jpg').name}"
        await _save_upload(upload, destination)
        paths.append(destination)
    job = jobs.create()
    asyncio.create_task(asyncio.to_thread(run_image_ingestion, job.id, [str(p) for p in paths], asset_name))
    return {"job_id": job.id, "image_count": len(paths)}


@router.post("/ingest/texts/path")
async def ingest_texts_path(request: LITextPathIngestRequest):
    paths = discover_text_paths(request.text_path)
    job = jobs.create()
    asyncio.create_task(asyncio.to_thread(run_text_ingestion, job.id, [str(p) for p in paths], request.asset_name))
    return {"job_id": job.id, "text_file_count": len(paths)}


@router.post("/ingest/texts/upload")
async def ingest_texts_upload(texts: list[UploadFile] = File(...), asset_name: str | None = Form(default=None)):
    if not texts:
        raise HTTPException(400, "Upload at least one text file")
    upload_dir = settings.li_uploads_dir / uuid4().hex / "texts"
    paths: list[Path] = []
    for index, upload in enumerate(texts):
        destination = upload_dir / f"{index:05d}_{Path(upload.filename or f'text_{index}.txt').name}"
        await _save_upload(upload, destination)
        paths.append(destination)
    job = jobs.create()
    asyncio.create_task(asyncio.to_thread(run_text_ingestion, job.id, [str(p) for p in paths], asset_name))
    return {"job_id": job.id, "text_file_count": len(paths)}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.post("/search/text")
def search_by_text(request: LISearchRequest):
    return search_text(
        request.query,
        asset_id=request.asset_id,
        modality=request.modality,
        final_limit=request.limit,
        late_candidate_limit=request.late_candidate_limit,
        m0_candidate_limit=request.m0_candidate_limit,
        run_reranker=request.rerank,
    )


@router.post("/search/image")
async def search_by_image(
    query_image: UploadFile = File(...),
    asset_id: str | None = Form(default=None),
    modality: str = Form(default="all"),
    limit: int = Form(default=12),
    late_candidate_limit: int | None = Form(default=None),
    m0_candidate_limit: int | None = Form(default=None),
    rerank: bool = Form(default=True),
):
    if modality not in {"all", "image", "text"}:
        raise HTTPException(400, "Invalid late-interaction modality")
    limit = max(1, min(50, int(limit)))
    query_dir = settings.li_query_tmp_dir
    query_dir.mkdir(parents=True, exist_ok=True)
    destination = query_dir / f"{uuid4().hex}_{Path(query_image.filename or 'query.png').name}"
    await _save_upload(query_image, destination)
    try:
        with Image.open(destination) as raw:
            raw.verify()
        return await asyncio.to_thread(
            search_image,
            destination,
            query_label=query_image.filename or "image query",
            asset_id=asset_id or None,
            modality=modality,
            final_limit=limit,
            late_candidate_limit=late_candidate_limit,
            m0_candidate_limit=m0_candidate_limit,
            run_reranker=rerank,
        )
    except Exception:
        raise
    finally:
        destination.unlink(missing_ok=True)


@router.get("/assets/{asset_id}/image/{image_index}")
def image_media(asset_id: str, image_index: int):
    payload = registry.get(asset_id)
    if not payload or payload.get("asset_type") != "images":
        raise HTTPException(404, "Image asset not found")
    paths = payload.get("image_paths") or []
    if image_index < 0 or image_index >= len(paths):
        raise HTTPException(404, "Image index out of range")
    path = Path(paths[image_index])
    if not path.is_file():
        raise HTTPException(404, "Original image is unavailable")
    return FileResponse(path)


@router.get("/assets/{asset_id}/thumb/{image_index}")
def thumbnail(asset_id: str, image_index: int):
    path = settings.li_assets_dir / asset_id / "thumbs" / f"{image_index:05d}.jpg"
    if not path.is_file():
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")
