from __future__ import annotations

import hashlib
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from .asr_parser import load_asr, recursive_chunks
from .config import settings
from .jina_mlx import embedder
from .job_store import jobs
from .registry import registry
from .video_chunks import iter_video_chunks, probe_video
from .weaviate_store import store


MODEL_NAME = "jina-embeddings-v5-omni-small-retrieval-mlx"


def _asset_id(asr: dict, video_path: Path) -> str:
    checksum = (asr.get("source") or {}).get("checksum_sha256")
    if checksum:
        return str(checksum)[:24]
    stat = video_path.stat()
    seed = f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(seed.encode()).hexdigest()[:24]


def _set(job_id: str, pct: float, stage: str, detail: str, **counters):
    current = jobs.get(job_id) or {}
    merged = dict(current.get("counters") or {})
    merged.update(counters)
    jobs.update(
        job_id,
        status="running",
        overall_pct=round(max(0.0, min(100.0, pct)), 2),
        stage=stage,
        detail=detail,
        counters=merged,
    )


def _run_ingestion(job_id: str, video_path: str, transcript_path: str, asset_name: str | None = None):
    try:
        video = Path(video_path).expanduser().resolve()
        transcript = Path(transcript_path).expanduser().resolve()
        _set(job_id, 2, "Validate inputs", "Checking video and ASR JSON")
        if not video.is_file():
            raise FileNotFoundError(f"Video not found: {video}")
        if not transcript.is_file():
            raise FileNotFoundError(f"Transcript JSON not found: {transcript}")

        asr = load_asr(transcript)
        asset_id = _asset_id(asr, video)
        jobs.update(job_id, asset_id=asset_id)

        _set(job_id, 5, "Connect Weaviate", "Checking local Weaviate and schema")
        store.ensure_collection()

        _set(job_id, 7, "Load Jina MLX", "Loading the local Jina checkpoint on Apple Silicon")
        embedder.load()

        _set(job_id, 9, "Video preflight", "Testing Jina video processor and MLX vision tower")
        embedder.video_preflight()

        store.delete_asset(asset_id)

        _set(job_id, 12, "Chunk transcript", "Recursive LangChain chunking with the Jina tokenizer")
        text_chunks = recursive_chunks(
            asr,
            token_length=embedder.token_length,
            chunk_size=settings.transcript_chunk_tokens,
            chunk_overlap=settings.transcript_overlap_tokens,
        )
        _set(
            job_id, 15, "Chunk transcript",
            f"Created {len(text_chunks)} timestamped recursive transcript chunks",
            transcript_total=len(text_chunks), transcript_done=0,
        )

        batch_size = 8
        done = 0
        for offset in range(0, len(text_chunks), batch_size):
            batch = text_chunks[offset: offset + batch_size]
            vectors = embedder.embed_documents_batch([c.text for c in batch])
            for chunk, vector in zip(batch, vectors):
                chunk_id = f"transcript_{chunk.chunk_index:05d}"
                store.insert(
                    {
                        "asset_id": asset_id,
                        "chunk_id": chunk_id,
                        "modality": "transcript",
                        "chunk_index": chunk.chunk_index,
                        "start_sec": chunk.start_sec,
                        "end_sec": chunk.end_sec,
                        "duration_sec": chunk.end_sec - chunk.start_sec,
                        "text": chunk.text,
                        "speaker_ids": chunk.speaker_ids,
                        "token_count": chunk.token_count,
                        "frame_count": 0,
                        "source_name": video.name,
                        "thumbnail_relpath": "",
                        "embedding_model": MODEL_NAME,
                        "embedding_dim": 1024,
                    },
                    vector,
                )
                done += 1
            pct = 15 + 15 * (done / max(1, len(text_chunks)))
            _set(
                job_id, pct, "Index transcript",
                f"Embedded/indexed transcript chunk {done}/{len(text_chunks)}",
                transcript_done=done,
                weaviate_objects=done,
            )

        info = probe_video(video)
        duration = info["duration_sec"]
        video_total = int(math.ceil(duration / settings.video_chunk_seconds))
        _set(
            job_id, 32, "Chunk video",
            f"Video duration {duration:.1f}s → about {video_total} fixed 10-second chunks",
            video_total=video_total, video_done=0,
        )

        asset_dir = settings.assets_dir / asset_id
        thumbs_dir = asset_dir / "thumbs"
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        video_done = 0
        for chunk in iter_video_chunks(
            video,
            chunk_seconds=settings.video_chunk_seconds,
            max_frames=settings.video_max_frames,
        ):
            vector = embedder.embed_video_frames(chunk.frames)
            thumb_name = f"{chunk.chunk_index:05d}.jpg"
            thumb_path = thumbs_dir / thumb_name
            chunk.frames[len(chunk.frames) // 2].save(thumb_path, format="JPEG", quality=82)

            chunk_id = f"video_{chunk.chunk_index:05d}"
            store.insert(
                {
                    "asset_id": asset_id,
                    "chunk_id": chunk_id,
                    "modality": "video",
                    "chunk_index": chunk.chunk_index,
                    "start_sec": chunk.start_sec,
                    "end_sec": chunk.end_sec,
                    "duration_sec": chunk.end_sec - chunk.start_sec,
                    "text": "",
                    "speaker_ids": [],
                    "token_count": 0,
                    "frame_count": len(chunk.frames),
                    "source_name": video.name,
                    "thumbnail_relpath": f"thumbs/{thumb_name}",
                    "embedding_model": MODEL_NAME,
                    "embedding_dim": 1024,
                },
                vector,
            )
            video_done += 1
            pct = 32 + 63 * (video_done / max(1, video_total))
            _set(
                job_id, pct, "Embed/index video",
                f"10-second video chunk {video_done}/{video_total} indexed",
                video_done=video_done,
                weaviate_objects=len(text_chunks) + video_done,
            )

        total_objects = store.count_asset(asset_id)
        registry.upsert(
            asset_id,
            {
                "name": asset_name or video.stem,
                "asset_type": "video",
                "video_path": str(video),
                "transcript_path": str(transcript),
                "duration_sec": duration,
                "video_chunks": video_done,
                "transcript_chunks": len(text_chunks),
                "image_count": 0,
                "weaviate_objects": total_objects,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        jobs.update(
            job_id,
            status="completed",
            stage="Complete",
            overall_pct=100.0,
            detail=f"Indexed {total_objects} objects for asset {asset_id}",
            counters={
                "transcript_total": len(text_chunks),
                "transcript_done": len(text_chunks),
                "video_total": video_total,
                "video_done": video_done,
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


INGEST_LOCK = Lock()


def run_ingestion(job_id: str, video_path: str, transcript_path: str, asset_name: str | None = None):
    if not INGEST_LOCK.acquire(blocking=False):
        jobs.update(
            job_id,
            status="queued",
            stage="Waiting for ingestion slot",
            detail="Another local MLX ingestion job is active; this job will start next.",
        )
        INGEST_LOCK.acquire()
    try:
        return _run_ingestion(job_id, video_path, transcript_path, asset_name)
    finally:
        INGEST_LOCK.release()
