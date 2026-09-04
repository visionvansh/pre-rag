from __future__ import annotations

import hashlib
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from app.config import settings
from app.job_store import jobs
from .jina_v4_client import embedder
from .registry import registry
from .weaviate_store import store


TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".yaml", ".yml", ".xml", ".html", ".htm",
    ".toml", ".ini", ".cfg", ".conf", ".sql",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".go", ".rs", ".sh", ".bash", ".zsh",
}
LI_INGEST_LOCK = Lock()


def discover_text_paths(source: str | Path) -> list[Path]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            raise ValueError(f"Unsupported text extension: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Text file/folder not found: {path}")
    files = sorted(
        (p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    if not files:
        raise ValueError(f"No supported text files found under: {path}")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_id(paths: list[Path]) -> str:
    digest = hashlib.sha256(b"jina-v4-late-text-v1\0")
    for path in sorted(paths, key=lambda p: (p.name.lower(), str(p).lower())):
        digest.update(path.name.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if not data:
        return ""
    if b"\x00" in data[:8192]:
        raise ValueError(f"File looks binary rather than text: {path.name}")
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def _set(job_id: str, pct: float, stage: str, detail: str, **counters) -> None:
    jobs.update(
        job_id,
        status="running",
        stage=stage,
        overall_pct=round(float(pct), 2),
        detail=detail,
        counters=counters,
    )


def _run(job_id: str, text_paths: list[str], asset_name: str | None) -> None:
    try:
        paths = [Path(value).expanduser().resolve() for value in text_paths]
        if not paths:
            raise ValueError("No text files supplied")
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
                raise ValueError(f"Unsupported or missing text file: {path}")

        asset_id = _asset_id(paths)
        jobs.update(job_id, asset_id=asset_id)
        _set(job_id, 3, "Validate text", f"Validated {len(paths)} text files")
        store.ensure_collection()
        _set(job_id, 7, "Jina v4 worker", "Starting the isolated Jina v4 worker")
        embedder.status()

        chunks: list[dict] = []
        _set(job_id, 10, "Chunk text", "Chunking with the Jina-v4 tokenizer")
        for file_index, path in enumerate(paths):
            content = _read_text(path).strip()
            if content:
                file_chunks = embedder.chunk_text(
                    content,
                    chunk_size=int(settings.li_text_chunk_tokens),
                    overlap=int(settings.li_text_overlap_tokens),
                )
                for local_index, row in enumerate(file_chunks):
                    chunks.append(
                        {
                            "source_name": path.name,
                            "file_index": file_index,
                            "local_index": local_index,
                            "text": str(row["text"]),
                            "token_count": int(row["token_count"]),
                        }
                    )
            _set(
                job_id,
                10 + 15 * ((file_index + 1) / max(1, len(paths))),
                "Chunk text",
                f"Prepared {file_index + 1}/{len(paths)} files · {len(chunks)} chunks",
                text_file_total=len(paths),
                text_file_done=file_index + 1,
                text_chunk_total=len(chunks),
            )
        if not chunks:
            raise ValueError("No non-empty text chunks were produced")

        # Replacement is isolated to the dedicated LI collection/registry.
        store.delete_asset(asset_id)
        total = len(chunks)
        late_total = 0
        _set(
            job_id,
            26,
            "Embed/index text",
            f"Encoding {total} chunks; each forward returns dense + multi-vector outputs",
            text_chunk_total=total,
            text_chunk_done=0,
            late_vectors=0,
            weaviate_objects=0,
        )
        for index, item in enumerate(chunks):
            encoded = embedder.encode_text(item["text"], role="passage")
            late_count = int(encoded["late_vector_count"])
            late_total += late_count
            chunk_id = f"text_{index:05d}"
            store.insert(
                {
                    "asset_id": asset_id,
                    "chunk_id": chunk_id,
                    "modality": "text",
                    "chunk_index": index,
                    "text": item["text"],
                    "token_count": item["token_count"],
                    "source_name": item["source_name"],
                    "thumbnail_relpath": "",
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
                26 + 70 * (done / total),
                "Embed/index text",
                f"Text chunk {done}/{total} indexed · {late_total:,} late vectors",
                text_chunk_total=total,
                text_chunk_done=done,
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
                "name": asset_name or default_name or f"li-text-{asset_id[:8]}",
                "asset_type": "texts",
                "text_paths": [str(path) for path in paths],
                "text_file_count": len(paths),
                "text_chunks": total,
                "image_count": 0,
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
            detail=f"Indexed {total} text chunks with {late_total:,} late vectors",
            counters={
                "text_file_total": len(paths),
                "text_file_done": len(paths),
                "text_chunk_total": total,
                "text_chunk_done": total,
                "late_vectors": late_total,
                "weaviate_objects": object_count,
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


def run_text_ingestion(job_id: str, text_paths: list[str], asset_name: str | None = None) -> None:
    if not LI_INGEST_LOCK.acquire(blocking=False):
        jobs.update(job_id, status="queued", stage="Waiting for LI ingestion slot", detail="Another Jina-v4 LI ingestion is active")
        LI_INGEST_LOCK.acquire()
    try:
        _run(job_id, text_paths, asset_name)
    finally:
        LI_INGEST_LOCK.release()
