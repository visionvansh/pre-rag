from __future__ import annotations

import hashlib
import traceback
from datetime import datetime, timezone
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .ingestion import INGEST_LOCK, MODEL_NAME, _set
from .jina_mlx import embedder
from .job_store import jobs
from .registry import registry
from .weaviate_store import store


TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".yaml", ".yml", ".xml", ".html", ".htm",
    ".toml", ".ini", ".cfg", ".conf", ".sql",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".go", ".rs", ".sh", ".bash", ".zsh",
}


def discover_text_paths(source: str | Path) -> list[Path]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            raise ValueError(f"Unsupported text extension: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Text file/folder not found: {path}")

    files = sorted(
        (
            p.resolve()
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS
        ),
        key=lambda p: str(p).lower(),
    )
    if not files:
        raise ValueError(f"No supported text files found under: {path}")
    return files


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_asset_id(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: (p.name.lower(), str(p).lower())):
        digest.update(path.name.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
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


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.transcript_chunk_tokens,
        chunk_overlap=settings.transcript_overlap_tokens,
        length_function=embedder.token_length,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )


def _run_text_ingestion(
    job_id: str,
    text_paths: list[str],
    asset_name: str | None = None,
):
    try:
        paths = [Path(p).expanduser().resolve() for p in text_paths]
        if not paths:
            raise ValueError("No text files supplied")

        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Text file not found: {path}")
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                raise ValueError(f"Unsupported text extension: {path.suffix}")

        _set(job_id, 3, "Validate text files", f"Checking {len(paths)} text files")
        asset_id = _text_asset_id(paths)
        jobs.update(job_id, asset_id=asset_id)

        _set(job_id, 7, "Connect Weaviate", "Checking local Weaviate and schema")
        store.ensure_collection()

        _set(job_id, 11, "Load Jina MLX", "Loading the local Jina text embedding path")
        embedder.load()

        splitter = _splitter()
        chunks: list[dict] = []
        non_empty_files = 0

        _set(
            job_id,
            15,
            "Chunk text",
            "Recursively chunking each file independently with the Jina tokenizer",
            text_file_total=len(paths),
            text_file_done=0,
            text_chunk_total=0,
            text_chunk_done=0,
        )

        for file_index, path in enumerate(paths):
            content = _read_text(path).strip()
            if content:
                non_empty_files += 1
                for local_index, chunk_text in enumerate(splitter.split_text(content)):
                    cleaned = chunk_text.strip()
                    if not cleaned:
                        continue
                    chunks.append(
                        {
                            "source_path": path,
                            "source_name": path.name,
                            "file_index": file_index,
                            "local_index": local_index,
                            "text": cleaned,
                            "token_count": embedder.token_length(cleaned),
                        }
                    )
            _set(
                job_id,
                15 + 10 * ((file_index + 1) / max(1, len(paths))),
                "Chunk text",
                f"Prepared {file_index + 1}/{len(paths)} files · {len(chunks)} chunks",
                text_file_done=file_index + 1,
                text_chunk_total=len(chunks),
            )

        if non_empty_files == 0 or not chunks:
            raise ValueError("No non-empty text content was found in the supplied files")

        store.delete_asset(asset_id)

        total = len(chunks)
        _set(
            job_id,
            26,
            "Embed/index text",
            f"Embedding {total} recursive text chunks",
            text_file_total=len(paths),
            text_file_done=len(paths),
            text_chunk_total=total,
            text_chunk_done=0,
            weaviate_objects=0,
        )

        done = 0
        batch_size = 8
        for offset in range(0, total, batch_size):
            batch = chunks[offset: offset + batch_size]
            vectors = embedder.embed_documents_batch([item["text"] for item in batch])

            for item, vector in zip(batch, vectors):
                chunk_index = done
                chunk_id = f"text_{chunk_index:05d}"
                store.insert(
                    {
                        "asset_id": asset_id,
                        "chunk_id": chunk_id,
                        "modality": "text",
                        "chunk_index": chunk_index,
                        "start_sec": 0.0,
                        "end_sec": 0.0,
                        "duration_sec": 0.0,
                        "text": item["text"],
                        "speaker_ids": [],
                        "token_count": item["token_count"],
                        "frame_count": 0,
                        "source_name": item["source_name"],
                        "thumbnail_relpath": "",
                        "embedding_model": MODEL_NAME,
                        "embedding_dim": 1024,
                    },
                    vector,
                )
                done += 1

            pct = 26 + 70 * (done / max(1, total))
            _set(
                job_id,
                pct,
                "Embed/index text",
                f"Text chunk {done}/{total} indexed",
                text_chunk_done=done,
                weaviate_objects=done,
            )

        total_objects = store.count_asset(asset_id)
        default_name = paths[0].parent.name if len(paths) > 1 else paths[0].stem
        registry.upsert(
            asset_id,
            {
                "name": asset_name or default_name or f"text-set-{asset_id[:8]}",
                "asset_type": "texts",
                "text_paths": [str(path) for path in paths],
                "text_file_count": len(paths),
                "text_chunks": total,
                "duration_sec": 0.0,
                "video_chunks": 0,
                "transcript_chunks": 0,
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
            detail=f"Indexed {total_objects} text chunks for asset {asset_id}",
            counters={
                "text_file_total": len(paths),
                "text_file_done": len(paths),
                "text_chunk_total": total,
                "text_chunk_done": total,
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


def run_text_ingestion(
    job_id: str,
    text_paths: list[str],
    asset_name: str | None = None,
):
    if not INGEST_LOCK.acquire(blocking=False):
        jobs.update(
            job_id,
            status="queued",
            stage="Waiting for ingestion slot",
            detail="Another local MLX ingestion job is active; this text job will start next.",
        )
        INGEST_LOCK.acquire()
    try:
        return _run_text_ingestion(job_id, text_paths, asset_name)
    finally:
        INGEST_LOCK.release()
