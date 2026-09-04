from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.jina_reranker import reranker
from .jina_v4_client import embedder
from .registry import registry
from .scoring import sort_by_m0
from .weaviate_store import store


MODALITIES = {"all", "text", "image"}


def _decorate(rows: list[dict], *, rank_field: str) -> list[dict]:
    output: list[dict] = []
    for rank, source in enumerate(rows, start=1):
        row = dict(source)
        row[rank_field] = rank
        asset_id = str(row.get("asset_id") or "")
        public = registry.get_public(asset_id) if asset_id else None
        row["asset_name"] = (public or {}).get("name") or asset_id
        modality = row.get("modality")
        if modality == "image" and asset_id:
            index = int(row.get("chunk_index") or 0)
            row["thumbnail_url"] = f"/api/li/assets/{asset_id}/thumb/{index}"
            row["image_url"] = f"/api/li/assets/{asset_id}/image/{index}"
        else:
            row["thumbnail_url"] = None
            row["image_url"] = None
        row.pop("_rerank_document", None)
        row.pop("_rerank_doc_type", None)
        output.append(row)
    return output


def _materialize(row: dict) -> tuple[str, Any] | None:
    modality = str(row.get("modality") or "")
    if modality == "text":
        text = str(row.get("text") or "").strip()
        return ("text", text) if text else None
    if modality == "image":
        asset_id = str(row.get("asset_id") or "")
        payload = registry.get(asset_id) or {}
        paths = payload.get("image_paths") or []
        index = int(row.get("chunk_index") or 0)
        if 0 <= index < len(paths):
            path = Path(paths[index])
            if path.is_file():
                return "image", str(path)
    return None


def _m0_rerank(query: Any, query_type: str, candidates: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    prepared: dict[str, list[tuple[int, Any]]] = {"text": [], "image": []}
    scored = [dict(row) for row in candidates]
    for index, row in enumerate(scored):
        value = _materialize(row)
        if value:
            doc_type, document = value
            prepared[doc_type].append((index, document))

    errors: list[str] = []
    for doc_type in ("text", "image"):
        group = prepared[doc_type]
        if not group:
            continue
        indices = [index for index, _ in group]
        documents = [document for _, document in group]
        try:
            scores = reranker.score_documents(
                query,
                documents,
                query_type=query_type,
                doc_type=doc_type,
            )
            for index, score in zip(indices, scores):
                scored[index]["rerank_score"] = float(score)
        except Exception as exc:
            errors.append(f"{doc_type}: {type(exc).__name__}: {exc}")
    scored = sort_by_m0(scored)
    return scored, {
        "scored_text": len(prepared["text"]),
        "scored_image": len(prepared["image"]),
        "errors": errors,
    }


def _resolve_late_limit(requested: int | None, final_limit: int) -> int:
    raw = int(requested or settings.li_late_candidate_limit)
    return max(final_limit, min(raw, int(settings.li_late_max_candidates), 200))


def _resolve_m0_limit(requested: int | None, final_limit: int, late_count: int) -> int:
    raw = int(requested or settings.li_m0_candidate_limit)
    return min(late_count, max(final_limit, min(raw, 100)))


def run_search(
    *,
    query: Any,
    query_label: str,
    query_type: str,
    encoded: dict[str, Any],
    asset_id: str | None,
    modality: str,
    final_limit: int,
    late_candidate_limit: int | None,
    m0_candidate_limit: int | None,
    run_reranker: bool,
) -> dict[str, Any]:
    if modality not in MODALITIES:
        raise ValueError(f"Unsupported LI modality: {modality}")
    started = time.perf_counter()

    dense_started = time.perf_counter()
    dense_rows = store.search_dense(encoded["dense"], asset_id, modality, final_limit)
    dense_ms = (time.perf_counter() - dense_started) * 1000.0
    dense_public = _decorate(dense_rows, rank_field="dense_rank")
    for row in dense_public:
        row["dense_distance"] = row.pop("distance", None)

    resolved_late = _resolve_late_limit(late_candidate_limit, final_limit)
    late_started = time.perf_counter()
    late_rows = store.search_late(encoded["multi"], asset_id, modality, resolved_late)
    late_ms = (time.perf_counter() - late_started) * 1000.0
    late_public_all = _decorate(late_rows, rank_field="late_rank")
    for row in late_public_all:
        row["late_distance"] = row.pop("distance", None)
    late_public = late_public_all[:final_limit]

    m0_rows: list[dict] = []
    m0_error: str | None = None
    m0_ms = 0.0
    m0_details: dict[str, Any] = {}
    resolved_m0 = _resolve_m0_limit(m0_candidate_limit, final_limit, len(late_public_all))

    if run_reranker and settings.reranker_enabled and late_public_all:
        # In low-memory mode release Jina-v4 before loading/using the 2.4B m0 worker.
        if settings.li_low_memory_mode:
            embedder.close()
        m0_started = time.perf_counter()
        try:
            m0_rows, m0_details = _m0_rerank(
                query,
                query_type,
                late_public_all[:resolved_m0],
            )
            if not any(row.get("rerank_score") is not None for row in m0_rows):
                m0_rows = []
            else:
                m0_rows = m0_rows[:final_limit]
            if m0_details.get("errors"):
                m0_error = "; ".join(m0_details["errors"])
        except Exception as exc:
            m0_error = f"{type(exc).__name__}: {exc}"
        m0_ms = (time.perf_counter() - m0_started) * 1000.0
    elif run_reranker and not settings.reranker_enabled:
        m0_error = "m0 is disabled by RERANKER_ENABLED=false"

    final_rows = m0_rows if m0_rows else late_public
    return {
        "query": query_label,
        "query_type": query_type,
        "dense_results": dense_public,
        "late_results": late_public,
        "m0_results": m0_rows,
        "results": final_rows,
        "diagnostics": {
            "dense_ms": round(dense_ms, 2),
            "late_ms": round(late_ms, 2),
            "m0_ms": round(m0_ms, 2),
            "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "query_dense_dim": int(encoded.get("dense_dim") or len(encoded["dense"])),
            "query_late_dim": int(encoded.get("late_dim") or (len(encoded["multi"][0]) if encoded["multi"] else 0)),
            "query_late_vectors": int(encoded.get("late_vector_count") or len(encoded["multi"])),
            "late_candidate_limit": resolved_late,
            "late_candidate_count": len(late_public_all),
            "m0_candidate_limit": resolved_m0,
            "m0_requested": bool(run_reranker),
            "m0_status": reranker.status(),
            "m0_error": m0_error,
            "m0_details": m0_details,
            "v4_status": embedder.status(),
        },
    }


def search_text(
    query: str,
    *,
    asset_id: str | None,
    modality: str,
    final_limit: int,
    late_candidate_limit: int | None,
    m0_candidate_limit: int | None,
    run_reranker: bool,
) -> dict[str, Any]:
    clean = query.strip()
    encoded = embedder.encode_text(clean, role="query")
    return run_search(
        query=clean,
        query_label=clean,
        query_type="text",
        encoded=encoded,
        asset_id=asset_id,
        modality=modality,
        final_limit=final_limit,
        late_candidate_limit=late_candidate_limit,
        m0_candidate_limit=m0_candidate_limit,
        run_reranker=run_reranker,
    )


def search_image(
    path: str | Path,
    *,
    query_label: str,
    asset_id: str | None,
    modality: str,
    final_limit: int,
    late_candidate_limit: int | None,
    m0_candidate_limit: int | None,
    run_reranker: bool,
) -> dict[str, Any]:
    path = Path(path).expanduser().absolute()
    encoded = embedder.encode_image(path)
    return run_search(
        query=str(path),
        query_label=query_label,
        query_type="image",
        encoded=encoded,
        asset_id=asset_id,
        modality=modality,
        final_limit=final_limit,
        late_candidate_limit=late_candidate_limit,
        m0_candidate_limit=m0_candidate_limit,
        run_reranker=run_reranker,
    )
