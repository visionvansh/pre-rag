from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from .config import settings
from .jina_mlx import embedder
from .jina_reranker import reranker
from .ranking import resolve_candidate_limit, sort_by_m0
from .registry import registry
from .weaviate_store import store


MODALITIES = ("video", "transcript", "image", "text")


def _decorate(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for source in rows:
        row = dict(source)
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
            row["image_url"] = (
                f"/api/assets/{asset_id}/image/{int(row.get('chunk_index', 0))}"
            )
        else:
            row["image_url"] = None

        if (
            payload
            and registry.asset_type(payload) == "video"
            and public
            and public["media_available"]
        ):
            row["media_url"] = f"/api/assets/{asset_id}/media"
        else:
            row["media_url"] = None

        # Internal reranking helpers must never leak filesystem paths to the browser.
        row.pop("_rerank_document", None)
        row.pop("_rerank_doc_type", None)
        output.append(row)
    return output


def _distance_key(row: dict) -> float:
    value = row.get("distance")
    return float(value) if value is not None else float("inf")


def _dedupe(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    output: list[dict] = []
    for row in rows:
        key = str(row.get("uuid") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(dict(row))
    return output


def _candidate_pool(
    query_vector: list[float],
    asset_id: str | None,
    modality: str,
    candidate_limit: int,
) -> tuple[list[dict], dict[str, int]]:
    """Build a high-recall union while preserving a chance for every modality."""
    counts: dict[str, int] = {}
    if modality != "all":
        rows = store.search(query_vector, asset_id, modality, candidate_limit)
        counts[modality] = len(rows)
        rows = _dedupe(rows)
        rows.sort(key=_distance_key)
        for rank, row in enumerate(rows, start=1):
            row["dense_rank"] = rank
        return rows[:candidate_limit], counts

    per_modality = max(1, math.ceil(candidate_limit / len(MODALITIES)))
    union: list[dict] = []
    for candidate_modality in MODALITIES:
        part = store.search(
            query_vector,
            asset_id,
            candidate_modality,
            per_modality,
        )
        counts[candidate_modality] = len(part)
        union.extend(part)

    # If one modality has few/no objects, use an ordinary global dense query to fill
    # the unused candidate budget without removing the balanced seeds above.
    if len(_dedupe(union)) < candidate_limit:
        union.extend(store.search(query_vector, asset_id, "all", candidate_limit))

    rows = _dedupe(union)
    rows.sort(key=_distance_key)
    rows = rows[:candidate_limit]
    for rank, row in enumerate(rows, start=1):
        row["dense_rank"] = rank
    return rows, counts


def _visual_path(row: dict) -> Path | None:
    asset_id = str(row.get("asset_id") or "")
    if not asset_id:
        return None
    chunk_index = int(row.get("chunk_index", 0))
    modality = row.get("modality")
    payload = registry.get(asset_id) or {}

    if modality == "image":
        paths = payload.get("image_paths") or []
        if 0 <= chunk_index < len(paths):
            original = Path(paths[chunk_index])
            if original.is_file():
                return original

    # New video ingestions create richer 2x2 contact sheets here. Existing video
    # assets, and missing original images, automatically fall back to thumbnails.
    rerank_preview = settings.assets_dir / asset_id / "rerank" / f"{chunk_index:05d}.jpg"
    if rerank_preview.is_file():
        return rerank_preview
    thumbnail = settings.assets_dir / asset_id / "thumbs" / f"{chunk_index:05d}.jpg"
    if thumbnail.is_file():
        return thumbnail
    return None


def _materialize_candidate(row: dict) -> tuple[str, Any] | None:
    modality = row.get("modality")
    if modality in {"text", "transcript"}:
        text = str(row.get("text") or "").strip()
        return ("text", text) if text else None
    if modality in {"image", "video"}:
        path = _visual_path(row)
        return ("image", str(path)) if path else None
    return None


def _rerank(
    query: Any,
    query_type: str,
    candidates: list[dict],
    final_limit: int,
) -> tuple[list[dict], dict[str, Any]]:
    prepared: dict[str, list[tuple[int, Any]]] = {"text": [], "image": []}
    skipped: list[int] = []
    for index, row in enumerate(candidates):
        materialized = _materialize_candidate(row)
        if materialized is None:
            skipped.append(index)
            continue
        doc_type, document = materialized
        prepared[doc_type].append((index, document))

    errors: list[str] = []
    scored = [dict(row) for row in candidates]
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

    # Primary ordering is ONLY m0 relevance. Dense rank is only a deterministic
    # tie-breaker / fallback for candidates that could not be materialized.
    scored = sort_by_m0(scored)

    diagnostics = {
        "scored_text": len(prepared["text"]),
        "scored_image": len(prepared["image"]),
        "skipped": len(skipped),
        "errors": errors,
    }
    return scored[:final_limit], diagnostics


def run_search(
    *,
    query: Any,
    query_label: str,
    query_type: str,
    query_vector: list[float],
    asset_id: str | None,
    modality: str,
    final_limit: int,
    candidate_limit: int | None,
    run_reranker: bool,
) -> dict[str, Any]:
    started = time.perf_counter()

    dense_started = time.perf_counter()
    # This is intentionally the exact same dense/vector result path the UI used
    # before m0 existed. It stays visible as the baseline comparison panel.
    dense_rows = store.search(query_vector, asset_id, modality, final_limit)
    for rank, row in enumerate(dense_rows, start=1):
        row["dense_rank"] = rank
    dense_ms = (time.perf_counter() - dense_started) * 1000.0

    resolved_candidates = resolve_candidate_limit(
        candidate_limit,
        final_limit,
        default_limit=int(settings.reranker_candidate_limit),
        max_limit=int(settings.reranker_max_candidates),
    )
    reranked_rows: list[dict] = []
    rerank_error: str | None = None
    candidate_counts: dict[str, int] = {}
    candidate_count = 0
    rerank_ms = 0.0
    rerank_details: dict[str, Any] = {}

    if run_reranker and settings.reranker_enabled:
        try:
            candidate_started = time.perf_counter()
            candidates, candidate_counts = _candidate_pool(
                query_vector,
                asset_id,
                modality,
                resolved_candidates,
            )
            candidate_count = len(candidates)
            candidate_ms = (time.perf_counter() - candidate_started) * 1000.0

            rerank_started = time.perf_counter()
            reranked_rows, rerank_details = _rerank(
                query,
                query_type,
                candidates,
                final_limit,
            )
            rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
            rerank_details["candidate_retrieval_ms"] = round(candidate_ms, 2)
            if rerank_details.get("errors"):
                rerank_error = "; ".join(rerank_details["errors"])
        except Exception as exc:
            rerank_error = f"{type(exc).__name__}: {exc}"
    elif run_reranker:
        rerank_error = "Reranking is disabled by configuration"

    total_ms = (time.perf_counter() - started) * 1000.0
    dense_public = _decorate(dense_rows)
    reranked_public = _decorate(reranked_rows)

    return {
        "query": query_label,
        "query_type": query_type,
        "dense_results": dense_public,
        "reranked_results": reranked_public,
        # Backward-compatible best available result list for API clients that still
        # read `results`; the webpage renders the two explicit panels above.
        "results": reranked_public if reranked_public else dense_public,
        "diagnostics": {
            "dense_ms": round(dense_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "total_ms": round(total_ms, 2),
            "final_limit": final_limit,
            "candidate_limit": resolved_candidates,
            "candidate_count": candidate_count,
            "candidate_counts": candidate_counts,
            "reranker_requested": bool(run_reranker),
            "reranker_status": reranker.status(),
            "rerank_error": rerank_error,
            "rerank_details": rerank_details,
        },
    }


def search_text(
    query: str,
    *,
    asset_id: str | None,
    modality: str,
    final_limit: int,
    candidate_limit: int | None,
    run_reranker: bool,
) -> dict[str, Any]:
    clean_query = query.strip()
    vector = embedder.embed_query(clean_query)
    return run_search(
        query=clean_query,
        query_label=clean_query,
        query_type="text",
        query_vector=vector,
        asset_id=asset_id,
        modality=modality,
        final_limit=final_limit,
        candidate_limit=candidate_limit,
        run_reranker=run_reranker,
    )


def search_image(
    image: Any,
    *,
    query_label: str,
    asset_id: str | None,
    modality: str,
    final_limit: int,
    candidate_limit: int | None,
    run_reranker: bool,
) -> dict[str, Any]:
    vector = embedder.embed_image(image)
    return run_search(
        query=image,
        query_label=query_label,
        query_type="image",
        query_vector=vector,
        asset_id=asset_id,
        modality=modality,
        final_limit=final_limit,
        candidate_limit=candidate_limit,
        run_reranker=run_reranker,
    )
