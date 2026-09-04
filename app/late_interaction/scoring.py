from __future__ import annotations

import math
from typing import Iterable


def validate_dense(vector: Iterable[float], dims: int = 2048) -> list[float]:
    values = [float(v) for v in vector]
    if len(values) != dims:
        raise ValueError(f"Expected dense vector with {dims} dimensions, got {len(values)}")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Dense vector contains NaN or infinity")
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0.0:
        raise ValueError("Dense vector is zero")
    if abs(norm - 1.0) > 0.02:
        raise ValueError(f"Dense vector is not L2-normalized (norm={norm:.6f})")
    return values


def validate_multi(vectors: Iterable[Iterable[float]], dims: int = 128) -> list[list[float]]:
    rows = [[float(v) for v in row] for row in vectors]
    if not rows:
        raise ValueError("Multi-vector embedding is empty")
    for index, row in enumerate(rows):
        if len(row) != dims:
            raise ValueError(
                f"Expected late-interaction vector {index} with {dims} dimensions, got {len(row)}"
            )
        if not all(math.isfinite(v) for v in row):
            raise ValueError(f"Late-interaction vector {index} contains NaN or infinity")
        norm = math.sqrt(sum(v * v for v in row))
        if norm <= 0.0:
            raise ValueError(f"Late-interaction vector {index} is zero")
        if abs(norm - 1.0) > 0.02:
            raise ValueError(
                f"Late-interaction vector {index} is not L2-normalized (norm={norm:.6f})"
            )
    return rows


def maxsim_score(query_vectors: list[list[float]], document_vectors: list[list[float]]) -> float:
    """Reference ColBERT-style sum-of-MaxSim scorer used only in tests/preflight."""
    q = validate_multi(query_vectors)
    d = validate_multi(document_vectors)
    total = 0.0
    for qv in q:
        best = max(sum(a * b for a, b in zip(qv, dv)) for dv in d)
        total += best
    return total


def sort_by_m0(rows: list[dict]) -> list[dict]:
    """m0 is the sole final ordering; late rank is only a deterministic tie-break."""
    output = [dict(row) for row in rows]
    output.sort(
        key=lambda row: (
            -(float(row["rerank_score"]) if row.get("rerank_score") is not None else -1.0),
            int(row.get("late_rank") or 10**9),
        )
    )
    for rank, row in enumerate(output, start=1):
        row["rerank_rank"] = rank
    return output
