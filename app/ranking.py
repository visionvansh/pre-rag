from __future__ import annotations


def resolve_candidate_limit(
    requested: int | None,
    final_limit: int,
    *,
    default_limit: int,
    max_limit: int,
) -> int:
    raw = int(requested or default_limit)
    raw = max(int(final_limit), raw)
    return min(raw, int(max_limit), 200)


def sort_by_m0(rows: list[dict]) -> list[dict]:
    """Sort strictly by m0 relevance; use dense rank only for ties/fallback."""
    output = [dict(row) for row in rows]
    output.sort(
        key=lambda row: (
            -(float(row["rerank_score"]) if row.get("rerank_score") is not None else -1.0),
            int(row.get("dense_rank") or 10**9),
        )
    )
    for rank, row in enumerate(output, start=1):
        row["rerank_rank"] = rank
    return output
