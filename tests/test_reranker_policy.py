from app.ranking import resolve_candidate_limit, sort_by_m0


def test_m0_score_is_primary_ranking_signal():
    rows = [
        {"uuid": "dense-first", "dense_rank": 1, "distance": 0.10, "rerank_score": 0.20},
        {"uuid": "dense-thirty", "dense_rank": 30, "distance": 0.42, "rerank_score": 0.95},
    ]
    ranked = sort_by_m0(rows)
    assert [row["uuid"] for row in ranked] == ["dense-thirty", "dense-first"]
    assert ranked[0]["rerank_rank"] == 1


def test_dense_rank_breaks_only_equal_m0_scores():
    rows = [
        {"uuid": "b", "dense_rank": 9, "rerank_score": 0.7},
        {"uuid": "a", "dense_rank": 2, "rerank_score": 0.7},
    ]
    ranked = sort_by_m0(rows)
    assert [row["uuid"] for row in ranked] == ["a", "b"]


def test_candidate_limit_never_below_final_limit():
    assert resolve_candidate_limit(8, 24, default_limit=64, max_limit=200) == 24
    assert resolve_candidate_limit(120, 12, default_limit=64, max_limit=200) == 120
    assert resolve_candidate_limit(999, 12, default_limit=64, max_limit=200) == 200
