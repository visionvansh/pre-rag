import math

from app.late_interaction.scoring import maxsim_score, sort_by_m0, validate_dense, validate_multi


def _unit(values):
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def test_maxsim_prefers_matching_token_vectors():
    q = [_unit([1, 0] + [0] * 126), _unit([0, 1] + [0] * 126)]
    relevant = [_unit([1, 0] + [0] * 126), _unit([0, 1] + [0] * 126)]
    unrelated = [_unit([-1, 0] + [0] * 126), _unit([0, -1] + [0] * 126)]
    assert maxsim_score(q, relevant) > maxsim_score(q, unrelated)


def test_m0_is_final_authority_over_late_rank():
    rows = [
        {"late_rank": 1, "rerank_score": 0.2},
        {"late_rank": 30, "rerank_score": 0.95},
    ]
    ranked = sort_by_m0(rows)
    assert ranked[0]["late_rank"] == 30
    assert ranked[0]["rerank_rank"] == 1


def test_vector_validation_contracts():
    dense = _unit([1.0] + [0.0] * 2047)
    multi = [_unit([1.0] + [0.0] * 127), _unit([0.0, 1.0] + [0.0] * 126)]
    assert len(validate_dense(dense)) == 2048
    assert len(validate_multi(multi)) == 2
