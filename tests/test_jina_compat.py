from app.jina_compat import patch_jina_model_source


def test_known_vision_repeat_bug_is_patched_exactly_once():
    source = """
for i in range(grid_thw.shape[0]):
    seq_len_i = grid_thw[i, 1] * grid_thw[i, 2]
    cu_seqlens.append(mx.repeat(seq_len_i, grid_thw[i, 0]))
"""
    patched, report = patch_jina_model_source(source)
    assert report.vision_repeat_patch_applied is True
    assert "mx.repeat(seq_len_i, int(grid_thw[i, 0].item()))" in patched
    assert "mx.repeat(seq_len_i, grid_thw[i, 0])" not in patched


def test_patch_is_idempotent_when_upstream_is_already_fixed():
    source = "cu_seqlens.append(mx.repeat(seq_len_i, int(grid_thw[i, 0].item())))"
    patched, report = patch_jina_model_source(source)
    assert patched == source
    assert report.vision_repeat_patch_applied is False


def test_unknown_upstream_source_is_not_modified():
    source = "cu_seqlens = build_cu_seqlens(grid_thw)"
    patched, report = patch_jina_model_source(source)
    assert patched == source
    assert report.vision_repeat_patch_applied is False
