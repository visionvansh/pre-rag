# jina-reranker-m0 integration

`pre-rag` uses a strict two-stage retrieval policy:

1. `jina-embeddings-v5-omni-small-retrieval-mlx` + Weaviate retrieve candidates.
2. `jina-reranker-m0` scores the actual candidate contents and decides the final order.

There is **no weighted blend** between vector distance and m0 relevance. Dense rank is retained only for diagnostics and deterministic tie-breaking when m0 scores are equal.

## Why m0 uses a separate Python environment

The main app needs the newer Transformers/Qwen3-VL stack used by Jina-v5 Omni image/video preprocessing.

`jina-reranker-m0`, however, was built on the older Qwen2-VL parameter layout. With newer Transformers, the Qwen2-VL text model is nested under `model.language_model`, while the m0 checkpoint contains keys such as `model.layers.*`. That can produce a dangerous partial load with many `UNEXPECTED` and `MISSING` parameters instead of a clean error.

To prevent either model from breaking the other, m0 runs in a persistent local worker process using:

```text
.venv-reranker/bin/python
transformers==4.48.3
```

The main FastAPI process stays in `.venv` with the newer Transformers version required by Jina-v5 Omni.

The worker is lazy: it starts on the first reranked search/preflight, loads m0 once on MPS, then remains resident and communicates with the main app over local stdin/stdout JSON messages. No extra TCP port is opened.

## Why the worker has an `mm_token_type_ids` bridge

Jina's Sentence-Transformers v5.4 integration commit (`94bfe0a...`) updated `modeling.py` so `JinaVLForRanking.forward()` forwards `mm_token_type_ids` into the base Qwen2-VL model.

That newer keyword is not accepted by the pre-refactor Qwen2-VL `forward()` in Transformers 4.48.3. However, 4.48.3 already computes the equivalent multimodal RoPE positions directly from the actual vision token IDs plus `image_grid_thw` / `video_grid_thw`.

Therefore the worker installs a narrow, version-gated adapter at the Qwen2-VL base-class boundary:

```text
Jina forward(..., mm_token_type_ids=...)
          ↓
compat adapter accepts that one newer keyword
          ↓
Transformers 4.48.3 Qwen2-VL forward(...)
          ↓
4.48.3 derives multimodal RoPE from input_ids/grid_thw
```

The adapter does **not** alter embeddings, ranking logits, score normalization, image pixels, or candidate order. It only removes the keyword that the older base forward does not understand.

The worker also requests Transformers loading diagnostics and refuses to run when there are missing or mismatched checkpoint weights. `lm_head.weight` is the only allowed unused key because Jina intentionally replaces the language-model head with `Identity` and uses the ranking `score` head instead.

## Local model path

Default:

```text
/Volumes/vision/Downloads/codes_necessary/models/jina-reranker-m0
```

### Existing download at revision `94bfe0a...`

You can keep it. The compatibility worker supports it and the 4.89 GB weights do not need to be downloaded again.

### Recommended revision for a fresh direct-Transformers download

Our app calls m0's `compute_score()` directly and does not use the Sentence-Transformers wrapper. For a new installation, prefer Jina's parent revision immediately before the Sentence-Transformers v5.4 integration:

```text
5b91da00be08ae2949e4e842b94d721c5c31eda3
```

The later `94bfe0a...` commit changed code/configuration files for Sentence Transformers but did not change `model.safetensors`, so the ranking weights are the same.

```bash
HF_XET_HIGH_PERFORMANCE=1 \
HF_HUB_DOWNLOAD_TIMEOUT=1800 \
HF_HUB_ETAG_TIMEOUT=300 \
hf download jinaai/jina-reranker-m0 \
  --revision 5b91da00be08ae2949e4e842b94d721c5c31eda3 \
  --local-dir "/Volumes/vision/Downloads/codes_necessary/models/jina-reranker-m0"
```

## Create the isolated reranker environment

From the repository root, with or without the main `.venv` activated:

```bash
zsh scripts/setup_reranker_env.sh
```

That creates `.venv-reranker`, installs `requirements-reranker.txt`, and verifies:

```text
Transformers: 4.48.3
MPS available: True
```

Do **not** downgrade Transformers inside the main `.venv`.

## Checks

Keep your normal main app `.venv` active and run:

```bash
python scripts/check_reranker_files.py
python scripts/check_reranker.py
```

`check_reranker_files.py` reports whether the local `modeling.py` is the Sentence-Transformers-era source that needs the compatibility bridge or the earlier direct-Transformers source.

`check_reranker.py` launches the isolated worker automatically. A healthy run must report values equivalent to:

```text
transformers_version: 4.48.3
architecture: qwen2_vl_pre_refactor
mm_token_type_compat_applied: True
checkpoint_loading_validated: True
```

and the obvious relevant text must score above the unrelated control.

The worker refuses to score if:

- the loaded model exposes the incompatible `model.language_model` layout;
- expected `model.layers.*` or ranking-head parameters are absent;
- any checkpoint weight is missing or mismatched;
- any unexpected checkpoint key other than the intentionally unused `lm_head.weight` appears.

This prevents silently reranking with randomly initialized layers.

## Important configuration

```text
RERANKER_ENABLED=true
RERANKER_DEVICE=auto
RERANKER_PYTHON_PATH=.venv-reranker/bin/python
RERANKER_WORKER_TIMEOUT_SEC=900
RERANKER_CANDIDATE_LIMIT=64
RERANKER_MAX_CANDIDATES=200
RERANKER_TEXT_BATCH_SIZE=4
RERANKER_IMAGE_BATCH_SIZE=1
RERANKER_TEXT_MAX_LENGTH=3072
RERANKER_IMAGE_MAX_LENGTH=4096
RERANKER_QUERY_MAX_LENGTH=512
RERANKER_ATTN_IMPLEMENTATION=eager
```

`auto` selects MPS when the reranker worker's PyTorch reports Apple-Silicon MPS support; otherwise it uses CPU.

## Search behavior

The UI intentionally displays two lists:

- **Semantic vector** — the same top-K dense results returned by Jina-v5/Weaviate before m0 was added.
- **m0 reranker** — final top-K after m0 scores a wider candidate pool.

For `All modalities`, the stage-one candidate pool seeds video, transcript, image, and text candidates separately, then fills unused capacity with a global dense query. This prevents one modality from crowding every other modality out before m0 can inspect them.

## Video reranking

m0 has text/image document modes, not a native video document mode. New video ingestions therefore save a 2x2 contact sheet from four positions within each video window under:

```text
data/assets/<asset_id>/rerank/<chunk_index>.jpg
```

Existing video assets remain rerankable without re-ingestion: if a contact sheet does not exist, the reranker falls back to the existing midpoint thumbnail.
