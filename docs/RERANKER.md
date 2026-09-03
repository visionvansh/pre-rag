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

## Local model path

Default:

```text
/Volumes/vision/Downloads/codes_necessary/models/jina-reranker-m0
```

## Recommended pinned download

```bash
HF_XET_HIGH_PERFORMANCE=1 \
HF_HUB_DOWNLOAD_TIMEOUT=1800 \
HF_HUB_ETAG_TIMEOUT=300 \
hf download jinaai/jina-reranker-m0 \
  --revision 94bfe0aeb2d4dd7978362699cddd5893d4e0adc8 \
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

`check_reranker.py` launches the isolated worker automatically. A healthy run must report:

```text
transformers_version: 4.48.3
architecture: qwen2_vl_pre_refactor
text_relevant_ranked_higher: True
```

The worker also refuses to score if the loaded model exposes the incompatible `model.language_model` layout or if the expected `model.layers.0.self_attn.q_proj.weight` parameter is absent. This prevents silently reranking with randomly initialized layers.

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
