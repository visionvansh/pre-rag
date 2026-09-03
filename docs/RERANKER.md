# jina-reranker-m0 integration

`pre-rag` uses a strict two-stage retrieval policy:

1. `jina-embeddings-v5-omni-small-retrieval-mlx` + Weaviate retrieve candidates.
2. `jina-reranker-m0` scores the actual candidate contents and decides the final order.

There is **no weighted blend** between vector distance and m0 relevance. Dense rank is retained only for diagnostics and deterministic tie-breaking when m0 scores are equal.

## Local model path

Default:

```text
/Volumes/vision/Downloads/codes_necessary/models/jina-reranker-m0
```

The app lazy-loads m0 only when a search requests reranking.

## Recommended pinned download

The model weights are about 4.89 GB. Use large network timeouts so a slow Xet/LFS transfer does not fail after 10 seconds:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=1800 \
HF_HUB_ETAG_TIMEOUT=300 \
hf download jinaai/jina-reranker-m0 \
  --revision 94bfe0aeb2d4dd7978362699cddd5893d4e0adc8 \
  --local-dir "/Volumes/vision/Downloads/codes_necessary/models/jina-reranker-m0"
```

Optional on a machine with ample RAM and a fast connection:

```bash
HF_XET_HIGH_PERFORMANCE=1 \
HF_HUB_DOWNLOAD_TIMEOUT=1800 \
HF_HUB_ETAG_TIMEOUT=300 \
hf download jinaai/jina-reranker-m0 \
  --revision 94bfe0aeb2d4dd7978362699cddd5893d4e0adc8 \
  --local-dir "/Volumes/vision/Downloads/codes_necessary/models/jina-reranker-m0"
```

## Checks

```bash
python scripts/check_reranker_files.py
python scripts/check_reranker.py
```

`check_reranker.py` exercises text-to-text and text-to-image scoring on the selected local device.

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

## Important configuration

```text
RERANKER_ENABLED=true
RERANKER_DEVICE=auto
RERANKER_CANDIDATE_LIMIT=64
RERANKER_MAX_CANDIDATES=200
RERANKER_TEXT_BATCH_SIZE=4
RERANKER_IMAGE_BATCH_SIZE=1
RERANKER_TEXT_MAX_LENGTH=3072
RERANKER_IMAGE_MAX_LENGTH=4096
RERANKER_QUERY_MAX_LENGTH=512
RERANKER_ATTN_IMPLEMENTATION=eager
```

`auto` selects MPS when PyTorch reports Apple-Silicon MPS support; otherwise it uses CPU. The model is lazy-loaded and protected by an inference lock so multiple requests do not concurrently run 2.4B-parameter reranking passes.
