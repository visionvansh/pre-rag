# pre-rag — local multimodal retrieval + m0 reranking

Apple-Silicon application for indexing and searching:

- video + existing ASR transcript
- image collections
- text collections

Stage 1 uses `jina-embeddings-v5-omni-small-retrieval-mlx` with a self-provided 1024-d Weaviate vector index. Stage 2 optionally uses `jina-reranker-m0` to rerank a wider candidate pool.

## Ranking policy

The ranking policy is intentionally strict:

```text
query
  ↓
Jina v5 Omni embedding
  ↓
Weaviate candidate retrieval
  ↓
jina-reranker-m0 relevance score
  ↓
final order
```

Vector distance is **not blended** with the m0 relevance score. The webpage shows both result lists so dense and reranked behavior can be compared directly.

See [`docs/RERANKER.md`](docs/RERANKER.md) for model download, configuration and reranker design.

## Ingestion

### Video + transcript
- Adjustable video windows: 1–120 seconds, no overlap.
- Up to 32 sampled frames/window.
- New ingestions create a 4-scene contact sheet/window for visual reranking.
- Transcript chunks use LangChain `RecursiveCharacterTextSplitter`, actual local Jina token counts, 800-token baseline and 120-token overlap.
- Existing ASR word timestamps remain the source of retrieval timestamps.

### Images
- One Jina vector/image.
- Mac-path mode accepts one file or recursive directory.
- Browser upload accepts multiple images.

### Text
- One file or recursive directory in Mac-path mode; multiple browser uploads are supported.
- Files are chunked independently with the same Jina-token recursive baseline.
- Indexed as `modality="text"`.

## Persistent asset library

`data/assets.json` persists videos, images and text collections across app restarts. Existing vectors remain searchable even when an original source file later moves; source playback/full preview may then be unavailable.

## Update/install

```bash
cd "/Volumes/vision/Downloads/codes_necessary/pre-rag"
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
```

If `.venv` does not exist:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## Health checks

```bash
python scripts/check_weaviate.py
python scripts/check_model.py
python scripts/check_image_model.py
python scripts/check_video_model.py
python scripts/check_reranker_files.py
python scripts/check_reranker.py
```

The first four validate Weaviate/Jina-v5. The last two validate the local m0 files and actual text/vision reranking path.

## Start

```bash
./run.sh
```

Open:

```text
http://127.0.0.1:8000
```

## Search comparison

The search UI provides:

- query type: text or image
- result modality: all / text / image / transcript / video
- final result count
- m0 candidate-pool size up to 200
- reranker on/off comparison

Every search displays:

1. **Semantic vector** — unchanged dense baseline from Jina-v5 + Weaviate.
2. **m0 reranker** — final ranking by `rerank_score` only.

Reranked cards retain `vector_distance`, candidate dense rank, `rerank_score` and final rank for debugging.

## Weaviate

The application computes all embeddings locally; Weaviate does no vectorization.

```text
REST  127.0.0.1:8080
gRPC  127.0.0.1:50051
```

No Weaviate schema migration is required for the reranker.

## Tests

```bash
python -m compileall -q app scripts tests
pytest -q
```
