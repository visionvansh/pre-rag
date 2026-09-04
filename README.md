# pre-rag — local multimodal retrieval + m0 reranking

Apple-Silicon application for indexing and searching:

- video + existing ASR transcript
- image collections
- text collections

Stage 1 uses `jina-embeddings-v5-omni-small-retrieval-mlx` with a self-provided 1024-d Weaviate vector index. Stage 2 optionally uses `jina-reranker-m0` to rerank a wider candidate pool.

A parallel experiment is available at `/late-interaction`. It leaves the existing v5 pipeline untouched and adds Jina Embeddings v4 text/image retrieval with both its 2048-d dense output and its N×128 trained late-interaction output from the same backbone forward. See [`docs/LATE_INTERACTION.md`](docs/LATE_INTERACTION.md).

## Ranking policy

The existing v5 ranking policy is intentionally strict:

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

## Late-interaction experiment

```text
Jina v4 one forward
├─ single_vec_emb: 2048-d → Weaviate dense_v4
└─ multi_vec_emb: N×128   → Weaviate late_v4 native multi-vector search
                                      ↓
                               optional m0
```

The new lab supports only **text and images**. It uses a separate `JinaV4LateChunk` collection and `data/late_interaction/` registry so deletion/reingestion cannot modify the existing `MediaChunk` data.

Open:

```text
http://127.0.0.1:8000/late-interaction
```

Setup:

```bash
zsh scripts/setup_jina_v4_env.sh
python scripts/check_jina_v4_files.py
python scripts/check_jina_v4.py
```

Download the pinned Jina-v4 snapshot:

```bash
HF_XET_HIGH_PERFORMANCE=1 \
HF_HUB_DOWNLOAD_TIMEOUT=1800 \
HF_HUB_ETAG_TIMEOUT=300 \
hf download jinaai/jina-embeddings-v4 \
  --revision 853c867b65b749f3c3c72a06868140d842e04f06 \
  --local-dir "/Volumes/vision/Downloads/codes_necessary/models/jina-embeddings-v4"
```

## Ingestion

### Video + transcript
- Adjustable video windows: 1–120 seconds, no overlap.
- Up to 32 sampled frames/window.
- New ingestions create a 4-scene contact sheet/window for visual reranking.
- Transcript chunks use LangChain `RecursiveCharacterTextSplitter`, actual local Jina token counts, 800-token baseline and 120-token overlap.
- Existing ASR word timestamps remain the source of retrieval timestamps.

### Images
- One Jina vector/image in the v5 lab.
- Mac-path mode accepts one file or recursive directory.
- Browser upload accepts multiple images.

### Text
- One file or recursive directory in Mac-path mode; multiple browser uploads are supported.
- Files are chunked independently with the same Jina-token recursive baseline.
- Indexed as `modality="text"`.

## Persistent asset library

`data/assets.json` persists videos, images and text collections across app restarts. Existing vectors remain searchable even when an original source file later moves; source playback/full preview may then be unavailable.

The late-interaction lab uses its own `data/late_interaction/assets.json`.

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
python scripts/check_jina_v4_files.py
python scripts/check_jina_v4.py
```

## Start

```bash
./run.sh
```

Open:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/late-interaction
```

## Search comparison

The existing search UI provides dense-v5 vs m0 comparison.

The late-interaction UI provides three independent lists:

1. **Jina v4 Dense** — direct 2048-d search.
2. **Jina v4 Late Interaction** — native Weaviate multi-vector search over N×128 representations.
3. **m0 Final** — optional reranking of the late-interaction candidate pool, ordered by m0 only.

## Weaviate

The application computes embeddings locally; Weaviate does no vectorization.

```text
REST  127.0.0.1:8080
gRPC  127.0.0.1:50051
```

Collections:

```text
MediaChunk       existing Jina-v5 lab
JinaV4LateChunk  parallel late-interaction lab
```

## Tests

```bash
python -m compileall -q app scripts tests
pytest -q
```
