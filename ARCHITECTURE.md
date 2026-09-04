# Architecture

## One semantic space

`jina-embeddings-v5-omni-small-retrieval-mlx` produces 1024-d retrieval embeddings for text, images and video in a shared vector space.

```text
Text query ───────┐
Image query ──────┼──> Jina 1024-d query vector
                  │
                  ▼
             Weaviate HNSW
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
 transcript     video        image
 vectors        vectors      vectors
```

## Video + transcript asset

```text
source video
 ├─ fixed 10-second windows
 │   └─ up to 32 frames
 │       └─ Jina encode_video()
 │
 └─ existing ASR JSON
     └─ words[] + utterances[].word_range
         └─ RecursiveCharacterTextSplitter
             └─ 800 Jina tokens / 120 overlap
                 └─ Jina encode_text("Document: ...")
```

Every object shares an `asset_id`. Video and transcript results use timestamps for synchronization.

## Image collection asset

```text
image folder / browser multi-upload
        │
        ├─ image 0 ─> Jina encode_image() ─> Weaviate
        ├─ image 1 ─> Jina encode_image() ─> Weaviate
        └─ image N ─> Jina encode_image() ─> Weaviate
```

One image equals one retrieval object.

The official Jina visual prompt is used:

```text
<|vision_start|><|image_pad|><|vision_end|>
```

## Persistent asset library

`data/assets.json` stores local asset metadata such as:
- asset name/type
- original video/transcript paths
- image paths
- counts
- latest successful ingestion timestamp

The webpage loads `/api/assets` on startup, so a restart no longer loses the active-index workflow.

Weaviate stores vectors. The registry stores local source-media information required for preview/playback.

## Search modes

### Text query

```text
"Query: " + user text
        ↓
Jina encode_text()
        ↓
near_vector
```

### Image query

```text
uploaded query image
        ↓
Jina encode_image()
        ↓
near_vector
```

Either query type can search:
- all modalities
- only transcript
- only video
- only image

and can be restricted to one asset or run globally.

## Failure isolation

Before video ingestion:
- synthetic video preflight

Before image ingestion:
- synthetic image preflight

Both exercise Transformers processing + Jina's MLX vision tower before replacing an existing asset index.

## Source-media lifecycle

Removing an asset from the UI:
- deletes its Weaviate objects
- deletes generated thumbnails
- removes its registry entry
- **does not delete original source files**

This prevents an indexing action from accidentally deleting user media.

---

## Parallel Jina v4 late-interaction lab

The existing architecture above remains available at `/`. A separate `/late-interaction` application tests Jina Embeddings v4 without changing `MediaChunk` or the v5 registry.

```text
text/image
    ↓
Jina Embeddings v4 retrieval adapter
ONE Qwen2.5-VL backbone forward
    │
    ├─ single_vec_emb [2048]
    │       ↓
    │   dense_v4 named vector
    │       ↓
    │   ordinary HNSW diagnostic search
    │
    └─ multi_vec_emb [N,128]
            ↓
        remove padding positions
            ↓
        late_v4 named multi-vector
            ↓
        native Weaviate MaxSim retrieval
            ↓
        optional jina-reranker-m0
```

The late-interaction lab supports only `text` and `image`. It stores data in the separate `JinaV4LateChunk` collection and local metadata under `data/late_interaction/`.

Search results are intentionally separated into three panels:

1. Jina-v4 dense result order.
2. Jina-v4 native late-interaction result order.
3. Optional m0 order, with m0 score as the only Stage-3 ranking authority.

See `docs/LATE_INTERACTION.md` for the pinned model revision, MPS environment, preflight and model download instructions.
