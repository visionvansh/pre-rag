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
