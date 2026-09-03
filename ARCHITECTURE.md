# Architecture

## Inputs

The application accepts exactly two content inputs:

1. source video
2. an already-produced ASR JSON containing word timestamps/speakers and utterance word ranges

ASR generation is outside this repository.

## Preflight sequence

Before replacing any existing asset vectors, ingestion now performs:

1. validate paths and ASR JSON
2. connect to Weaviate / ensure schema
3. load Jina MLX text/model weights
4. run a **synthetic 4-frame video preflight** through `AutoProcessor` + `encode_video`
5. only then delete/rebuild vectors for a re-ingested asset

This ordering catches processor, torchvision, Jina model-code, and MLX vision-forward failures before transcript indexing starts.

## Jina upstream compatibility boundary

The current Jina MLX vision code calculates cumulative frame sequence lengths with a scalar MLX array passed as `mx.repeat(..., repeats=...)`. MLX documents `repeats` as an integer.

`app/jina_compat.py` therefore:

- reads the local checkpoint's `model.py`;
- fingerprints it with SHA-256;
- looks for one exact known buggy expression;
- replaces only that expression with an `.item()` → `int(...)` conversion;
- compiles the transformed source **in memory** under a private module name;
- never changes the downloaded checkpoint on disk;
- is idempotent and automatically no-ops if upstream already contains the corrected expression.

No broad monkey-patching of `mlx.core.repeat` is used.

## Transcript path

`words[]` is authoritative for time/speaker metadata. `utterances[].word_range` inserts paragraph boundaries into a normalized transcript. Each normalized word receives character offsets.

LangChain `RecursiveCharacterTextSplitter` uses:

```text
chunk_size    = 800 Jina tokens
chunk_overlap = 120 Jina tokens
```

with the local Jina tokenizer as `length_function`. Returned chunks are mapped from character ranges back to the underlying ASR word spans to produce precise `start_sec`, `end_sec`, word indices, and speaker IDs.

Transcript chunks are embedded as:

```text
Document: <chunk text>
```

## Video path

Video retrieval windows are fixed:

```text
0–10s
10–20s
20–30s
...
```

There is no overlap in the baseline.

Within each interval PyAV seeks/decodes and uniformly samples up to 32 frames. The frame count is forced even because Jina's Qwen3-VL vision path uses `temporal_patch_size=2` temporal pairs.

Video processor prompt follows the official MLX quickstart exactly:

```text
<|vision_start|><|video_pad|><|vision_end|>
```

The processor remains torch/torchvision-side; model inference is MLX-side.

## Weaviate

One collection: `MediaChunk`.

Each object receives one self-provided 1024-d normalized vector and metadata including:

- `asset_id`
- `chunk_id`
- `modality` (`video` or `transcript`)
- `chunk_index`
- `start_sec`, `end_sec`, `duration_sec`
- transcript text/speaker/token fields when applicable
- video frame count/thumbnail when applicable
- embedding model/dimension

Cosine HNSW is used. Weaviate does not vectorize data.

## Progress/UI

The backend reports stages and counters to the local webpage. `weaviate_objects` is updated incrementally during transcript and video insertion rather than only at final completion.

On success, search is enabled. A query is embedded as:

```text
Query: <user query>
```

and searched across the shared Jina vector space. The UI can filter to all/video/transcript. Clicking any result seeks the HTML5 player to `start_sec`.

## Known limits of this baseline

- one ingestion job drives MLX at a time
- no reranker yet
- no hybrid BM25 yet
- no answer-generating LLM yet
- no transactional/staging swap of an existing complete asset after a later-stage failure
- fixed 10-second video windows are intentional for this experiment
