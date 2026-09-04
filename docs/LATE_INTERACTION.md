# Jina v4 late-interaction lab

This is an additive experiment. The existing `/` Jina-v5 Omni application and its `MediaChunk` collection remain unchanged.

## Architecture

```text
/                                  /late-interaction
Jina v5 Omni                       Jina Embeddings v4
MediaChunk                         JinaV4LateChunk
1024-d single vector               one backbone forward
                                   ├─ dense_v4: 2048-d
                                   └─ late_v4: N × 128-d
                                              ↓
                                   native Weaviate multi-vector search
                                              ↓
                                   optional jina-reranker-m0
```

Each Jina-v4 forward returns both `single_vec_emb` and `multi_vec_emb`; the app does not run the 3.8B/4B backbone twice just to obtain the two representations. Padding positions are removed from the stored multi-vector output.

The first comparison keeps the existing text chunk baseline: 800 Jina-v4 tokenizer tokens with 120-token overlap. That isolates the retrieval-architecture change from chunk-size changes.

## Model download

Pinned evaluation revision:

```text
853c867b65b749f3c3c72a06868140d842e04f06
```

Download to the default path used by `app/config.py`:

```bash
HF_HUB_DISABLE_XET=1 \
HF_HUB_DOWNLOAD_TIMEOUT=1800 \
HF_HUB_ETAG_TIMEOUT=300 \
hf download jinaai/jina-embeddings-v4 \
  --revision 853c867b65b749f3c3c72a06868140d842e04f06 \
  --local-dir "/Volumes/vision/Downloads/codes_necessary/models/jina-embeddings-v4"
```

The snapshot is roughly 7.9 GB. Do not use the GGUF or MLX-8bit variants for this lab: the experiment uses the official full checkpoint and its trained 128-d multi-vector projection.

## Precision policy: no FP16 downgrade

The official Jina v4 checkpoint is BF16. The worker now has a quality-first precision contract:

```text
MPS + auto  -> BF16
CPU + auto  -> FP32
explicit BF16 -> BF16
explicit FP32 -> FP32
FP16/float16 -> rejected
```

Configure it with:

```text
JINA_V4_DTYPE=auto
```

`auto` preserves the checkpoint's BF16 precision on Apple Silicon when the local MPS runtime passes a BF16 capability probe. If that probe fails, the worker falls back to full FP32, never FP16. No 4-bit/8-bit quantization is used by this runtime.

The model forward is checked for non-finite values **before** FP32 normalization and before anything reaches Weaviate. A numerical failure therefore names the model output, device, dtype, and image context rather than being silently repaired or inserted.

## Isolated Apple-Silicon environment

Keep the main `.venv` and `.venv-reranker` unchanged.

```bash
zsh scripts/setup_jina_v4_env.sh
```

The v4 worker uses MPS automatically when available and starts conservatively with eager attention. FlashAttention is not required on Apple Silicon.

## Preflight

```bash
python scripts/check_jina_v4_files.py
python scripts/check_jina_v4.py
```

The actual-model preflight requires:

- arm64/MPS-compatible model loading
- BF16 on supported MPS, or full FP32 fallback
- no FP16 runtime
- 2048-d dense output
- N×128 multi-vector output
- finite text and image inputs/outputs
- obvious relevant text beats an unrelated text under reference MaxSim
- synthetic text→image scoring is reported as a diagnostic only

If this preflight fails, do not trust ingestion/search results.

## Image-ingestion failure behavior

Image batches are treated transactionally at the asset level. If image `N` fails after earlier images were inserted, the job now rolls back:

- all Weaviate objects for that asset
- the late-interaction registry entry
- generated local thumbnails for that asset

The failure detail also names the exact image and dimensions that triggered the model/processor error. A failed batch should therefore not leave a half-indexed asset behind.

## Weaviate

The existing Docker service is reused. A separate collection is created:

```text
JinaV4LateChunk
```

Named vectors:

```text
dense_v4   regular self-provided HNSW vector, 2048-d
late_v4    self-provided Weaviate multi-vector, N × 128-d
```

A single object contains both representations. Native late-interaction queries send the raw query matrix to `near_vector(..., target_vector="late_v4")`.

No objects from `MediaChunk` are deleted or migrated.

## Search ordering

The page deliberately shows three independent lists:

1. **Dense** — direct `dense_v4` search.
2. **Late interaction** — direct native `late_v4` multi-vector search.
3. **m0** — optional reranking of the late-interaction candidate pool.

When m0 is enabled, m0 relevance alone determines the Stage-3 order. Dense/late distances are diagnostics only; there is no weighted blend.

When m0 is disabled, the final result list is exactly the late-interaction result list.

## URLs

```text
Existing lab:         http://127.0.0.1:8000/
Late-interaction lab: http://127.0.0.1:8000/late-interaction
```

## Data isolation

The LI experiment stores local metadata/generated files under:

```text
data/late_interaction/
```

Its delete route only touches that directory, its registry, and `JinaV4LateChunk`. Original source files are never deleted.

## Low-memory mode

If unified-memory pressure is high while running both v4 and m0, set:

```text
LI_LOW_MEMORY_MODE=true
```

Then the app releases the v4 worker immediately before the optional m0 stage. The next query reloads v4, trading latency for lower simultaneous model residency.

## Licensing note

`jina-embeddings-v4` uses the Qwen Research License inherited from Qwen2.5-VL. The public `jina-reranker-m0` weights also have non-commercial licensing constraints. Resolve licensing before using this exact model combination in a commercial product.
