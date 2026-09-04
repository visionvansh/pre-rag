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
HF_XET_HIGH_PERFORMANCE=1 \
HF_HUB_DOWNLOAD_TIMEOUT=1800 \
HF_HUB_ETAG_TIMEOUT=300 \
hf download jinaai/jina-embeddings-v4 \
  --revision 853c867b65b749f3c3c72a06868140d842e04f06 \
  --local-dir "/Volumes/vision/Downloads/codes_necessary/models/jina-embeddings-v4"
```

The snapshot is roughly 7.9 GB. Do not use the GGUF variant for this lab: the experiment requires the trained 128-d multi-vector projection.

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
- 2048-d dense output
- N×128 multi-vector output
- obvious relevant text beats an unrelated text under reference MaxSim
- a synthetic red image beats a blue control for a text query about a red image

If this preflight fails, do not trust ingestion/search results.

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
