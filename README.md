# pre-rag — Jina v5 Omni MLX video + transcript retrieval lab

Local Apple-Silicon test application for ingesting an existing timestamped ASR JSON plus a source video into one Weaviate collection using `jina-embeddings-v5-omni-small-retrieval-mlx`.

## Locked ingestion design

- **No ASR implementation in this repo.** You provide the video and your existing ASR JSON.
- Video retrieval chunks: **fixed 10 seconds**, no overlap.
- Up to **32 sampled frames** per 10-second chunk, normalized to an even frame count for Qwen3-VL temporal pairs.
- Transcript chunks: LangChain `RecursiveCharacterTextSplitter`.
- Transcript length function: the **local Jina tokenizer**.
- Baseline transcript chunking: **800 Jina tokens** with **120-token overlap**.
- Both modalities produce **1024-d** Jina embeddings and are stored as separate objects in one self-provided Weaviate vector space.
- Video/transcript synchronization is via `asset_id`, `start_sec`, and `end_sec`.

## Important Jina MLX compatibility fix

As of the current upstream `jina-embeddings-v5-omni-small-retrieval-mlx/model.py`, the vision forward contains:

```python
mx.repeat(seq_len_i, grid_thw[i, 0])
```

Recent MLX expects `repeats` to be a native Python `int`, while `grid_thw[i, 0]` is an MLX scalar array. This produces:

```text
TypeError: repeat(): incompatible function arguments
Invoked with types: mlx.core.array, mlx.core.array
```

`app/jina_compat.py` applies a narrow, in-memory, idempotent transformation to the checkpoint source at load time:

```python
mx.repeat(seq_len_i, int(grid_thw[i, 0].item()))
```

The downloaded Hugging Face model folder is **not modified**. If upstream Jina already contains the fix, the compatibility layer detects that and does nothing.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Default model path:

```text
/Volumes/vision/Downloads/codes_necessary/models/jina-embeddings-v5-omni-small-retrieval-mlx
```

Change `JINA_MODEL_PATH` in `.env` if needed.

## Weaviate

If you already have Weaviate running with host ports `8080` and `50051`, keep using it.

Or start the included pinned local configuration:

```bash
docker compose up -d
```

Check:

```bash
python scripts/check_weaviate.py
```

Expected:

```text
Weaviate ready: True
Collection ready: MediaChunk
Weaviate check OK
```

## Model checks — run both before ingestion

Text path:

```bash
python scripts/check_model.py
```

Video path (processor + vision tower + compatibility patch):

```bash
python scripts/check_video_model.py
```

The second check is specifically designed to catch video-only failures before any transcript vectors are written.

Expected final line:

```text
Jina MLX video check OK
```

The output also prints whether the upstream `mx.repeat` compatibility patch was applied.

## Run the webpage

```bash
./run.sh
```

Open:

```text
http://127.0.0.1:8000
```

Use either:

- **Mac path mode** for large source files on `/Volumes/...` (recommended), or
- browser upload mode.

The UI displays live stage/progress counters for transcript chunks, fixed 10-second video chunks, and Weaviate objects. After ingestion completes, semantic search is enabled; clicking a result seeks the source video to the result timestamp.

## ASR JSON contract

The parser consumes your existing top-level `words[]` objects:

```json
{"text":"example", "start":10.32, "end":10.72, "speaker":0}
```

and `utterances[]` objects:

```json
{"speaker":0, "start":0.16, "end":69.48, "text":"...", "word_range":[0,264]}
```

`words[]` are the timing source of truth. `utterances[].word_range` provides natural paragraph boundaries before recursive splitting. Each LangChain chunk is mapped back to exact first/last ASR words to recover timestamps and speaker IDs.

## Retrieval behavior

Text queries are encoded with Jina's retrieval-side `Query: ` prefix. Transcript chunks use `Document: `.

Video preprocessing follows Jina's official MLX video quickstart and uses only:

```text
<|vision_start|><|video_pad|><|vision_end|>
```

rather than adding a `Document:` text token to the visual input.

Weaviate does **no** embedding generation. It stores the externally generated normalized 1024-d Jina vectors using cosine HNSW.

## Tests

```bash
pytest -q
python -m compileall -q app scripts tests
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'app'`

The checked-in scripts add the repository root to `sys.path`; run them from the repo root:

```bash
python scripts/check_model.py
```

### `Qwen3VLVideoProcessor requires the Torchvision library`

```bash
python -m pip install torchvision
```

`torchvision` is included in `requirements.txt`.

### `mx.repeat ... mlx.core.array, mlx.core.array`

Pull the current repo code and run:

```bash
python scripts/check_video_model.py
```

Do **not** hand-edit the downloaded Jina `model.py`; the application compatibility layer handles this narrowly in memory.

See `ARCHITECTURE.md` for the pipeline details.
