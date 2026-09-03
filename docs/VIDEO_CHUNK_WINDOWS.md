# Adjustable video window controls

The video + transcript workflow supports a per-ingestion video window size.

## Range

- Minimum: 1 second
- Maximum: 120 seconds
- UI step: 0.5 seconds
- Default: `VIDEO_CHUNK_SECONDS` (10 seconds unless overridden in `.env`)

The selected duration controls how many seconds one Jina video vector represents. It does **not** change the frame cap: each window still samples at most `VIDEO_MAX_FRAMES` frames (32 by default).

The chosen value is stored in `data/assets.json`, appears in the indexed asset library, and is used in progress labels. Re-ingesting the same source video with a new duration replaces the previous vectors for that asset instead of mixing two chunk grids.

Run:

```bash
python scripts/check_chunk_config.py
```

to verify the request validation before starting a full ingestion.
