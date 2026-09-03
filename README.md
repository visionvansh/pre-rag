# pre-rag — Jina v5 Omni MLX multimodal retrieval lab

Local Apple-Silicon application for indexing and searching:

- **Video + existing ASR transcript**
- **Image collections**

with `jina-embeddings-v5-omni-small-retrieval-mlx` and a local Weaviate container.

## Current ingestion design

### Video + transcript
- Video chunks: **fixed 10 seconds**, no overlap.
- Up to **32 sampled frames** per 10-second video chunk.
- Transcript: LangChain `RecursiveCharacterTextSplitter`.
- Transcript length function: the **local Jina tokenizer**.
- Baseline transcript size: **800 Jina tokens**, **120-token overlap**.
- Existing word timestamps / `utterances[].word_range` are preserved for retrieval timestamps.

### Images
- One Jina embedding per image.
- Browser upload accepts many images at once.
- Mac-path mode accepts either one image or a directory; directories are scanned recursively.
- Generated thumbnails are stored under `data/assets/<asset_id>/thumbs`.

### Shared search space
All video, transcript and image embeddings are 1024-d Jina retrieval vectors stored in the same self-provided Weaviate vector space.

That enables:
- text → transcript
- text → video
- text → image
- image → image
- image → video/transcript
- global retrieval across every indexed asset

## Persistent asset library

The webpage now loads `data/assets.json` on startup and shows previously indexed assets in the **Indexed asset library** dropdown.

This fixes the old behavior where an asset could only become active immediately after a fresh ingestion job.

Selecting an existing asset enables search immediately. Selecting **All indexed assets** performs global retrieval.

The UI also supports:
- refresh asset list
- remove an asset from Weaviate without deleting original source files
- video timestamp seeking from retrieved video/transcript results
- image preview/gallery from retrieved image results
- text-query or image-query mode
- modality filter and result-count selector

## Jina MLX compatibility fix

`app/jina_compat.py` preserves the earlier MLX scalar/int compatibility patch for the current vision tower:

```python
mx.repeat(seq_len_i, int(grid_thw[i, 0].item()))
```

The downloaded Hugging Face checkpoint is not changed on disk.

## Install / update

From your repo:

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

Default Jina model path:

```text
/Volumes/vision/Downloads/codes_necessary/models/jina-embeddings-v5-omni-small-retrieval-mlx
```

## Health checks

Weaviate:

```bash
python scripts/check_weaviate.py
```

Text:

```bash
python scripts/check_model.py
```

Image vision path:

```bash
python scripts/check_image_model.py
```

Video vision path:

```bash
python scripts/check_video_model.py
```

Run all four before a large ingestion when changing MLX / Transformers versions.

## Start the app

```bash
./run.sh
```

Open:

```text
http://127.0.0.1:8000
```

## Image ingestion

Choose **Image collection** from the ingestion workflow dropdown.

### Mac paths
Provide:
- one image path, or
- a directory containing images.

Supported by this app's PIL ingestion layer:

```text
.jpg .jpeg .png .webp .bmp .tif .tiff .gif .avif
```

### Browser upload
Select multiple images in the file picker and click **Upload & ingest**.

Each image is indexed as:

```text
modality = image
chunk_id = image_00000 ...
embedding_dim = 1024
```

## Searching

The asset library dropdown controls search scope.

- Select one asset to restrict retrieval to it.
- Select **All indexed assets** for global retrieval.

Then choose:

```text
Query type:
- Text query
- Image query

Result modality:
- All
- Images
- Transcript
- Video
```

Image query mode sends the query image through Jina `encode_image()` and performs vector search in exactly the same Weaviate space.

## Existing assets

`git pull` does not remove your ignored `data/` folder, so existing registry entries, uploaded media and generated thumbnails remain in place.

If an original local-path video/image has moved, its vectors remain searchable. Playback/full-image viewing may be unavailable until the source is restored to the recorded path.

## Weaviate

The application generates all embeddings locally. Weaviate does no vectorization.

Expected local ports:

```text
REST  127.0.0.1:8080
gRPC  127.0.0.1:50051
```

## Tests

```bash
python -m compileall -q app scripts tests
pytest -q
```
