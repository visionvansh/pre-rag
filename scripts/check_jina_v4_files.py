from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.config import settings

path = settings.jina_v4_model_path.expanduser()
required = [
    "config.json", "modeling_jina_embeddings_v4.py", "qwen2_5_vl.py",
    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    "model.safetensors.index.json", "tokenizer.json", "preprocessor_config.json",
]
print("Jina v4 path:", path)
print("Pinned revision:", settings.jina_v4_revision)
missing = [name for name in required if not (path / name).is_file()]
if missing:
    raise SystemExit("Missing Jina v4 files:\n- " + "\n- ".join(missing))
size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
print(f"Model directory size: {size / 1_000_000_000:.2f} GB")
config = json.loads((path / "config.json").read_text(encoding="utf-8"))
print("Architectures:", config.get("architectures"))
print("Jina v4 files OK")
