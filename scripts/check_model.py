from pathlib import Path
import platform
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.jina_mlx import embedder


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Architecture: {platform.machine()}")
    print(f"Jina model path: {settings.jina_model_path}")

    required = [
        settings.jina_model_path / "config.json",
        settings.jina_model_path / "model.py",
        settings.jina_model_path / "model.safetensors",
        settings.jina_model_path / "tokenizer.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("\nERROR: Required model files are missing:")
        for path in missing:
            print(" -", path)
        raise SystemExit(1)

    try:
        print("\nLoading Jina MLX text path...")
        embedder.load()
        print("Model loaded. Creating test query embedding...")
        vec = embedder.embed_query("test query about a video")

        print("Embedding dimension:", len(vec))
        print("First five values:", vec[:5])
        print("Compatibility:", embedder.compatibility_report())
        if len(vec) != 1024:
            raise RuntimeError(f"Expected 1024 dimensions, got {len(vec)}")
        print("Jina MLX text check OK")
    except Exception:
        print("\nJina MLX check FAILED. Full traceback:\n")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
