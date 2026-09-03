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

    try:
        print("\nLoading Jina MLX and running a synthetic 4-frame video preflight...")
        result = embedder.video_preflight()
        print("Embedding dimension:", result["embedding_dim"])
        print("Compatibility:", result["compatibility"])
        if result["embedding_dim"] != 1024:
            raise RuntimeError(
                f"Expected 1024 dimensions, got {result['embedding_dim']}"
            )
        print("Jina MLX video check OK")
    except Exception:
        print("\nJina MLX video check FAILED. Full traceback:\n")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
