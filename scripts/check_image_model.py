from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.jina_mlx import embedder


def main() -> None:
    try:
        print("Loading Jina MLX and running a synthetic image preflight...")
        result = embedder.image_preflight()
        print("Embedding dimension:", result["embedding_dim"])
        print("Compatibility:", result["compatibility"])
        if result["embedding_dim"] != 1024:
            raise RuntimeError(
                f"Expected 1024 dimensions, got {result['embedding_dim']}"
            )
        print("Jina MLX image check OK")
    except Exception:
        print("\nJina MLX image check FAILED. Full traceback:\n")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
