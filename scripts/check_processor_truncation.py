from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.jina_mlx import embedder


def main() -> None:
    try:
        print("Running high-resolution Jina image processor regression check...")
        result = embedder.image_preflight()
        print("Embedding dimension:", result["embedding_dim"])
        print("Test image size:", result.get("test_image_size"))
        print("Processor context ceiling:", result.get("processor_context_limit"))
        if result["embedding_dim"] != 1024:
            raise RuntimeError(
                f"Expected 1024 dimensions, got {result['embedding_dim']}"
            )
        print("Multimodal processor truncation check OK")
    except Exception:
        print("\nMultimodal processor truncation check FAILED. Full traceback:\n")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
