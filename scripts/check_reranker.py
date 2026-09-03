from pathlib import Path
import platform
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.jina_reranker import reranker


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Architecture: {platform.machine()}")
    print("Reranker status before load:", reranker.status())
    try:
        result = reranker.preflight()
        print("\nReranker status after load:", result["status"])
        print("Text relevance scores:", result["text_scores"])
        print("Visual relevance scores:", result["image_scores"])
        if not result["text_relevant_ranked_higher"]:
            raise RuntimeError(
                "Reranker loaded, but the obvious relevant text did not outrank the unrelated control."
            )
        print("\njina-reranker-m0 preflight OK")
    except Exception:
        print("\njina-reranker-m0 preflight FAILED. Full traceback:\n")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
