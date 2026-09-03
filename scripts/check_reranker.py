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
    print(f"Main app Python: {sys.executable}")
    print(f"Architecture: {platform.machine()}")
    print("Reranker status before worker start:", reranker.status())
    try:
        result = reranker.preflight()
        status = result["status"]
        print("\nReranker worker status:", status)
        print("Text relevance scores:", result["text_scores"])
        print("Visual relevance scores:", result["image_scores"])

        if status.get("transformers_version") != "4.48.3":
            raise RuntimeError(
                "Reranker worker is not using the required transformers==4.48.3 environment."
            )
        if status.get("architecture") != "qwen2_vl_pre_refactor":
            raise RuntimeError("Reranker worker loaded the wrong Qwen2-VL architecture.")
        if not status.get("checkpoint_loading_validated"):
            raise RuntimeError(
                "Reranker checkpoint loading diagnostics were not validated; refusing to trust scores."
            )
        if not result["text_relevant_ranked_higher"]:
            raise RuntimeError(
                "Reranker loaded, but the obvious relevant text did not outrank the unrelated control."
            )
        print("\njina-reranker-m0 isolated-worker preflight OK")
    except Exception:
        print("\njina-reranker-m0 preflight FAILED. Full traceback:\n")
        traceback.print_exc()
        print(
            "\nIf the error mentions the reranker Python environment, run:\n"
            "  zsh scripts/setup_reranker_env.sh\n"
            "Then rerun:\n"
            "  python scripts/check_reranker.py"
        )
        raise SystemExit(1)
    finally:
        reranker.close()


if __name__ == "__main__":
    main()
