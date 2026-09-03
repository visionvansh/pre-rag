from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.weaviate_store import store


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    try:
        client = store.connect()
        print("Weaviate ready:", client.is_ready())
        collection = store.ensure_collection()
        print("Collection ready:", collection.name)
        print("Weaviate check OK")
    except Exception:
        print("\nWeaviate check FAILED. Full traceback:\n")
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        store.close()


if __name__ == "__main__":
    main()
