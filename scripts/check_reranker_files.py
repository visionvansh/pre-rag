from pathlib import Path
import hashlib
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings


EXPECTED_WEIGHT_SHA256 = "1d0a7b5fd0966512850481633159f357450dc738665870c9ac4f2b2da252f5e2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = settings.jina_reranker_path
    print("Reranker path:", root)
    required = [root / "config.json", root / "modeling.py", root / "model.safetensors"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        for path in missing:
            print("MISSING:", path)
        raise SystemExit(1)
    weights = root / "model.safetensors"
    print("Weights size GB:", round(weights.stat().st_size / 1_000_000_000, 3))
    actual = sha256(weights)
    print("Weights SHA256:", actual)
    if actual != EXPECTED_WEIGHT_SHA256:
        print("WARNING: weight hash differs from the researched upstream m0 weight file.")
    else:
        print("Reranker files OK")


if __name__ == "__main__":
    main()
