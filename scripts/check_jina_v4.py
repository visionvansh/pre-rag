from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.late_interaction.jina_v4_client import embedder


def _require_health(label: str, health: dict) -> None:
    if int(health.get("dense_dim") or 0) != 2048:
        raise RuntimeError(f"{label}: expected 2048-d dense output, got {health.get('dense_dim')}")
    if int(health.get("late_dim") or 0) != 128:
        raise RuntimeError(f"{label}: expected 128-d late vectors, got {health.get('late_dim')}")
    if int(health.get("late_vector_count") or 0) <= 0:
        raise RuntimeError(f"{label}: no late-interaction vectors returned")
    if not health.get("dense_finite"):
        raise RuntimeError(f"{label}: dense output contains non-finite values")
    if not health.get("multi_finite"):
        raise RuntimeError(f"{label}: late output contains non-finite values")
    dtype = str(health.get("runtime_dtype") or "")
    if dtype and dtype not in {"bfloat16", "float32"}:
        raise RuntimeError(f"{label}: unsupported runtime dtype {dtype}; FP16 is not allowed")


print("Project root:", ROOT)
print("Main app Python:", sys.executable)
print("Architecture:", platform.machine())
print("Jina v4 status before worker:", embedder.status())
try:
    result = embedder.preflight()
    print("\nWorker status:", result["status"])
    print("Text preflight:", result["text"])
    print("Image preflight:", result["image"])

    status = result["status"]
    if status.get("dense_dim") != 2048 or status.get("late_dim") != 128:
        raise RuntimeError("Unexpected declared Jina v4 output dimensions")

    dtype = str(status.get("dtype") or "")
    if dtype not in {"bfloat16", "float32"}:
        raise RuntimeError(
            f"Jina v4 is running with unexpected dtype {dtype!r}; expected BF16 or full FP32"
        )
    if dtype == "float32":
        print("Precision note: running full FP32 (quality-preserving fallback, higher memory use).")
    else:
        print("Precision note: running native BF16 checkpoint precision on MPS.")

    _require_health("text query", result["text"]["query_health"])
    _require_health("text passage", result["text"]["passage_health"])
    _require_health("image query", result["image"]["query_health"])
    _require_health("image document", result["image"]["image_health"])

    if not result["text"]["relevant_ranked_higher"]:
        raise RuntimeError("Late interaction failed the obvious text relevance sanity check")

    # The red-vs-blue synthetic image comparison is reported for diagnostics only.
    # Solid-color squares are not a reliable semantic benchmark for a VLM retriever.
    print(
        "Synthetic text→image diagnostic (not a hard gate): red_ranked_higher =",
        result["image"]["red_ranked_higher"],
    )
    print("\nJina v4 dense + late-interaction precision preflight OK")
except Exception:
    print("\nJina v4 preflight FAILED:\n")
    traceback.print_exc()
    raise SystemExit(1)
finally:
    embedder.close()
