from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.late_interaction.jina_v4_client import embedder

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
        raise RuntimeError("Unexpected Jina v4 output dimensions")
    if not result["text"]["relevant_ranked_higher"]:
        raise RuntimeError("Late interaction failed the obvious text relevance sanity check")
    if not result["image"]["red_ranked_higher"]:
        raise RuntimeError("Text→image late interaction failed the synthetic color sanity check")
    print("\nJina v4 dense + late-interaction MPS preflight OK")
except Exception:
    print("\nJina v4 preflight FAILED:\n")
    traceback.print_exc()
    raise SystemExit(1)
finally:
    embedder.close()
