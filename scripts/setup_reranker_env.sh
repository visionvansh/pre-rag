#!/bin/zsh
set -e
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${RERANKER_VENV:-.venv-reranker}"

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements-reranker.txt

python - <<'PY'
import platform
import torch
import transformers
print("Reranker environment ready")
print("Python architecture:", platform.machine())
print("Transformers:", transformers.__version__)
print("Torch:", torch.__version__)
print("MPS available:", bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
if transformers.__version__ != "4.48.3":
    raise SystemExit("Expected transformers==4.48.3")
PY
