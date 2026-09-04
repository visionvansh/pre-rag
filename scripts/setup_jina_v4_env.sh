#!/bin/zsh
set -e
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${JINA_V4_VENV:-.venv-jina-v4}"

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements-jina-v4.txt

python - <<'PY'
import platform
import torch
import transformers
print("Jina v4 environment ready")
print("Architecture:", platform.machine())
print("Transformers:", transformers.__version__)
print("Torch:", torch.__version__)
print("MPS available:", bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
PY
