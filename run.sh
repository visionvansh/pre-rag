#!/bin/zsh
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
exec uvicorn app.main:app --host 127.0.0.1 --port 8000
