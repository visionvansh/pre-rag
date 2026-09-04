from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "scripts" / "jina_v4_worker.py"


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("jina_v4_worker_precision_test", WORKER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTorch:
    float32 = object()
    bfloat16 = object()


def test_auto_cpu_uses_full_float32(monkeypatch):
    worker = _load_worker_module()
    monkeypatch.setattr(worker, "DTYPE_REQUEST", "auto")
    dtype, name = worker._resolve_dtype(_FakeTorch(), "cpu")
    assert dtype is _FakeTorch.float32
    assert name == "float32"


def test_explicit_bfloat16_is_allowed(monkeypatch):
    worker = _load_worker_module()
    monkeypatch.setattr(worker, "DTYPE_REQUEST", "bfloat16")
    dtype, name = worker._resolve_dtype(_FakeTorch(), "cpu")
    assert dtype is _FakeTorch.bfloat16
    assert name == "bfloat16"


@pytest.mark.parametrize("value", ["float16", "fp16", "half", "torch.float16"])
def test_fp16_is_rejected(monkeypatch, value):
    worker = _load_worker_module()
    monkeypatch.setattr(worker, "DTYPE_REQUEST", value)
    with pytest.raises(ValueError, match="forbidden"):
        worker._resolve_dtype(_FakeTorch(), "cpu")
