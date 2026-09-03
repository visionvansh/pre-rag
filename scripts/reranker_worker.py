#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

EXPECTED_TRANSFORMERS = "4.48.3"
MODEL_PATH = Path(os.environ.get("JINA_RERANKER_PATH", "")).expanduser().resolve()
DEVICE_REQUEST = os.environ.get("RERANKER_DEVICE", "auto").strip().lower()
ATTENTION = os.environ.get("RERANKER_ATTN_IMPLEMENTATION", "eager").strip()
TEXT_BATCH = max(1, int(os.environ.get("RERANKER_TEXT_BATCH_SIZE", "4")))
IMAGE_BATCH = max(1, int(os.environ.get("RERANKER_IMAGE_BATCH_SIZE", "1")))
TEXT_MAX = max(512, int(os.environ.get("RERANKER_TEXT_MAX_LENGTH", "3072")))
IMAGE_MAX = max(512, int(os.environ.get("RERANKER_IMAGE_MAX_LENGTH", "4096")))
QUERY_MAX = max(64, int(os.environ.get("RERANKER_QUERY_MAX_LENGTH", "512")))

_model = None
_torch = None
_device = None
_transformers_version = None


def _log(message: str) -> None:
    print(f"[reranker-worker] {message}", file=sys.stderr, flush=True)


def _validate_environment():
    global _torch, _transformers_version
    import torch
    import transformers

    _torch = torch
    _transformers_version = transformers.__version__
    if _transformers_version != EXPECTED_TRANSFORMERS:
        raise RuntimeError(
            "jina-reranker-m0 must run in the dedicated compatibility environment. "
            f"Expected transformers=={EXPECTED_TRANSFORMERS}, got {_transformers_version}. "
            "Run ./scripts/setup_reranker_env.sh and keep the main .venv unchanged."
        )
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Jina reranker path not found: {MODEL_PATH}")
    for name in ("config.json", "model.safetensors", "modeling.py"):
        path = MODEL_PATH / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing reranker checkpoint file: {path}")
    return torch


def _resolve_device(torch) -> str:
    if DEVICE_REQUEST == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if DEVICE_REQUEST == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("RERANKER_DEVICE=mps requested but PyTorch MPS is unavailable")
        return "mps"
    if DEVICE_REQUEST == "cpu":
        return "cpu"
    raise ValueError("RERANKER_DEVICE must be one of: auto, mps, cpu")


def _architecture_sanity(model) -> None:
    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("Loaded reranker has no .model module")
    if hasattr(base, "language_model"):
        raise RuntimeError(
            "Incompatible Qwen2-VL architecture detected: model.language_model exists. "
            "This is the newer Transformers layout and does not match m0's checkpoint keys."
        )
    if not hasattr(base, "layers") or not hasattr(base, "embed_tokens"):
        raise RuntimeError(
            "Unexpected Qwen2-VL architecture: expected model.layers and model.embed_tokens."
        )
    if not hasattr(model, "score"):
        raise RuntimeError("Loaded model is missing the Jina ranking head (.score)")

    names = dict(model.named_parameters())
    representative = "model.layers.0.self_attn.q_proj.weight"
    if representative not in names:
        raise RuntimeError(
            f"Expected checkpoint parameter layout is absent ({representative}). "
            "Refusing to rerank with partially/randomly initialized weights."
        )


def _load():
    global _model, _device
    if _model is not None:
        return _model

    torch = _validate_environment()
    from transformers import AutoModel

    _device = _resolve_device(torch)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": "auto",
    }
    if ATTENTION:
        kwargs["attn_implementation"] = ATTENTION

    _log(
        f"loading {MODEL_PATH} with transformers={_transformers_version}, "
        f"torch={torch.__version__}, device={_device}"
    )
    with contextlib.redirect_stdout(sys.stderr):
        model = AutoModel.from_pretrained(str(MODEL_PATH), **kwargs)
        _architecture_sanity(model)
        model.eval()
        model.to(_device)

    _model = model
    _log("model loaded with compatible pre-refactor Qwen2-VL parameter layout")
    return model


def _status(load: bool = False) -> dict[str, Any]:
    if load:
        _load()
    else:
        _validate_environment()
    return {
        "loaded": _model is not None,
        "device": _device or _resolve_device(_torch),
        "transformers_version": _transformers_version,
        "torch_version": _torch.__version__,
        "expected_transformers": EXPECTED_TRANSFORMERS,
        "model_path": str(MODEL_PATH),
        "architecture": "qwen2_vl_pre_refactor",
    }


def _score(query: Any, documents: list[Any], query_type: str, doc_type: str) -> list[float]:
    if query_type not in {"text", "image"} or doc_type not in {"text", "image"}:
        raise ValueError("query_type/doc_type must each be text or image")
    if not documents:
        return []

    model = _load()
    batch_size = IMAGE_BATCH if (query_type == "image" or doc_type == "image") else TEXT_BATCH
    max_length = IMAGE_MAX if (query_type == "image" or doc_type == "image") else TEXT_MAX
    max_query_length = min(QUERY_MAX, max_length // 2)
    pairs = [[query, document] for document in documents]

    with _torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
        raw = model.compute_score(
            pairs,
            batch_size=batch_size,
            max_length=max_length,
            max_query_length=max_query_length,
            query_type=query_type,
            doc_type=doc_type,
            normalize_scores=True,
            show_progress=False,
        )
    if len(documents) == 1 and isinstance(raw, (int, float)):
        raw = [raw]
    scores = [float(value) for value in raw]
    if len(scores) != len(documents):
        raise RuntimeError(f"m0 returned {len(scores)} scores for {len(documents)} documents")
    return scores


def _preflight() -> dict[str, Any]:
    from PIL import Image

    text_scores = _score(
        "red sports car",
        [
            "A bright red sports car is parked beside the road.",
            "This document explains how to bake sourdough bread.",
        ],
        "text",
        "text",
    )
    red = Image.new("RGB", (384, 384), (220, 30, 30))
    blue = Image.new("RGB", (384, 384), (25, 70, 200))
    image_scores = _score("a mostly red image", [red, blue], "text", "image")
    return {
        "status": _status(load=True),
        "text_scores": text_scores,
        "image_scores": image_scores,
        "text_relevant_ranked_higher": bool(text_scores[0] > text_scores[1]),
    }


def _handle(request: dict[str, Any]) -> Any:
    op = request.get("op")
    if op == "status":
        return _status(load=False)
    if op == "preflight":
        return _preflight()
    if op == "score":
        return {
            "scores": _score(
                request.get("query"),
                list(request.get("documents") or []),
                str(request.get("query_type")),
                str(request.get("doc_type")),
            )
        }
    raise ValueError(f"Unsupported reranker worker operation: {op}")


def main() -> None:
    _log("worker started; stdout reserved for JSON RPC")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            result = _handle(request)
            response = {"id": request_id, "ok": True, "result": result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "id": request_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
