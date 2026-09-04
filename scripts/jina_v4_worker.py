#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL_PATH = Path(os.environ.get("JINA_V4_MODEL_PATH", "")).expanduser().absolute()
DEVICE_REQUEST = os.environ.get("JINA_V4_DEVICE", "auto").strip().lower()
DTYPE_REQUEST = os.environ.get("JINA_V4_DTYPE", "auto").strip().lower()
ATTENTION = os.environ.get("JINA_V4_ATTN_IMPLEMENTATION", "eager").strip()
TEXT_MAX_LENGTH = max(128, int(os.environ.get("JINA_V4_TEXT_MAX_LENGTH", "8192")))
REVISION = os.environ.get("JINA_V4_REVISION", "")

_model = None
_processor = None
_torch = None
_device = None
_dtype = None
_dtype_name = None
_transformers_version = None


def _log(message: str) -> None:
    print(f"[jina-v4-worker] {message}", file=sys.stderr, flush=True)


def _resolve_device(torch) -> str:
    if DEVICE_REQUEST == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if DEVICE_REQUEST == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("JINA_V4_DEVICE=mps requested but PyTorch MPS is unavailable")
        return "mps"
    if DEVICE_REQUEST == "cpu":
        return "cpu"
    raise ValueError("JINA_V4_DEVICE must be one of: auto, mps, cpu")


def _probe_mps_bfloat16(torch) -> tuple[bool, str | None]:
    """Check that this PyTorch/macOS build can execute basic BF16 work on MPS."""
    try:
        probe = torch.ones((8,), device="mps", dtype=torch.bfloat16)
        value = (probe * probe).sum().item()
        del probe
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if not math.isfinite(float(value)):
            return False, "BF16 MPS probe returned a non-finite value"
        return True, None
    except Exception as exc:  # hardware/runtime capability probe
        return False, f"{type(exc).__name__}: {exc}"


def _resolve_dtype(torch, device: str):
    """Quality-first dtype policy: native BF16 on MPS, FP32 otherwise; never FP16."""
    request = DTYPE_REQUEST.replace("torch.", "")
    if request in {"float16", "fp16", "half"}:
        raise ValueError(
            "JINA_V4_DTYPE=float16/fp16 is intentionally forbidden. "
            "jina-embeddings-v4 is a BF16 model; use auto, bfloat16, or float32."
        )

    if request in {"float32", "fp32"}:
        return torch.float32, "float32"

    if request in {"bfloat16", "bf16"}:
        if device == "mps":
            ok, reason = _probe_mps_bfloat16(torch)
            if not ok:
                raise RuntimeError(
                    "JINA_V4_DTYPE=bfloat16 was requested but this MPS runtime failed "
                    f"the BF16 capability probe: {reason}"
                )
        return torch.bfloat16, "bfloat16"

    if request != "auto":
        raise ValueError("JINA_V4_DTYPE must be one of: auto, bfloat16, float32")

    if device == "mps":
        ok, reason = _probe_mps_bfloat16(torch)
        if ok:
            return torch.bfloat16, "bfloat16"
        # FP32 is slower and uses more memory, but it preserves numerical range and
        # precision. We explicitly never fall back to FP16.
        _log(f"BF16 is unavailable on this MPS runtime; using FP32 instead: {reason}")
        return torch.float32, "float32"

    return torch.float32, "float32"


def _core_model(model):
    # Current v4 can be returned through a PEFT wrapper. Attribute access on the
    # wrapper delegates to the underlying embedding model, so we keep the wrapper
    # when it exposes Jina's multi-vector projector; that preserves the retrieval
    # LoRA adapter path.
    if hasattr(model, "multi_vector_projector"):
        return model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "multi_vector_projector"):
        return inner
    return model


def _load():
    global _model, _processor, _torch, _device, _dtype, _dtype_name, _transformers_version
    if _model is not None:
        return _model

    import torch
    import transformers
    from transformers import AutoModel, AutoProcessor

    _torch = torch
    _transformers_version = transformers.__version__
    _device = _resolve_device(torch)
    _dtype, _dtype_name = _resolve_dtype(torch, _device)
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Jina v4 model path not found: {MODEL_PATH}")

    # Retrieval is selected at forward/encoding time in the pinned v4 snapshot.
    # Do not pass task="retrieval" to from_pretrained: the custom loader can leak
    # that keyword into JinaEmbeddingsV4Model.__init__, which does not accept it.
    # The checkpoint is BF16. On Apple Silicon we preserve BF16 rather than forcing
    # FP16; if BF16 cannot run, auto mode falls back to full FP32, never FP16.
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "dtype": _dtype,
    }
    if ATTENTION:
        kwargs["attn_implementation"] = ATTENTION

    _log(
        f"loading {MODEL_PATH} with transformers={_transformers_version}, "
        f"torch={torch.__version__}, device={_device}, dtype={_dtype_name}"
    )
    with contextlib.redirect_stdout(sys.stderr):
        try:
            model = AutoModel.from_pretrained(str(MODEL_PATH), **kwargs)
        except (TypeError, ValueError) as exc:
            if "attn" not in str(exc).lower() and "attention" not in str(exc).lower():
                raise
            kwargs.pop("attn_implementation", None)
            _log(f"attention override was rejected; retrying model load without it: {exc}")
            model = AutoModel.from_pretrained(str(MODEL_PATH), **kwargs)

        core = _core_model(model)
        processor = getattr(model, "processor", None) or getattr(core, "processor", None)
        if processor is None:
            processor = AutoProcessor.from_pretrained(
                str(MODEL_PATH), trust_remote_code=True, local_files_only=True
            )
        model.eval()
        model.to(_device)

    core = _core_model(model)
    if not hasattr(core, "multi_vector_projector"):
        raise RuntimeError("Loaded Jina v4 model has no trained multi-vector projector")

    try:
        first_parameter = next(model.parameters())
        actual_dtype = str(first_parameter.dtype).replace("torch.", "")
    except StopIteration:
        actual_dtype = _dtype_name
    if _dtype_name == "bfloat16" and actual_dtype != "bfloat16":
        raise RuntimeError(
            f"Jina v4 requested BF16 but loaded parameter dtype is {actual_dtype}; refusing silent precision change"
        )

    _model = model
    _processor = processor
    _log(
        "model loaded; one forward will return 2048-d dense + N×128 multi-vectors "
        f"using {_dtype_name} inference"
    )
    return model


def _move_batch(batch: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in dict(batch).items():
        result[key] = value.to(_device) if hasattr(value, "to") else value
    return result


def _assert_finite_tensor(name: str, tensor: Any, *, context: str) -> None:
    if not hasattr(tensor, "numel") or tensor.numel() == 0:
        raise RuntimeError(f"Jina v4 {name} is empty ({context})")
    finite = _torch.isfinite(tensor)
    if bool(finite.all().item()):
        return
    finite_count = int(finite.sum().item())
    total = int(tensor.numel())
    nonfinite = total - finite_count
    raise RuntimeError(
        f"Jina v4 produced {nonfinite}/{total} non-finite values in {name} before storage "
        f"({context}; device={_device}; dtype={_dtype_name})."
    )


def _forward(batch: dict[str, Any], *, context: str) -> tuple[list[float], list[list[float]]]:
    model = _load()
    core = _core_model(model)
    moved = _move_batch(batch)
    kwargs = dict(moved)

    # Catch processor/input corruption separately from model numerical instability.
    for key, value in moved.items():
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            _assert_finite_tensor(f"input:{key}", value, context=context)

    # IMPORTANT: JinaEmbeddingsV4Model.forward requires task_label. A PEFT wrapper
    # exposes a generic forward signature, so signature introspection is unreliable.
    # PEFT forwards this kwarg to the Jina base model, which then selects the
    # retrieval adapter for both the single-vector and multi-vector outputs.
    kwargs["task_label"] = "retrieval"

    autocast_context = contextlib.nullcontext()
    if _dtype_name == "bfloat16":
        # Jina's own batching path performs BF16 autocast. Mirror that behavior for
        # both text and image inference while preserving the BF16 checkpoint dtype.
        autocast_context = _torch.autocast(device_type=_device, dtype=_torch.bfloat16)

    with _torch.inference_mode(), autocast_context, contextlib.redirect_stdout(sys.stderr):
        output = core(**kwargs)

    dense_tensor = output.single_vec_emb[0]
    multi_tensor = output.multi_vec_emb[0]
    mask = moved.get("attention_mask")
    if mask is not None:
        multi_tensor = multi_tensor[mask[0].bool()]

    # Validate the model outputs BEFORE float conversion/normalization. This makes a
    # precision failure attributable to the model forward instead of surfacing later
    # as a vague Weaviate validation error.
    _assert_finite_tensor("single_vec_emb", dense_tensor, context=context)
    _assert_finite_tensor("multi_vec_emb", multi_tensor, context=context)

    # Normalize in FP32 for numerically stable final vectors. This is an increase in
    # precision after the native BF16 forward, not a quantization or downgrade.
    dense_tensor = dense_tensor.float()
    multi_tensor = multi_tensor.float()
    dense_tensor = _torch.nn.functional.normalize(dense_tensor, p=2, dim=-1)
    multi_tensor = _torch.nn.functional.normalize(multi_tensor, p=2, dim=-1)
    _assert_finite_tensor("normalized single_vec_emb", dense_tensor, context=context)
    _assert_finite_tensor("normalized multi_vec_emb", multi_tensor, context=context)
    return dense_tensor.cpu().tolist(), multi_tensor.cpu().tolist()


def _embedding_health(result: dict[str, Any]) -> dict[str, Any]:
    dense = result.get("dense") or []
    multi = result.get("multi") or []
    dense_finite = bool(dense) and all(math.isfinite(float(v)) for v in dense)
    multi_finite = bool(multi) and all(
        math.isfinite(float(v)) for vector in multi for v in vector
    )
    return {
        "dense_dim": len(dense),
        "late_dim": len(multi[0]) if multi else 0,
        "late_vector_count": len(multi),
        "dense_finite": dense_finite,
        "multi_finite": multi_finite,
        "runtime_dtype": result.get("runtime_dtype"),
    }


def _process_text(text: str, role: str) -> dict[str, Any]:
    if role not in {"query", "passage"}:
        raise ValueError("role must be query or passage")
    if not text.strip():
        raise ValueError("text must not be empty")
    _load()
    prefix = "Query" if role == "query" else "Passage"
    batch = _processor.process_texts(
        texts=[text], prefix=prefix, max_length=TEXT_MAX_LENGTH
    )
    dense, multi = _forward(batch, context=f"text:{role}")
    return {
        "dense": dense,
        "multi": multi,
        "late_vector_count": len(multi),
        "dense_dim": len(dense),
        "late_dim": len(multi[0]) if multi else 0,
        "runtime_dtype": _dtype_name,
    }


def _process_image(path: str) -> dict[str, Any]:
    from PIL import Image, ImageOps

    _load()
    image_path = Path(path).expanduser().absolute()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = image.size
    batch = _processor.process_images(images=[image])
    dense, multi = _forward(
        batch,
        context=f"image:{image_path.name} {width}x{height}",
    )
    return {
        "dense": dense,
        "multi": multi,
        "late_vector_count": len(multi),
        "dense_dim": len(dense),
        "late_dim": len(multi[0]) if multi else 0,
        "runtime_dtype": _dtype_name,
        "image_width": width,
        "image_height": height,
    }


def _token_length(text: str) -> int:
    _load()
    tokenizer = getattr(_processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Jina v4 processor has no tokenizer")
    return len(tokenizer.encode(text, add_special_tokens=False))


def _chunk_text(text: str, chunk_size: int, overlap: int) -> dict[str, Any]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if chunk_size < 32:
        raise ValueError("chunk_size must be >= 32")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=_token_length,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    chunks = []
    for value in splitter.split_text(text):
        cleaned = value.strip()
        if cleaned:
            chunks.append({"text": cleaned, "token_count": _token_length(cleaned)})
    return {"chunks": chunks}


def _maxsim(q: list[list[float]], d: list[list[float]]) -> float:
    if not q or not d:
        raise ValueError("MaxSim requires non-empty multi-vector inputs")
    q_tensor = _torch.tensor(q, dtype=_torch.float32)
    d_tensor = _torch.tensor(d, dtype=_torch.float32)
    return float((q_tensor @ d_tensor.T).max(dim=1).values.sum().item())


def _status(load: bool = False) -> dict[str, Any]:
    global _torch, _transformers_version, _device, _dtype, _dtype_name
    if load:
        _load()
    else:
        import torch
        import transformers

        _torch = torch
        _transformers_version = transformers.__version__
        _device = _device or _resolve_device(torch)
        if _dtype is None:
            _dtype, _dtype_name = _resolve_dtype(torch, _device)
    return {
        "loaded": _model is not None,
        "device": _device,
        "dtype": _dtype_name,
        "dtype_request": DTYPE_REQUEST,
        "torch_version": _torch.__version__,
        "transformers_version": _transformers_version,
        "model_path": str(MODEL_PATH),
        "revision": REVISION,
        "dense_dim": 2048,
        "late_dim": 128,
        "text_max_length": TEXT_MAX_LENGTH,
    }


def _preflight() -> dict[str, Any]:
    from PIL import Image
    from tempfile import NamedTemporaryFile

    query = _process_text("red sports car", "query")
    relevant = _process_text("A bright red sports car is parked beside the road.", "passage")
    unrelated = _process_text("This document explains how to bake sourdough bread.", "passage")
    relevant_score = _maxsim(query["multi"], relevant["multi"])
    unrelated_score = _maxsim(query["multi"], unrelated["multi"])

    with NamedTemporaryFile(suffix=".png", delete=False) as red_file, NamedTemporaryFile(
        suffix=".png", delete=False
    ) as blue_file:
        red_path, blue_path = red_file.name, blue_file.name
    try:
        Image.new("RGB", (256, 256), (220, 30, 30)).save(red_path)
        Image.new("RGB", (256, 256), (25, 70, 200)).save(blue_path)
        image_query = _process_text("a mostly red image", "query")
        red = _process_image(red_path)
        blue = _process_image(blue_path)
        red_score = _maxsim(image_query["multi"], red["multi"])
        blue_score = _maxsim(image_query["multi"], blue["multi"])
    finally:
        Path(red_path).unlink(missing_ok=True)
        Path(blue_path).unlink(missing_ok=True)

    return {
        "status": _status(load=True),
        "text": {
            "query_health": _embedding_health(query),
            "passage_health": _embedding_health(relevant),
            "relevant_score": relevant_score,
            "unrelated_score": unrelated_score,
            "relevant_ranked_higher": relevant_score > unrelated_score,
        },
        "image": {
            "query_health": _embedding_health(image_query),
            "image_health": _embedding_health(red),
            "red_score": red_score,
            "blue_score": blue_score,
            # Diagnostic only: solid-color squares are intentionally not a hard
            # semantic correctness gate for a multimodal retrieval model.
            "red_ranked_higher": red_score > blue_score,
        },
    }


def _handle(request: dict[str, Any]) -> Any:
    op = request.get("op")
    if op == "status":
        return _status(load=False)
    if op == "encode_text":
        return _process_text(str(request.get("text") or ""), str(request.get("role") or "query"))
    if op == "encode_image":
        return _process_image(str(request.get("path") or ""))
    if op == "chunk_text":
        return _chunk_text(
            str(request.get("text") or ""),
            int(request.get("chunk_size") or 800),
            int(request.get("overlap") or 120),
        )
    if op == "preflight":
        return _preflight()
    raise ValueError(f"Unsupported Jina v4 worker operation: {op}")


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
            response = {"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
