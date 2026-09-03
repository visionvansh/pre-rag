from __future__ import annotations

import math
import os
from pathlib import Path
from threading import Lock
from typing import Any

from .config import settings


# Must be present before torch/MPS initializes. Unsupported MPS ops may then fall
# back to CPU instead of crashing the entire local search request.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class JinaM0Reranker:
    """Lazy local wrapper around jinaai/jina-reranker-m0.

    Dense Jina-v5/Weaviate retrieval decides which candidates enter stage two.
    This model alone decides the ordering inside the stage-two result list.
    """

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        self.torch = None
        self.device = None
        self._load_lock = Lock()
        self._infer_lock = Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _resolve_device(self, torch) -> str:
        requested = str(settings.reranker_device or "auto").strip().lower()
        if requested == "auto":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError(
                    "RERANKER_DEVICE=mps was requested, but PyTorch MPS is unavailable. "
                    "Use RERANKER_DEVICE=cpu or fix the Apple-Silicon PyTorch install."
                )
            return "mps"
        if requested == "cpu":
            return "cpu"
        raise ValueError("RERANKER_DEVICE must be one of: auto, mps, cpu")

    def _validate_local_checkpoint(self) -> None:
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"Jina reranker path not found: {self.model_path}. "
                "Download jinaai/jina-reranker-m0 first."
            )
        required = [
            self.model_path / "config.json",
            self.model_path / "model.safetensors",
            self.model_path / "modeling.py",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "jina-reranker-m0 checkpoint is incomplete. Missing:\n- "
                + "\n- ".join(missing)
            )

    def load(self) -> None:
        if self.loaded:
            return
        if not settings.reranker_enabled:
            raise RuntimeError("Reranking is disabled by RERANKER_ENABLED=false")

        with self._load_lock:
            if self.loaded:
                return
            self._validate_local_checkpoint()

            import torch
            from transformers import AutoModel

            device = self._resolve_device(torch)
            kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": True,
                "torch_dtype": "auto",
            }
            attention = str(settings.reranker_attn_implementation or "").strip()
            if attention:
                kwargs["attn_implementation"] = attention

            model = AutoModel.from_pretrained(str(self.model_path.resolve()), **kwargs)
            model.eval()
            model.to(device)

            self.torch = torch
            self.device = device
            self.model = model

    def status(self) -> dict[str, Any]:
        device = self.device
        if device is None:
            try:
                import torch

                device = self._resolve_device(torch)
            except Exception:
                device = "unavailable"
        return {
            "enabled": bool(settings.reranker_enabled),
            "model_path": str(self.model_path),
            "model_exists": self.model_path.is_dir(),
            "loaded": self.loaded,
            "device": device,
            "candidate_default": int(settings.reranker_candidate_limit),
            "candidate_max": int(settings.reranker_max_candidates),
            "text_max_length": int(settings.reranker_text_max_length),
            "image_max_length": int(settings.reranker_image_max_length),
        }

    @staticmethod
    def _as_score_list(scores: Any, expected: int) -> list[float]:
        if expected == 1 and isinstance(scores, (int, float)):
            scores = [scores]
        output = [float(value) for value in scores]
        if len(output) != expected:
            raise RuntimeError(
                f"jina-reranker-m0 returned {len(output)} scores for {expected} documents"
            )
        if not all(math.isfinite(value) for value in output):
            raise RuntimeError("jina-reranker-m0 returned NaN or Inf relevance scores")
        return output

    def score_documents(
        self,
        query: Any,
        documents: list[Any],
        *,
        query_type: str,
        doc_type: str,
    ) -> list[float]:
        if not documents:
            return []
        if query_type not in {"text", "image"}:
            raise ValueError(f"Unsupported reranker query_type: {query_type}")
        if doc_type not in {"text", "image"}:
            raise ValueError(f"Unsupported reranker doc_type: {doc_type}")

        self.load()
        batch_size = (
            int(settings.reranker_image_batch_size)
            if (query_type == "image" or doc_type == "image")
            else int(settings.reranker_text_batch_size)
        )
        max_length = (
            int(settings.reranker_image_max_length)
            if (query_type == "image" or doc_type == "image")
            else int(settings.reranker_text_max_length)
        )
        max_query_length = min(int(settings.reranker_query_max_length), max_length // 2)
        pairs = [[query, document] for document in documents]

        with self._infer_lock:
            with self.torch.inference_mode():
                scores = self.model.compute_score(
                    pairs,
                    batch_size=max(1, batch_size),
                    max_length=max_length,
                    max_query_length=max_query_length,
                    query_type=query_type,
                    doc_type=doc_type,
                    normalize_scores=True,
                    show_progress=False,
                )
        return self._as_score_list(scores, len(documents))

    def preflight(self) -> dict[str, Any]:
        """Exercise text and visual scoring without requiring user assets."""
        from PIL import Image

        self.load()
        text_scores = self.score_documents(
            "red sports car",
            [
                "A bright red sports car is parked beside the road.",
                "This document explains how to bake sourdough bread.",
            ],
            query_type="text",
            doc_type="text",
        )

        red = Image.new("RGB", (384, 384), (220, 30, 30))
        blue = Image.new("RGB", (384, 384), (25, 70, 200))
        image_scores = self.score_documents(
            "a mostly red image",
            [red, blue],
            query_type="text",
            doc_type="image",
        )

        return {
            "status": self.status(),
            "text_scores": text_scores,
            "image_scores": image_scores,
            "text_relevant_ranked_higher": bool(text_scores[0] > text_scores[1]),
        }


reranker = JinaM0Reranker(settings.jina_reranker_path)
