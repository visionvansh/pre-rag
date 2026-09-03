from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import numpy as np

from .config import settings
from .jina_compat import JinaCompatibilityReport, load_jina_model_module


class JinaMLXEmbedder:
    """Lazy, single-process wrapper around the official Jina MLX checkpoint."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._loaded = False
        self._load_lock = Lock()
        self._processor_lock = Lock()
        self._infer_lock = Lock()
        self.mx = None
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.compatibility: JinaCompatibilityReport | None = None
        self._model_module = None

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            if not self.model_path.exists():
                raise FileNotFoundError(f"Jina model path not found: {self.model_path}")

            required = [
                self.model_path / "config.json",
                self.model_path / "model.py",
                self.model_path / "model.safetensors",
                self.model_path / "tokenizer.json",
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Jina checkpoint is incomplete. Missing:\n- " + "\n- ".join(missing)
                )

            import mlx.core as mx
            from tokenizers import Tokenizer

            model_module, compatibility = load_jina_model_module(self.model_path)
            JinaOmniSmallEmbeddingModel = model_module.JinaOmniSmallEmbeddingModel
            OmniSmallConfig = model_module.OmniSmallConfig

            cfg = OmniSmallConfig.from_dict(
                json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
            )
            model = JinaOmniSmallEmbeddingModel(cfg)
            model.load_weights(str(self.model_path / "model.safetensors"))
            mx.eval(model.parameters())

            self.mx = mx
            self.model = model
            self.tokenizer = Tokenizer.from_file(str(self.model_path / "tokenizer.json"))
            self.processor = None
            self.compatibility = compatibility
            self._model_module = model_module
            self._loaded = True

    def compatibility_report(self) -> dict:
        self.load()
        return self.compatibility.to_dict() if self.compatibility else {}

    def _load_processor(self):
        """Lazy-load Transformers/Qwen processing only for vision paths."""
        self.load()
        if self.processor is not None:
            return self.processor
        with self._processor_lock:
            if self.processor is not None:
                return self.processor
            try:
                from transformers import AutoProcessor

                self.processor = AutoProcessor.from_pretrained(
                    str(self.model_path.resolve()),
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except ImportError as exc:
                message = str(exc)
                if "Torchvision" in message or "torchvision" in message:
                    raise RuntimeError(
                        "Vision embedding requires torchvision. Install it inside the "
                        "active virtual environment with: python -m pip install torchvision"
                    ) from exc
                raise
        return self.processor

    def token_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer.encode(text).ids)

    def _to_list(self, emb) -> list[float]:
        mx = self.mx
        arr = np.asarray(emb[0].astype(mx.float32), dtype=np.float32)
        if arr.shape != (1024,):
            raise ValueError(f"Expected 1024-d Jina embedding, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("Embedding contains NaN or Inf")
        norm = float(np.linalg.norm(arr))
        if not 0.98 <= norm <= 1.02:
            raise ValueError(f"Expected an L2-normalized Jina embedding; norm={norm:.6f}")
        return arr.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text("Query: " + text.strip())

    def embed_document(self, text: str) -> list[float]:
        return self._embed_text("Document: " + text.strip())

    def _embed_text(self, prefixed_text: str) -> list[float]:
        self.load()
        with self._infer_lock:
            enc = self.tokenizer.encode(prefixed_text)
            input_ids = self.mx.array([enc.ids])
            attn = self.mx.array([enc.attention_mask])
            emb = self.model.encode_text(input_ids, attn)
            self.mx.eval(emb)
            return self._to_list(emb)

    def embed_documents_batch(self, texts: list[str]) -> list[list[float]]:
        self.load()
        if not texts:
            return []
        with self._infer_lock:
            encs = self.tokenizer.encode_batch(["Document: " + t.strip() for t in texts])
            max_len = max(len(e.ids) for e in encs)
            pad = self.tokenizer.token_to_id("<|endoftext|>") or 0
            input_ids, attention = [], []
            for enc in encs:
                n = max_len - len(enc.ids)
                input_ids.append(enc.ids + [pad] * n)
                attention.append(enc.attention_mask + [0] * n)
            embs = self.model.encode_text(self.mx.array(input_ids), self.mx.array(attention))
            self.mx.eval(embs)
            output = []
            for i in range(len(texts)):
                arr = np.asarray(embs[i].astype(self.mx.float32), dtype=np.float32)
                if arr.shape != (1024,) or not np.isfinite(arr).all():
                    raise ValueError("Invalid Jina batch embedding")
                norm = float(np.linalg.norm(arr))
                if not 0.98 <= norm <= 1.02:
                    raise ValueError(f"Invalid Jina batch embedding norm: {norm:.6f}")
                output.append(arr.tolist())
            return output

    def embed_image(self, image) -> list[float]:
        """Embed one PIL image using Jina's official image prompt."""
        self.load()
        image = image.convert("RGB")
        prompt = "<|vision_start|><|image_pad|><|vision_end|>"
        processor = self._load_processor()

        with self._infer_lock:
            inputs = processor(images=[image], text=prompt, return_tensors="pt")
            required = {"pixel_values", "image_grid_thw", "input_ids", "attention_mask"}
            missing = required.difference(inputs.keys())
            if missing:
                raise RuntimeError(
                    "Jina/Qwen image processor output is missing required fields: "
                    + ", ".join(sorted(missing))
                )

            pixel_values_np = inputs["pixel_values"].detach().cpu().numpy()
            grid_thw_np = inputs["image_grid_thw"].detach().cpu().numpy()
            input_ids_np = inputs["input_ids"].detach().cpu().numpy()
            attn_np = inputs["attention_mask"].detach().cpu().numpy()

            emb = self.model.encode_image(
                self.mx.array(pixel_values_np),
                self.mx.array(grid_thw_np),
                self.mx.array(input_ids_np),
                self.mx.array(attn_np),
            )
            self.mx.eval(emb)
            return self._to_list(emb)

    def embed_video_frames(self, frames) -> list[float]:
        """Embed a sequence of PIL RGB frames using Jina's official video prompt."""
        self.load()
        if not frames:
            raise ValueError("No video frames supplied")

        frames = list(frames)
        if len(frames) == 1:
            frames = [frames[0], frames[0].copy()]
        elif len(frames) % 2:
            frames = frames[:-1]
        if not frames:
            raise ValueError("Video frame normalization produced an empty frame list")

        prompt = "<|vision_start|><|video_pad|><|vision_end|>"
        processor = self._load_processor()

        with self._infer_lock:
            inputs = processor(text=prompt, videos=frames, return_tensors="pt")
            required = {"pixel_values_videos", "video_grid_thw", "input_ids", "attention_mask"}
            missing = required.difference(inputs.keys())
            if missing:
                raise RuntimeError(
                    "Jina/Qwen video processor output is missing required fields: "
                    + ", ".join(sorted(missing))
                )

            pixel_values_np = inputs["pixel_values_videos"].detach().cpu().numpy()
            grid_thw_np = inputs["video_grid_thw"].detach().cpu().numpy()
            input_ids_np = inputs["input_ids"].detach().cpu().numpy()
            attn_np = inputs["attention_mask"].detach().cpu().numpy()

            if grid_thw_np.ndim != 2 or grid_thw_np.shape[1] != 3:
                raise RuntimeError(f"Unexpected video_grid_thw shape: {grid_thw_np.shape}")
            if int(grid_thw_np[:, 0].sum()) <= 0:
                raise RuntimeError(f"Invalid temporal video grid: {grid_thw_np.tolist()}")

            try:
                emb = self.model.encode_video(
                    self.mx.array(pixel_values_np),
                    self.mx.array(grid_thw_np),
                    self.mx.array(input_ids_np),
                    self.mx.array(attn_np),
                )
            except TypeError as exc:
                message = str(exc)
                if "repeat(): incompatible function arguments" in message:
                    report = self.compatibility_report()
                    raise RuntimeError(
                        "Jina MLX vision forward hit the known mx.repeat scalar/int "
                        f"compatibility failure. Compatibility report: {report}"
                    ) from exc
                raise
            self.mx.eval(emb)
            return self._to_list(emb)

    def image_preflight(self) -> dict:
        from PIL import Image

        image = Image.new("RGB", (256, 256), (48, 88, 132))
        vector = self.embed_image(image)
        return {
            "embedding_dim": len(vector),
            "compatibility": self.compatibility_report(),
        }

    def video_preflight(self) -> dict:
        """Run a tiny synthetic 4-frame video through processor + MLX vision tower."""
        from PIL import Image

        frames = [
            Image.new("RGB", (256, 256), (24, 36, 52)),
            Image.new("RGB", (256, 256), (64, 86, 110)),
            Image.new("RGB", (256, 256), (108, 78, 48)),
            Image.new("RGB", (256, 256), (36, 112, 74)),
        ]
        vector = self.embed_video_frames(frames)
        return {
            "embedding_dim": len(vector),
            "compatibility": self.compatibility_report(),
        }


embedder = JinaMLXEmbedder(settings.jina_model_path)
