from __future__ import annotations

import json
import os
import select
import subprocess
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import settings


PROJECT_ROOT = Path(__file__).absolute().parents[2]


class JinaV4Client:
    """Persistent JSON-RPC client for the isolated Jina-v4 MPS worker."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path).expanduser()
        self._proc: subprocess.Popen[str] | None = None
        self._lock = Lock()
        self._cached_status: dict[str, Any] | None = None

    @property
    def worker_python(self) -> Path:
        path = Path(settings.jina_v4_python_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        # Preserve the venv symlink. Resolving it can bypass pyvenv.cfg and use the
        # base interpreter's site-packages, as we previously observed with m0.
        return path.absolute()

    def _validate_files(self) -> None:
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"Jina v4 model path not found: {self.model_path}. See docs/LATE_INTERACTION.md."
            )
        required = [
            "config.json",
            "modeling_jina_embeddings_v4.py",
            "qwen2_5_vl.py",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
        ]
        missing = [name for name in required if not (self.model_path / name).is_file()]
        if missing:
            raise FileNotFoundError("Incomplete Jina v4 checkpoint; missing: " + ", ".join(missing))

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        env["JINA_V4_MODEL_PATH"] = str(self.model_path.absolute())
        env["JINA_V4_DEVICE"] = str(settings.jina_v4_device)
        env["JINA_V4_ATTN_IMPLEMENTATION"] = str(settings.jina_v4_attn_implementation)
        env["JINA_V4_TEXT_MAX_LENGTH"] = str(settings.jina_v4_text_max_length)
        env["JINA_V4_REVISION"] = str(settings.jina_v4_revision)
        return env

    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._validate_files()
        if not self.worker_python.is_file():
            raise FileNotFoundError(
                f"Jina v4 worker Python not found: {self.worker_python}. "
                "Run: zsh scripts/setup_jina_v4_env.sh"
            )
        worker = PROJECT_ROOT / "scripts" / "jina_v4_worker.py"
        self._proc = subprocess.Popen(
            [str(self.worker_python), "-u", str(worker)],
            cwd=str(PROJECT_ROOT),
            env=self._worker_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        try:
            self._cached_status = self._rpc_unlocked("status")
        except Exception:
            self.close()
            raise

    def _readline(self) -> str:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Jina v4 worker is not running")
        timeout = float(settings.jina_v4_worker_timeout_sec)
        ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError(f"Jina v4 worker did not respond within {timeout:g}s")
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"Jina v4 worker exited unexpectedly ({self._proc.poll()})")
        return line

    def _rpc_unlocked(self, op: str, **payload: Any) -> Any:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Jina v4 worker is not running")
        request_id = uuid.uuid4().hex
        self._proc.stdin.write(json.dumps({"id": request_id, "op": op, **payload}) + "\n")
        self._proc.stdin.flush()
        raw = self._readline()
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from Jina v4 worker: {raw[:500]!r}") from exc
        if response.get("id") != request_id:
            raise RuntimeError("Jina v4 worker response ID mismatch")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "Unknown Jina v4 worker error"))
        result = response.get("result")
        if isinstance(result, dict) and "loaded" in result:
            self._cached_status = result
        return result

    def _rpc(self, op: str, **payload: Any) -> Any:
        with self._lock:
            self._start()
            try:
                return self._rpc_unlocked(op, **payload)
            except (BrokenPipeError, RuntimeError):
                if self._proc is not None and self._proc.poll() is not None:
                    self.close()
                    self._start()
                    return self._rpc_unlocked(op, **payload)
                raise

    def status(self) -> dict[str, Any]:
        running = self._proc is not None and self._proc.poll() is None
        return {
            "model_path": str(self.model_path),
            "model_exists": self.model_path.is_dir(),
            "worker_python": str(self.worker_python),
            "worker_python_exists": self.worker_python.is_file(),
            "worker_running": running,
            "loaded": bool(self._cached_status and self._cached_status.get("loaded")),
            "device": (self._cached_status or {}).get("device", settings.jina_v4_device),
            "dense_dim": 2048,
            "late_dim": 128,
            "revision": str(settings.jina_v4_revision),
            "worker_transformers": (self._cached_status or {}).get("transformers_version"),
            "worker_torch": (self._cached_status or {}).get("torch_version"),
        }

    def encode_text(self, text: str, *, role: str) -> dict[str, Any]:
        return self._rpc("encode_text", text=text, role=role)

    def encode_image(self, image_path: str | Path) -> dict[str, Any]:
        return self._rpc("encode_image", path=str(Path(image_path).expanduser().absolute()))

    def chunk_text(self, text: str, *, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
        result = self._rpc("chunk_text", text=text, chunk_size=chunk_size, overlap=overlap)
        return list(result["chunks"])

    def preflight(self) -> dict[str, Any]:
        return self._rpc("preflight")

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        self._cached_status = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


embedder = JinaV4Client(settings.jina_v4_model_path)
