from __future__ import annotations

import json
import os
import select
import subprocess
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from .config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class JinaM0Reranker:
    """Client for an isolated persistent jina-reranker-m0 worker.

    The main app intentionally keeps its newer Transformers stack for Jina-v5
    Omni/Qwen3-VL. m0 runs in .venv-reranker with Transformers 4.48.3 because
    its Qwen2-VL checkpoint/custom code uses the pre-refactor parameter layout.

    Dense Jina-v5/Weaviate retrieval decides which candidates enter stage two.
    m0 alone decides the ordering inside the stage-two result list.
    """

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._proc: subprocess.Popen[str] | None = None
        self._rpc_lock = Lock()
        self._cached_status: dict[str, Any] | None = None

    @property
    def worker_python(self) -> Path:
        path = Path(settings.reranker_python_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @property
    def loaded(self) -> bool:
        return bool(self._cached_status and self._cached_status.get("loaded"))

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

    def _validate_worker_python(self) -> None:
        if self.worker_python.is_file():
            return
        raise FileNotFoundError(
            f"Dedicated reranker Python not found: {self.worker_python}\n"
            "Create it with: ./scripts/setup_reranker_env.sh\n"
            "Do not downgrade Transformers in the main .venv; Jina-v5 Omni needs the newer stack."
        )

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        env["JINA_RERANKER_PATH"] = str(self.model_path.resolve())
        env["RERANKER_DEVICE"] = str(settings.reranker_device)
        env["RERANKER_TEXT_BATCH_SIZE"] = str(settings.reranker_text_batch_size)
        env["RERANKER_IMAGE_BATCH_SIZE"] = str(settings.reranker_image_batch_size)
        env["RERANKER_TEXT_MAX_LENGTH"] = str(settings.reranker_text_max_length)
        env["RERANKER_IMAGE_MAX_LENGTH"] = str(settings.reranker_image_max_length)
        env["RERANKER_QUERY_MAX_LENGTH"] = str(settings.reranker_query_max_length)
        env["RERANKER_ATTN_IMPLEMENTATION"] = str(settings.reranker_attn_implementation)
        return env

    def _start_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not settings.reranker_enabled:
            raise RuntimeError("Reranking is disabled by RERANKER_ENABLED=false")
        self._validate_local_checkpoint()
        self._validate_worker_python()

        worker = PROJECT_ROOT / "scripts" / "reranker_worker.py"
        self._proc = subprocess.Popen(
            [str(self.worker_python), "-u", str(worker)],
            cwd=str(PROJECT_ROOT),
            env=self._worker_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Keep stderr visible in the terminal. Model/tqdm diagnostics must never
            # share stdout with the JSON-line RPC protocol.
            stderr=None,
            text=True,
            bufsize=1,
        )
        try:
            self._cached_status = self._rpc_unlocked("status")
        except Exception:
            self.close()
            raise

    def _readline_with_timeout(self) -> str:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Reranker worker is not running")
        timeout = float(settings.reranker_worker_timeout_sec)
        ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError(
                f"Reranker worker did not respond within {timeout:g}s. "
                "The first MPS model load can be slow; increase RERANKER_WORKER_TIMEOUT_SEC if needed."
            )
        line = self._proc.stdout.readline()
        if not line:
            code = self._proc.poll()
            raise RuntimeError(f"Reranker worker exited unexpectedly (exit code {code})")
        return line

    def _rpc_unlocked(self, op: str, **payload: Any) -> Any:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Reranker worker is not running")
        request_id = uuid.uuid4().hex
        message = {"id": request_id, "op": op, **payload}
        self._proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        raw = self._readline_with_timeout()
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from reranker worker: {raw[:500]!r}") from exc
        if response.get("id") != request_id:
            raise RuntimeError("Reranker worker response ID mismatch")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "Unknown reranker worker error"))
        result = response.get("result")
        if isinstance(result, dict) and "loaded" in result:
            self._cached_status = result
        return result

    def _rpc(self, op: str, **payload: Any) -> Any:
        with self._rpc_lock:
            self._start_worker()
            try:
                return self._rpc_unlocked(op, **payload)
            except (BrokenPipeError, RuntimeError):
                # A dead child gets one clean restart. Model errors returned by a
                # live worker are not retried blindly.
                if self._proc is not None and self._proc.poll() is not None:
                    self.close()
                    self._start_worker()
                    return self._rpc_unlocked(op, **payload)
                raise

    def status(self) -> dict[str, Any]:
        worker_running = self._proc is not None and self._proc.poll() is None
        return {
            "enabled": bool(settings.reranker_enabled),
            "model_path": str(self.model_path),
            "model_exists": self.model_path.is_dir(),
            "loaded": self.loaded,
            "device": (self._cached_status or {}).get("device", str(settings.reranker_device)),
            "candidate_default": int(settings.reranker_candidate_limit),
            "candidate_max": int(settings.reranker_max_candidates),
            "text_max_length": int(settings.reranker_text_max_length),
            "image_max_length": int(settings.reranker_image_max_length),
            "worker_python": str(self.worker_python),
            "worker_python_exists": self.worker_python.is_file(),
            "worker_running": worker_running,
            "worker_transformers": (self._cached_status or {}).get("transformers_version"),
            "worker_torch": (self._cached_status or {}).get("torch_version"),
        }

    @staticmethod
    def _image_to_path(value: Any, cleanup: list[Path]) -> str:
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Reranker image file not found: {path}")
            return str(path)

        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required to materialize image reranker queries") from exc

        if isinstance(value, Image.Image):
            temp_dir = settings.app_data_dir / "reranker_tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            path = temp_dir / f"query-{uuid.uuid4().hex}.png"
            value.convert("RGB").save(path, format="PNG")
            cleanup.append(path)
            return str(path.resolve())
        raise TypeError(f"Unsupported image value for reranking: {type(value).__name__}")

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

        cleanup: list[Path] = []
        try:
            wire_query: Any = str(query) if query_type == "text" else self._image_to_path(query, cleanup)
            wire_documents: list[Any] = (
                [str(item) for item in documents]
                if doc_type == "text"
                else [self._image_to_path(item, cleanup) for item in documents]
            )
            result = self._rpc(
                "score",
                query=wire_query,
                documents=wire_documents,
                query_type=query_type,
                doc_type=doc_type,
            )
            scores = [float(value) for value in result["scores"]]
            if len(scores) != len(documents):
                raise RuntimeError(
                    f"jina-reranker-m0 returned {len(scores)} scores for {len(documents)} documents"
                )
            return scores
        finally:
            for path in cleanup:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

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


reranker = JinaM0Reranker(settings.jina_reranker_path)
