from __future__ import annotations

import hashlib
import sys
import types
from dataclasses import dataclass, asdict
from pathlib import Path


# Current upstream jina-embeddings-v5-omni-small-retrieval-mlx/model.py builds
# vision cumulative sequence lengths with an MLX scalar array as mx.repeat's
# `repeats` argument. MLX requires a native Python int there.
_BAD_VISION_REPEAT = "cu_seqlens.append(mx.repeat(seq_len_i, grid_thw[i, 0]))"
_FIXED_VISION_REPEAT = (
    "cu_seqlens.append(mx.repeat(seq_len_i, int(grid_thw[i, 0].item())))"
)


@dataclass(frozen=True)
class JinaCompatibilityReport:
    source_sha256: str
    vision_repeat_patch_applied: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def patch_jina_model_source(source: str) -> tuple[str, JinaCompatibilityReport]:
    """Patch only the known upstream MLX scalar/int incompatibility.

    The transformation is intentionally exact and idempotent:
    - if Jina already contains the fixed expression, do nothing;
    - if exactly one known buggy expression is present, replace only that line;
    - if neither is present, do nothing rather than guessing at a new upstream layout.
    """
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()

    if _FIXED_VISION_REPEAT in source:
        return source, JinaCompatibilityReport(
            source_sha256=digest,
            vision_repeat_patch_applied=False,
            detail="Upstream model already uses a Python int for mx.repeat(repeats).",
        )

    count = source.count(_BAD_VISION_REPEAT)
    if count == 1:
        patched = source.replace(_BAD_VISION_REPEAT, _FIXED_VISION_REPEAT, 1)
        return patched, JinaCompatibilityReport(
            source_sha256=digest,
            vision_repeat_patch_applied=True,
            detail=(
                "Applied Jina MLX vision compatibility patch: converted "
                "grid_thw[i, 0] to int(grid_thw[i, 0].item()) for mx.repeat."
            ),
        )

    if count > 1:
        raise RuntimeError(
            "Refusing to patch Jina model.py because the known mx.repeat pattern "
            f"appeared {count} times instead of once."
        )

    return source, JinaCompatibilityReport(
        source_sha256=digest,
        vision_repeat_patch_applied=False,
        detail=(
            "Known mx.repeat incompatibility pattern was not present. "
            "No source transformation was applied."
        ),
    )


def load_jina_model_module(model_dir: Path):
    """Load the local Jina model.py under a private module name.

    We compile a patched copy in memory. The downloaded Hugging Face checkpoint is
    never modified on disk, so its files/checksums stay intact. Keeping a private
    module name also avoids collisions with unrelated packages named `model`.
    """
    model_dir = Path(model_dir).resolve()
    source_path = model_dir / "model.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"Jina model.py not found: {source_path}")

    source = source_path.read_text(encoding="utf-8")
    source, report = patch_jina_model_source(source)

    # Preserve the model directory on sys.path in case a future upstream model.py
    # imports sibling helper modules.
    model_dir_str = str(model_dir)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)

    module_name = f"_pre_rag_jina_model_{report.source_sha256[:12]}"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(source_path), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module, report
