"""Master switch for all Qwen3.5 decode fusions (C/D/E/F/G).

Set ``RTP_QWEN35_DECODE_FUSION=1`` (or true/on/yes) to enable every decode
fusion kernel. Unset / ``0`` / ``false`` keeps the old unfused path.

Host-side env only: CUDA-graph safe, no tensor ``.item()`` / ``.all()``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar

DECODE_FUSION_ENV = "RTP_QWEN35_DECODE_FUSION"
_TRUE_ENV_VALUES = {"1", "true", "t", "yes", "y", "on"}


_phase_enabled = ContextVar("qwen35_fusion_phase_enabled", default=None)


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in _TRUE_ENV_VALUES


@contextmanager
def fusion_phase(*, is_prefill: bool):
    """Enable reused fusion helpers only for decode; prefill stays native."""
    enabled = not is_prefill and _env_enabled(DECODE_FUSION_ENV)
    token = _phase_enabled.set(enabled)
    try:
        yield
    finally:
        _phase_enabled.reset(token)


def is_decode_fusion_enabled() -> bool:
    enabled = _phase_enabled.get()
    return _env_enabled(DECODE_FUSION_ENV) if enabled is None else enabled


def quantized_linear_for(linear):
    """Resolve a compatible UE8M0 consumer without changing backend selection."""
    if getattr(linear, "scale_ue8m0", False) and callable(
        getattr(linear, "forward_quantized", None)
    ):
        return linear
    return None
