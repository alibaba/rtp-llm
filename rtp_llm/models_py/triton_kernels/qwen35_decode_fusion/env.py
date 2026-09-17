"""Master switch for all Qwen3.5 decode fusions (C/D/E/F/G).

Set ``RTP_QWEN35_DECODE_FUSION=1`` (or true/on/yes) to enable every decode
fusion kernel. Unset / ``0`` / ``false`` keeps the old unfused path.

Host-side env only: CUDA-graph safe, no tensor ``.item()`` / ``.all()``.
"""

from __future__ import annotations

import os

DECODE_FUSION_ENV = "RTP_QWEN35_DECODE_FUSION"
_TRUE_ENV_VALUES = {"1", "true", "t", "yes", "y", "on"}


def is_decode_fusion_enabled() -> bool:
    raw = os.environ.get(DECODE_FUSION_ENV, "0").strip().lower()
    return raw in _TRUE_ENV_VALUES
