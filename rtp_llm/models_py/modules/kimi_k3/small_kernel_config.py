"""Runtime opt-in for Kimi K3 small kernels shared across execution phases.

The environment variable retains its original decode name for compatibility;
eligible Prefill and target-verification operations use it as well.
"""

import os


def decode_small_kernels_enabled() -> bool:
    """Return whether the existing exact-value small-kernel opt-in is enabled."""

    return os.environ.get("KIMI_K3_DECODE_SMALL_KERNELS", "0") == "1"


__all__ = ["decode_small_kernels_enabled"]
