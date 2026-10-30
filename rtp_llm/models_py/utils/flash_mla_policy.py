"""Process-wide FlashMLA initialization shared by sparse model adapters."""

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def configure_flash_mla_l2_persist() -> None:
    """Bind the existing V4.1 policy once, before inference.

    Per-call persisting-L2 setup invokes cudaDeviceSetLimit and can block the
    host behind previously queued layer work. Disable it by default. An
    explicit FLASH_MLA_NO_L2_PERSIST=0 restores the wheel's original policy;
    the wheel tests presence rather than value, so remove the native flag in
    that case. Cache this initialization so later layers do not reset it.
    """
    if os.environ.get("FLASH_MLA_NO_L2_PERSIST", "1") == "0":
        os.environ.pop("FLASH_MLA_NO_L2_PERSIST", None)
    else:
        os.environ.setdefault("FLASH_MLA_NO_L2_PERSIST", "1")
