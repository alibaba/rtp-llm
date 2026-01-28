from __future__ import annotations

from typing import Optional


def is_hip_runtime() -> bool:
    """Return True if running under ROCm/HIP torch build.

    We use torch.version.hip because torch.cuda.is_available() is often True on ROCm.
    """
    try:
        import torch  # type: ignore

        return bool(getattr(getattr(torch, "version", None), "hip", None))
    except Exception:
        return False


def skip_if_hip(reason: str, *, allow_module_level: bool = True) -> None:
    """Skip current test module/case when running on ROCm/HIP."""
    if not is_hip_runtime():
        return
    import pytest  # type: ignore

    pytest.skip(reason, allow_module_level=allow_module_level)

