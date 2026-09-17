"""Optional FA4 path for the measured V4.1 93x93 vision patch grid."""

from functools import lru_cache
import logging

import torch

_LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_fa4():
    try:
        from flash_attn.cute.interface import flash_attn_func
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        _LOGGER.warning("V4.1 vision FA4 unavailable; using SDPA: %s", reason)
        return None, reason
    _LOGGER.info("V4.1 vision FA4 available for the 93x93 patch grid")
    return flash_attn_func, None


def vision_fa4_status():
    """Report the optional dependency result, including the cached import failure."""
    function, reason = _load_fa4()
    return {
        "enabled": True,
        "available": function is not None,
        "unavailable_reason": reason,
        "patch_counts": [8649],
    }


def _enabled_or_supported(q, k, v):
    # Smaller images favor cuDNN once the FA4 Python launch cost is included;
    # on SM100 FA4 also measured slower than cuDNN at every grid, so it stays (10, 3).
    return (
        not torch.is_grad_enabled()
        and q.is_cuda
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.shape == k.shape == v.shape == (8649, 16, 64)
        and q.stride() == k.stride() == (2048, 64, 1)
        and v.stride() == (3072, 64, 1)
        and q.device == k.device == v.device
        and torch.cuda.get_device_capability(q.device) == (10, 3)
    )


def vision_attention_fa4(q, k, v):
    """Return [patches, heads, dim], or None for an unsupported/unavailable path."""
    if not _enabled_or_supported(q, k, v):
        return None
    function, _ = _load_fa4()
    if function is None:
        return None
    output, _ = function(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal=False)
    return output.squeeze(0)
