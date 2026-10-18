"""Availability of the SM120 CUTLASS FP8 blockwise GEMM binding.

The binding is compiled only by ``--config=cuda13_sm120``.  Its presence on an
exact SM120 device selects the CUTLASS FP8 PER_BLOCK linear and the float
weight-scale layout it consumes; otherwise SM12x keeps the DeepGEMM path with
packed UE8M0 scales.  Loader and consumers must agree on this single decision.
"""

import functools
from typing import Callable, Optional, Union

import torch

from rtp_llm.models_py.utils.arch import is_sm120


@functools.lru_cache(maxsize=None)
def _load_op() -> Optional[Callable]:
    try:
        from rtp_llm.ops.compute_ops import (
            cutlass_scaled_mm_blockwise_sm120_fp8,
            has_cutlass_scaled_mm_blockwise_sm120_fp8,
        )
    except ImportError:
        return None
    if not has_cutlass_scaled_mm_blockwise_sm120_fp8():
        return None
    return cutlass_scaled_mm_blockwise_sm120_fp8


def sm120_blockwise_fp8_op(
    device_id: Optional[Union[int, str, torch.device]] = None,
) -> Optional[Callable]:
    """Return the CUTLASS SM120 FP8 blockwise op, or ``None`` if unavailable."""
    if not is_sm120(device_id):
        return None
    return _load_op()


def sm120_blockwise_fp8_available(
    device_id: Optional[Union[int, str, torch.device]] = None,
) -> bool:
    return sm120_blockwise_fp8_op(device_id) is not None
