"""Selection and availability for the DSv4 SM120 blockwise FP8 binding."""

from typing import Callable, Optional, Union

import torch

from rtp_llm.models_py.utils.arch import is_sm120

_loaded_op: Optional[Callable] = None
_probe_done = False


def _load_op() -> Optional[Callable]:
    # A missing symbol after the ops library loads is stable, so the probe
    # result is cached. An ImportError only means the library has not been
    # loaded yet, so it is retried instead of disabling the backend for the
    # whole process.
    global _loaded_op, _probe_done
    if _probe_done:
        return _loaded_op
    try:
        from rtp_llm.ops.compute_ops import (
            cutlass_scaled_mm_blockwise_sm120_fp8,
            has_cutlass_scaled_mm_blockwise_sm120_fp8,
        )
    except ImportError:
        return None
    _probe_done = True
    if has_cutlass_scaled_mm_blockwise_sm120_fp8():
        _loaded_op = cutlass_scaled_mm_blockwise_sm120_fp8
    return _loaded_op


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


def prefers_sm120_blockwise(quant_config: object) -> bool:
    return getattr(quant_config, "prefer_sm120_blockwise", False) is True


def sm120_blockwise_layout(
    quant_config: object,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    *,
    device_id: Optional[Union[int, str, torch.device]] = None,
) -> bool:
    """Check compact FP8 layout and reject malformed SM120 opt-ins."""
    if not prefers_sm120_blockwise(quant_config) or weight_scales.dtype not in (
        torch.float32,
        torch.float8_e8m0fnu,
    ):
        return False
    valid = (
        weight.dtype == torch.float8_e4m3fn
        and weight.device == weight_scales.device
        and weight.ndim == 2
        and weight.is_contiguous()
        and weight_scales.is_contiguous()
        and tuple(weight_scales.shape)
        == tuple((size + 127) // 128 for size in weight.shape)
    )
    if not valid and is_sm120(weight.device if device_id is None else device_id):
        raise ValueError(
            "DSv4 SM120 blockwise FP8 requires contiguous output-major FP8 "
            "weights and same-device contiguous compact scales with shape "
            "(ceil(N/128), ceil(K/128))"
        )
    return valid
