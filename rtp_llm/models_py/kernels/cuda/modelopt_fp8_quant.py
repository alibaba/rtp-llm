"""Static FP8 quantization and row-scale materialization in one launch."""

import os

import torch
import triton as tr
import triton.language as tl

from rtp_llm.models_py.utils.arch import get_sm

# This fast path is bounded by the shapes/architecture validated against the
# native quantizer. Other layouts retain the existing implementation.
_VALIDATED_WIDTHS = frozenset((3072, 5120, 6144))
_VALIDATED_SM = (12, 0)
_FUSED_QUANT_ENABLED = os.getenv("RTP_LLM_MODELOPT_FP8_FUSED_QUANT", "1") == "1"


@tr.jit
def _quantize_rows(
    X, SCALE, Q, ROW_SCALE, K: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    scale = tl.load(SCALE)
    # Match per_tensor_quant_fp8_kernel's FP32 reciprocal and multiply order.
    reciprocal = tl.div_rn(1.0, tl.maximum(scale, 1.0e-9))
    x = tl.load(X + row * STRIDE + col, col < K, other=0).to(tl.float32)
    value = tl.maximum(
        tl.minimum(x * reciprocal, 448.0, propagate_nan=tl.PropagateNan.NONE),
        -448.0,
        propagate_nan=tl.PropagateNan.NONE,
    )
    tl.store(Q + row * K + col, value, col < K)
    tl.store(ROW_SCALE + row, scale)


def is_supported(input, scale):
    return (
        input.is_cuda
        and input.ndim == 2
        and input.dtype in (torch.bfloat16, torch.float16)
        and input.stride(1) == 1
        and input.shape[1] in _VALIDATED_WIDTHS
        and scale.is_cuda
        and scale.device == input.device
        and scale.dtype == torch.float32
        and scale.numel() == 1
        and get_sm(input.device) == _VALIDATED_SM
    )


def static_fp8_quant_rows(input, scale):
    if not _FUSED_QUANT_ENABLED or not is_supported(input, scale):
        return None
    m, k = input.shape
    quantized = torch.empty((m, k), device=input.device, dtype=torch.float8_e4m3fn)
    rows = torch.empty((m, 1), device=input.device, dtype=torch.float32)
    if m:
        _quantize_rows[(m,)](
            input,
            scale,
            quantized,
            rows,
            k,
            input.stride(0),
            tr.next_power_of_2(k),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return quantized, rows
