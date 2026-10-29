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
_FUSED_PRODUCER_ENABLED = os.getenv("RTP_LLM_SM120_FUSION", "0") == "1"


def residual_norm_fp8_quant(input, residual, weight, scale, eps, communicator=None):
    """Consume TP2 partials and preserve the BF16 norm before static FP8 quant."""
    if not _FUSED_PRODUCER_ENABLED:
        return None
    from rtp_llm.models_py.kernels.cuda.modelopt_fp4_quant import (
        _norm_module,
        _residual_norm_supported,
    )

    if not _residual_norm_supported(input, residual, weight, scale, communicator):
        return None
    q = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    rows = torch.empty((input.shape[0], 1), device=input.device, dtype=torch.float32)
    _norm_module().run_fp8(
        input,
        residual,
        weight,
        scale,
        q.view(torch.uint8),
        rows,
        input,
        communicator.trt_pointers,
        communicator.rank,
        eps,
    )
    return q, rows


@tr.jit
def _quantize_rows(
    X, SCALE, Q, ROW_SCALE, K: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    # A large prefill can address more than 2^31 input/output elements.
    row = tl.program_id(0).to(tl.int64)
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


@tr.jit
def _gate_quantize_rows(
    X,
    Z,
    W,
    SCALE,
    Q,
    ROW_SCALE,
    K: tl.constexpr,
    STRIDE_X: tl.constexpr,
    STRIDE_Z: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EPS: tl.constexpr,
    RMS_GATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    offset = head * HEAD_DIM + col
    mask = col < HEAD_DIM
    x = tl.load(X + row * STRIDE_X + offset, mask, other=0).to(tl.float32)
    z = tl.load(Z + row * STRIDE_Z + offset, mask, other=0).to(tl.float32)
    if RMS_GATE:
        # Same reduction and FP32 expression order as layernorm_gated.py.
        var = tl.sum(x * x, 0) / HEAD_DIM
        rstd = 1.0 / tl.sqrt(var + EPS)
        weight = tl.load(W + col, mask, other=0).to(tl.float32)
        value = x * rstd * weight
        value *= z * tl.sigmoid(z)
    else:
        # The eager sigmoid and multiply each round to the activation dtype.
        gate = tl.sigmoid(z).to(X.dtype.element_ty).to(tl.float32)
        value = x * gate
    value = value.to(X.dtype.element_ty).to(tl.float32)
    scale = tl.load(SCALE)
    reciprocal = tl.div_rn(1.0, tl.maximum(scale, 1.0e-9))
    value = tl.maximum(
        tl.minimum(value * reciprocal, 448.0, propagate_nan=tl.PropagateNan.NONE),
        -448.0,
        propagate_nan=tl.PropagateNan.NONE,
    )
    tl.store(Q + row * K + offset, value, mask)
    if head == 0:
        tl.store(ROW_SCALE + row, scale)


def gated_fp8_quant_rows(input, gate, scale, *, norm_weight=None, eps=1e-6):
    """Fuse a projection's private gate/norm output into its static quantizer.

    Returns None for unsupported paths so the caller retains its old producer.
    The BF16 intermediate rounding is preserved before FP8 conversion.
    """
    if not _FUSED_PRODUCER_ENABLED or not is_supported(input, scale):
        return None
    if (
        input.dtype != torch.bfloat16
        or gate.shape != input.shape
        or gate.device != input.device
        or gate.dtype != input.dtype
        or gate.stride(1) != 1
    ):
        return None
    rms_gate = norm_weight is not None
    m, k = input.shape
    head_dim = 128 if rms_gate else k
    if rms_gate and (
        norm_weight.shape != (head_dim,)
        or norm_weight.device != input.device
        or norm_weight.dtype != input.dtype
        or not norm_weight.is_contiguous()
        or k % head_dim
    ):
        return None
    q = torch.empty((m, k), device=input.device, dtype=torch.float8_e4m3fn)
    rows = torch.empty((m, 1), device=input.device, dtype=torch.float32)
    if m:
        _gate_quantize_rows[(m, k // head_dim)](
            input,
            gate,
            norm_weight,
            scale,
            q,
            rows,
            k,
            input.stride(0),
            gate.stride(0),
            head_dim,
            eps,
            rms_gate,
            tr.next_power_of_2(head_dim),
            num_warps=1 if rms_gate else 4,
        )
    return q, rows
