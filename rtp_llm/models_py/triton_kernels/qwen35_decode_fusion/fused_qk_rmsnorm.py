"""True fused Q/K RMSNorm: one Triton launch over all Q heads + K heads.

Replaces the two ``flashinfer.norm.rmsnorm`` launches in
``FusedQKRMSNorm.forward`` (see ``modules/base/cuda/norm.py``).  V heads are
not touched.  Not fused into ``decode_add_fusedQKV_bias_transpose_kernel``.

Math matches DSV4 / flashinfer per-head RMSNorm (last dim, weight broadcast)::

    x_f32 = x.to(fp32)
    var = sum(x_f32 * x_f32) / D
    y = x_f32 * rsqrt(var + eps) * w_f32
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.env import (
    is_decode_fusion_enabled,
)
from rtp_llm.models_py.utils.prefill_input_log import trace_triton

_MAX_HEAD_DIM = 4096
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _fusion_enabled() -> bool:
    return is_decode_fusion_enabled()


def _hidden_last_dim(head_num: int, kv_head_num: int, size_per_head: int) -> int:
    return (head_num + kv_head_num * 2) * size_per_head


def _inputs_supported(
    hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    head_num: int,
    kv_head_num: int,
    size_per_head: int,
) -> bool:
    last_dim = _hidden_last_dim(head_num, kv_head_num, size_per_head)
    return (
        isinstance(hidden, torch.Tensor)
        and hidden.dim() == 2
        and hidden.is_cuda
        and hidden.stride(-1) == 1
        and hidden.dtype in _SUPPORTED_DTYPES
        and head_num > 0
        and kv_head_num > 0
        and 0 < size_per_head <= _MAX_HEAD_DIM
        and hidden.shape[-1] == last_dim
        and q_weight.shape == (size_per_head,)
        and k_weight.shape == (size_per_head,)
        and q_weight.dtype == hidden.dtype
        and k_weight.dtype == hidden.dtype
        and q_weight.is_cuda
        and k_weight.is_cuda
        and q_weight.stride(-1) == 1
        and k_weight.stride(-1) == 1
    )


def is_supported(
    hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    head_num: int,
    kv_head_num: int,
    size_per_head: int,
    eps: float = 1e-6,
    *,
    zero_centered: bool = False,
) -> bool:
    del eps, zero_centered
    return _fusion_enabled() and _inputs_supported(
        hidden, q_weight, k_weight, head_num, kv_head_num, size_per_head
    )


@triton.jit
def _fused_qk_rmsnorm_kernel(
    hidden_ptr,
    q_weight_ptr,
    k_weight_ptr,
    stride_m,
    head_num,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
):
    # One launch, grid (M, Q+K). Each CTA is one (token, head); V is never touched.
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    w_q = tl.load(q_weight_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
    w_k = tl.load(k_weight_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
    if ZERO_CENTERED:
        w_q = w_q + 1.0
        w_k = w_k + 1.0
    row = hidden_ptr + token * stride_m + head * D
    x = tl.load(row + d_off, mask=d_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    y = x * tl.rsqrt(var + EPS) * tl.where(head < head_num, w_q, w_k)
    tl.store(row + d_off, y, mask=d_mask)


def fused_qk_rmsnorm(
    hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    head_num: int,
    kv_head_num: int,
    size_per_head: int,
    eps: float = 1e-6,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    *,
    zero_centered: bool = False,
) -> torch.Tensor:
    """Inplace Q/K RMSNorm on packed ``[M, (Q+K+V)*D]``.

    With ``zero_centered=True``, weights are raw values and the kernel adds
    one after widening to FP32. Ordinary gamma semantics remain the default.
    """
    if not _inputs_supported(
        hidden, q_weight, k_weight, head_num, kv_head_num, size_per_head
    ):
        raise ValueError("unsupported input for fused_qk_rmsnorm")

    m = hidden.shape[0]
    n_qk_heads = head_num + kv_head_num
    if m == 0 or n_qk_heads == 0:
        return hidden

    block_d = triton.next_power_of_2(size_per_head)
    if num_warps is None:
        # V's actual fused QK+RoPE norm uses HEAD_BLOCK//64 warps.
        # Ordinary effective-gamma callers keep their original launch.
        num_warps = (
            max(1, block_d // 64) if zero_centered and size_per_head == 256 else 2
        )
    if num_stages is None:
        num_stages = 2
    trace_triton(
        "fused_qk_rmsnorm:_fused_qk_rmsnorm_kernel[m, n_qk_heads]",
        _fused_qk_rmsnorm_kernel,
        (m, n_qk_heads),
        hidden,
        q_weight,
        k_weight,
        hidden.stride(0),
        head_num,
        D=size_per_head,
        BLOCK_D=block_d,
        EPS=eps,
        ZERO_CENTERED=bool(zero_centered),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return hidden


def maybe_fused_qk_rmsnorm(
    hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    head_num: int,
    kv_head_num: int,
    size_per_head: int,
    eps: float = 1e-6,
    *,
    zero_centered: bool = False,
) -> Optional[torch.Tensor]:
    if not is_supported(
        hidden,
        q_weight,
        k_weight,
        head_num,
        kv_head_num,
        size_per_head,
        eps,
        zero_centered=zero_centered,
    ):
        return None
    return fused_qk_rmsnorm(
        hidden,
        q_weight,
        k_weight,
        head_num,
        kv_head_num,
        size_per_head,
        eps,
        zero_centered=zero_centered,
    )


@triton.jit
def _vllm_style_qk_zero_centered_rmsnorm(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_stride_m,
    output_stride_m,
    N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    # Norm portion of V's actual fused QK+RoPE+gate operator. RoPE is untouched.
    token = tl.program_id(0)
    offsets = tl.arange(0, HEAD_BLOCK)
    mask = offsets < N
    value = tl.load(
        input_ptr + token * input_stride_m + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    variance = tl.sum(value * value, axis=0) / N
    inv = tl.rsqrt(variance + EPS)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32) + 1.0
    normed = (value * inv * weight).to(input_ptr.dtype.element_ty)
    tl.store(output_ptr + token * output_stride_m + offsets, normed, mask=mask)


def vllm_style_zero_centered_rmsnorm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    output: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Width-256 Q/K norm with raw gamma + 1 computed in FP32.

    Match vLLM's QK norm reduction and launch configuration. Width-4096
    hidden-state norms use FlashInfer and may have different rounding.
    """
    if (
        hidden.ndim != 2
        or weight.ndim != 1
        or hidden.shape[1] != weight.numel()
        or hidden.dtype not in _SUPPORTED_DTYPES
        or weight.dtype != hidden.dtype
        or not hidden.is_cuda
        or not weight.is_cuda
        or hidden.device != weight.device
        or hidden.stride(-1) != 1
        or weight.stride(-1) != 1
    ):
        raise ValueError(
            "zero-centered aligned norm expects CUDA row matrices and matching raw gamma"
        )
    m, n = hidden.shape
    if n != 256:
        raise ValueError("vLLM-aligned QK norm supports width 256 only")
    if output is None:
        output = torch.empty_like(hidden)
    if (
        output.shape != hidden.shape
        or output.dtype != hidden.dtype
        or output.device != hidden.device
        or output.stride(-1) != 1
    ):
        raise ValueError("zero-centered aligned norm output ABI differs")
    if m == 0:
        return output
    if residual is not None:
        raise ValueError("Width256 policy is the QK norm without residual-add")
    trace_triton(
        "rmsnorm:_vllm_style_qk_zero_centered_rmsnorm",
        _vllm_style_qk_zero_centered_rmsnorm,
        (m,),
        hidden,
        weight,
        output,
        hidden.stride(0),
        output.stride(0),
        N=n,
        HEAD_BLOCK=256,
        EPS=eps,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=True,
    )
    return output
