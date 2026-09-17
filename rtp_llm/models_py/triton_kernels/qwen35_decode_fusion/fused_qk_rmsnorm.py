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
) -> bool:
    del eps
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
):
    # One launch, grid (M, Q+K). Each CTA is one (token, head); V is never touched.
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    w_q = tl.load(q_weight_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
    w_k = tl.load(k_weight_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
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
) -> torch.Tensor:
    """Inplace Q/K RMSNorm on packed ``[M, (Q+K+V)*D]``. Returns ``hidden``."""
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
        num_warps = 2
    if num_stages is None:
        num_stages = 2
    _fused_qk_rmsnorm_kernel[(m, n_qk_heads)](
        hidden,
        q_weight,
        k_weight,
        hidden.stride(0),
        head_num,
        D=size_per_head,
        BLOCK_D=block_d,
        EPS=eps,
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
) -> Optional[torch.Tensor]:
    if not is_supported(
        hidden, q_weight, k_weight, head_num, kv_head_num, size_per_head, eps
    ):
        return None
    return fused_qk_rmsnorm(
        hidden, q_weight, k_weight, head_num, kv_head_num, size_per_head, eps
    )
