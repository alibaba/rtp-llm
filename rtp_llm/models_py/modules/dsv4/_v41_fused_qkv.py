# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One MXFP8 projection for Q LoRA and KV, with copy-free Q normalization.

The strided norm is adapted from the Q branch of vLLM's
``vllm/models/common/ops/fused_qk_rmsnorm.py``. Keep its int64 row indexing,
FP32 arithmetic and eight-warps schedule for 2048-element tiles. KV stays in
RTP's existing fused norm/RoPE kernel to preserve its rounding boundary.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear


@triton.jit
def _strided_q_rmsnorm_kernel(
    X,
    W,
    Y,
    stride_x,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    w = tl.load(W + cols, cols < D, 0).to(tl.float32)
    x = tl.load(X + row * stride_x + cols, cols < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    # Match the framework RMSNorm: FP32 norm and multiplication, one BF16 store.
    tl.store(Y + row * D + cols, x * inv * w, cols < D)


def strided_q_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """Normalize a regular row-strided view and return a dense BF16 tensor."""
    shape = x.shape
    flat = x.view(-1, shape[-1])
    output = torch.empty(shape, device=x.device, dtype=x.dtype)
    if flat.shape[0]:
        _strided_q_rmsnorm_kernel[(flat.shape[0],)](
            flat,
            weight,
            output,
            flat.stride(0),
            D=shape[-1],
            EPS=eps,
            BLOCK=triton.next_power_of_2(shape[-1]),
            num_warps=8 if shape[-1] > 1024 else 4,
            enable_fp_fusion=False,
        )
    return output


def is_supported(linear, x, q_norm, q_rank) -> bool:
    """Metadata-only gate; unsupported inputs retain the two original linears."""
    return (
        isinstance(linear, V41MXFP8Linear)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim in (2, 3)
        and x.is_contiguous()
        and x.shape[-1] == linear.K
        and x.device == linear.weight.device == q_norm.device
        and q_norm.dtype == torch.bfloat16
        and q_norm.shape == (q_rank,)
        and q_norm.is_contiguous()
        and 0 < q_rank < linear.N
        and q_rank <= 8192
        and not (torch.is_grad_enabled() and (x.requires_grad or q_norm.requires_grad))
    )


def try_project_qr_kv(linear, x, q_norm, q_rank: int, eps: float):
    """Return normalized Q LoRA and raw KV, or None for the old path."""
    if not is_supported(linear, x, q_norm, q_rank):
        return None
    projected = linear(x)
    qr = strided_q_rmsnorm(projected[..., :q_rank], q_norm, eps)
    # KV's existing RMSNorm/RoPE kernel takes an explicit row stride.
    return qr, projected[..., q_rank:]
