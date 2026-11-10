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

from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


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


@triton.jit
def _widen_bf16_bits(value):
    return (value.to(tl.uint16, bitcast=True).to(tl.uint32) << 16).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _rmsnorm_bf16_square_kernel(X, Square, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    x = _widen_bf16_bits(tl.load(X + offsets, mask, other=0))
    square = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(Square + offsets, square, mask)


def _rmsnorm_bf16_square(x: torch.Tensor) -> torch.Tensor:
    square = torch.empty_like(x, dtype=torch.float32)
    elements = square.numel()
    if elements:
        block = 1024
        _rmsnorm_bf16_square_kernel[(triton.cdiv(elements, block),)](
            x, square, elements, BLOCK=block
        )
    return square


@triton.jit
def _prefill_rmsnorm_epilogue_kernel(
    X,
    Inv,
    Gamma,
    Out,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    X_BF16: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < D
    x = tl.load(X + row * D + col, mask, other=0)
    if X_BF16:
        x = _widen_bf16_bits(x)
    inv = tl.load(Inv + row)
    gamma = tl.load(Gamma + col, mask, other=0)
    # Widen by bit pattern to retain BF16 subnormals before multiplication.
    gamma_fp32 = _widen_bf16_bits(gamma)
    out = tl.inline_asm_elementwise(
        """{
        .reg .f32 normalized;
        mul.rn.f32 normalized, $1, $2;
        mul.rn.f32 $0, normalized, $3;
        }""",
        "=f,f,f,f",
        [x, inv, gamma_fp32],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(Out + row * D + col, out.to(tl.bfloat16), mask)


def _rmsnorm_epilogue(x, inv, gamma_bf16, *, x_bf16):
    out = torch.empty_like(x, dtype=torch.bfloat16)
    if out.numel():
        dim = x.shape[-1]
        block = min(1024, triton.next_power_of_2(dim))
        _prefill_rmsnorm_epilogue_kernel[(inv.numel(), triton.cdiv(dim, block))](
            x,
            inv,
            gamma_bf16,
            out,
            D=dim,
            BLOCK=block,
            X_BF16=x_bf16,
            enable_fp_fusion=False,
        )
    return out


def prefill_rmsnorm_epilogue(
    x_fp32: torch.Tensor, inv: torch.Tensor, gamma_bf16: torch.Tensor
) -> torch.Tensor:
    """Scale FP32 rows by inverse RMS and BF16 gamma with separate FP32 rounds."""
    return _rmsnorm_epilogue(x_fp32, inv, gamma_bf16, x_bf16=False)


def verify_rmsnorm_bf16(
    x: torch.Tensor, gamma: torch.Tensor, eps: float
) -> torch.Tensor:
    """Retain the ATen reduction while keeping input activations in BF16."""
    square = _rmsnorm_bf16_square(x)
    inv = torch.rsqrt(square.mean(-1, keepdim=True) + eps)
    return _rmsnorm_epilogue(x, inv, gamma, x_bf16=True)


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


def try_project_qr_kv(linear, x, q_norm, q_rank: int, eps: float, quantized_input=None):
    """Return normalized Q LoRA and raw KV, or None for the old path."""
    if not is_supported(linear, x, q_norm, q_rank):
        return None
    if quantized_input is None:
        projected = linear(x)
    else:
        x_q, x_s = quantized_input
        projected = linear.forward_quantized(x_q, x_s).view(*x.shape[:-1], linear.N)
    qr = strided_q_rmsnorm(projected[..., :q_rank], q_norm, eps)
    # KV's existing RMSNorm/RoPE kernel takes an explicit row stride.
    return qr, projected[..., q_rank:]
