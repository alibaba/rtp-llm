"""Fused hc-mean + final RMSNorm Triton kernel for the Xing4.0 model tail.

The mathematical authority is ``Xing4Model._capture_canonical_final`` in
``rtp_llm/models_py/model_desc/xing4_0_model.py`` (read-only; this module must
stay equivalent to it):

    m = x.mean(dim=-2)          # [T, H] bf16, fp32 accumulation (HC streams)
    y = rmsnorm(m, w, eps)      # flashinfer RMSNorm (fp32 internal, bf16 out)

The eager chain lowers to a mean kernel plus the norm kernel and materializes
the [T, H] intermediate; this module fuses both into one row kernel that
reads x exactly once:

    m_bf16 = bf16((sum_s fp32(x[t, s, h])) / HC)     # torch-mean-shaped
    inv    = rsqrt(sum_h(fp32(m_bf16)**2) / H + eps)
    out    = bf16(fp32(m_bf16) * inv * fp32(w))

Design points:

* Single-read discipline: the whole H-wide row (BLOCK_H = next_pow_2(H)) stays
  in registers across the two phases — the mean is rounded to bf16 in-register
  and reused for the square-sum and the output multiply. No second x pass, no
  intermediate tensor. H is bounded to 8192 lanes (Xing4 H=3584) so the row
  fits one block; larger H falls back to the torch chain.
* The mean is rounded to bf16 BEFORE the norm reads it, matching the reference
  chain bit-for-bit (flashinfer norms the bf16 mean, not the fp32 one).
* Index-arithmetic contract (same pitfall as ``hc/norm_triton.py``): the
  program id is cast to int64 BEFORE the row-stride multiply; row * (HC*H)
  exceeds INT32_MAX from large T on and int32 offsets would silently wrap.
* Equivalence is tolerance-level, not bitwise: the fp32 reduction orders of
  the 4-stream sum and the H-wide square-sum differ from torch/flashinfer, so
  rare elements can differ by 1 bf16 ulp — the same equivalence class the K1
  lane measured for every fused candidate. The UT gates with the family
  discipline (bit-exact rate + ulp bound + fp32 anchor).

An alternative backend selected with ``XING4_MEAN_NORM_IMPL=torch`` keeps the
plain chain. Both backends are covered by
``rtp_llm/models_py/modules/xing4/test/test_mean_norm_triton.py``.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

_IMPL_ENV = "XING4_MEAN_NORM_IMPL"
_MAX_BLOCK_H = 8192


@triton.jit
def _mean_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    eps,
    x_stride_t,
    out_stride_t,
    HC: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Cast pid to int64 BEFORE the row-stride multiply: row * (HC*H) exceeds
    # INT32_MAX from large T on and int32 offsets would silently wrap.
    row = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, BLOCK_H)
    mask = offs_h < H

    # Phase 1: fp32 sum over the HC streams, block-resident (single read).
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for s in tl.static_range(HC):
        xs = tl.load(
            x_ptr + row * x_stride_t + s * H + offs_h, mask=mask, other=0.0
        ).to(tl.float32)
        acc += xs
    # Round the mean to bf16 BEFORE the norm phase: the reference chain
    # (torch.mean -> flashinfer rmsnorm) norms the bf16 mean.
    m = (acc / HC).to(tl.bfloat16).to(tl.float32)

    # Phase 2: RMSNorm from the register-resident mean.
    ss = tl.sum(m * m, axis=0) / H
    inv = tl.math.rsqrt(ss + eps)
    w = tl.load(w_ptr + offs_h, mask=mask, other=0.0).to(tl.float32)
    y = (m * inv * w).to(tl.bfloat16)
    tl.store(out_ptr + row * out_stride_t + offs_h, y, mask=mask)


def _can_fuse(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.dim() == 3
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[-1] <= _MAX_BLOCK_H
        and x.shape[-1] > 0
        and 0 < x.shape[1] <= 16
        and weight.dim() == 1
        and weight.dtype == torch.bfloat16
        and weight.is_cuda
        and weight.is_contiguous()
        and weight.shape == x.shape[-1:]
    )


def fused_mean_rmsnorm_triton(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Fused ``rmsnorm(x.mean(dim=-2), weight, eps)`` for [T, HC, H] bf16."""
    if not _can_fuse(x, weight):
        raise ValueError(
            f"fused_mean_rmsnorm_triton expects contiguous CUDA bf16 "
            f"[T, HC<=16, H<=8192] x with matching [H] weight, got "
            f"{tuple(x.shape)} {x.dtype}"
        )
    t, hc, h = x.shape
    out = torch.empty((t, h), device=x.device, dtype=torch.bfloat16)
    if t == 0:
        return out
    block_h = triton.next_power_of_2(h)
    _mean_rmsnorm_kernel[(t,)](
        x,
        weight,
        out,
        eps,
        x.stride(0),
        out.stride(0),
        HC=hc,
        H=h,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=1,
    )
    return out


def _torch_chain(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    import flashinfer

    return flashinfer.norm.rmsnorm(x.mean(dim=-2), weight, eps=eps)


def mean_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Env-dispatched entry: ``XING4_MEAN_NORM_IMPL`` in {"triton", "torch"}.

    Falls back to the torch chain whenever the fused kernel's contract is not
    met, so callers can hand over the layer tail unconditionally.
    """
    impl = os.environ.get(_IMPL_ENV, "triton").strip().lower()
    if impl == "triton" and _can_fuse(x, weight):
        return fused_mean_rmsnorm_triton(x, weight, eps)
    return _torch_chain(x, weight, eps)
