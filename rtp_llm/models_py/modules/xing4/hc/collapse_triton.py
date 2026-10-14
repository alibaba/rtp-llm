"""Fused Triton kernel for the Xing4.0 hyper-connection stream collapse.

Reference math (authority, do not change):
``rtp_llm/models_py/model_desc/xing4_0_model.py`` ``Xing4HCUnit._pre_impl``::

    y = torch.sum(pre.to(dtype).unsqueeze(-1) * x, dim=-2)   # [T, hc, H] -> [T, H]

This module fuses that readout into a single Triton launch:

  * ``pre`` is cast to ``x.dtype`` before the product (exactly the
    reference's ``pre.to(dtype)``);
  * one program per token row (``grid=(T,)``) walks ``H`` in ``BLOCK_H``
    chunks, so the production ``H=3584`` never needs an oversized block;
  * the ``hc`` stream products are accumulated in fp32 and each row is
    stored once in ``x.dtype``.

Numerical contract vs the reference: the eager path materializes the
broadcast product ``pre.to(dtype).unsqueeze(-1) * x`` in bf16 -- each
per-stream product is rounded to bf16 on store -- and then
``torch.sum(dim=-2)`` accumulates those rounded products in fp32. This
kernel multiplies and accumulates in fp32 directly, skipping the
per-product bf16 landing. The structural difference is at most 1 bf16 ulp
per summed term (4 terms for hc=4), which the equivalence UT
``rtp_llm/models_py/modules/xing4/test/test_hc_collapse_triton.py``
covers with ``rtol=atol=2**-7``. Products whose ``pre`` factor is a power
of two are bit-exact, because scaling a bf16 value by a power of two does
not change its mantissa.

Index arithmetic: the program id is cast to int64 *before* any stride
multiply. ``pid * x_stride_t`` indexes up to ``T * hc * H``, which silently
overflows int32 for large token counts (at ``H=3584, hc=4`` that happens
around ``T ~ 150k``, and non-contiguous stride variants inflate the
strides further) -- same pitfall as
``modules/dsv4/_fused_rmsnorm_rope_triton.py``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["hc_collapse"]

_MAX_BLOCK_H = 1024
_DEFAULT_NUM_WARPS = 4


@triton.jit
def _hc_collapse_kernel(
    pre_ptr,
    x_ptr,
    out_ptr,
    pre_stride_t,
    pre_stride_hc,
    x_stride_t,
    x_stride_hc,
    x_stride_h,
    out_stride_t,
    out_stride_h,
    H,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # int64 before the row-stride multiply: pid * x_stride_t reaches
    # T * hc * H and overflows int32 for large T (see module docstring).
    pid = tl.program_id(0).to(tl.int64)
    i_off = tl.arange(0, HC).to(tl.int64)

    pre_row = pre_ptr + pid * pre_stride_t + i_off * pre_stride_hc
    p = tl.load(pre_row).to(tl.float32)

    x_row = x_ptr + pid * x_stride_t
    out_row = out_ptr + pid * out_stride_t
    for h_start in range(0, H, BLOCK_H):
        h_off = (h_start + tl.arange(0, BLOCK_H)).to(tl.int64)
        h_mask = h_off < H
        x_ptrs = (
            x_row
            + i_off[:, None] * x_stride_hc
            + h_off[None, :] * x_stride_h
        )
        x = tl.load(x_ptrs, mask=h_mask[None, :], other=0.0).to(tl.float32)
        acc = tl.sum(x * p[:, None], axis=0)
        tl.store(
            out_row + h_off * out_stride_h,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )


def hc_collapse(
    pre: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_h: int | None = None,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """Collapse the HC streams: ``y[..., h] = sum_i pre[..., i] * x[..., i, h]``.

    Args:
        pre: mixer weights ``[..., HC]``. Cast to ``x.dtype`` before the
            product (reference semantics); fp32 input is accepted and cast.
        x: residual streams ``[..., HC, H]`` (bf16 in production; fp16/fp32
            also work). Arbitrary strides are supported for the flat 3D
            ``[T, HC, H]`` layout; higher-rank inputs must be contiguous
            and are flattened to token rows.
        out: optional preallocated output ``[..., H]``; must be contiguous
            with matching dtype/device. Allocated when omitted.
        block_h: override the ``H`` chunk size (power of two); defaults to
            ``min(next_power_of_2(H), 1024)``.
        num_warps: Triton launch parameter.

    Returns:
        ``y`` with shape ``x.shape[:-2] + (H,)`` in ``x.dtype``.
    """
    if x.dim() < 3:
        raise ValueError(
            f"hc_collapse expected x [..., hc, H] with dim >= 3, "
            f"got shape={tuple(x.shape)}"
        )
    if pre.dim() < 1 or pre.shape[-1] != x.shape[-2]:
        raise ValueError(
            f"hc_collapse expected pre trailing [hc]={tuple(x.shape[-2:-1])}, "
            f"got pre shape={tuple(pre.shape)}"
        )
    if not x.is_cuda:
        raise ValueError(f"hc_collapse requires CUDA tensors, got device={x.device}")
    if pre.device != x.device:
        raise ValueError(
            f"hc_collapse device mismatch: pre={pre.device} vs x={x.device}"
        )
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"hc_collapse unsupported x dtype {x.dtype}")

    hc = int(x.shape[-2])
    H = int(x.shape[-1])
    if hc < 1 or (hc & (hc - 1)) != 0:
        # tl.arange(0, HC) requires a power-of-two size.
        raise ValueError(f"hc_collapse requires a power-of-two hc, got {hc}")

    expected_out_shape = tuple(x.shape[:-2]) + (H,)
    if out is None:
        out = torch.empty(expected_out_shape, dtype=x.dtype, device=x.device)
    else:
        if tuple(out.shape) != expected_out_shape:
            raise ValueError(
                f"hc_collapse out shape={tuple(out.shape)}, "
                f"expected {expected_out_shape}"
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise ValueError(
                f"hc_collapse out dtype/device {out.dtype}/{out.device}, "
                f"expected {x.dtype}/{x.device}"
            )
        if not out.is_contiguous():
            raise ValueError("hc_collapse out must be contiguous")

    if x.numel() == 0:
        return out

    if x.dim() == 3:
        x3 = x
    elif x.is_contiguous():
        x3 = x.view(-1, hc, H)
    else:
        raise ValueError(
            "hc_collapse non-contiguous x is only supported for the "
            "3D [T, hc, H] layout"
        )
    T = int(x3.shape[0])

    if pre.numel() != T * hc:
        raise ValueError(
            f"hc_collapse pre numel={pre.numel()}, expected T*hc={T * hc}"
        )
    if pre.dim() == 2:
        pre2 = pre
    elif pre.is_contiguous():
        pre2 = pre.view(-1, hc)
    else:
        raise ValueError(
            "hc_collapse non-contiguous pre is only supported for the "
            "2D [T, hc] layout"
        )
    if pre2.dtype != x.dtype:
        # Reference: pre.to(dtype) happens BEFORE the product.
        pre2 = pre2.to(x.dtype)

    out2 = out.view(-1, H)

    selected_block_h = (
        block_h
        if block_h is not None
        else min(triton.next_power_of_2(H), _MAX_BLOCK_H)
    )
    if selected_block_h < 1 or (selected_block_h & (selected_block_h - 1)) != 0:
        raise ValueError(
            f"hc_collapse block_h must be a power of two, got {selected_block_h}"
        )

    _hc_collapse_kernel[(T,)](
        pre2,
        x3,
        out2,
        pre2.stride(0),
        pre2.stride(1),
        x3.stride(0),
        x3.stride(1),
        x3.stride(2),
        out2.stride(0),
        out2.stride(1),
        H,
        HC=hc,
        BLOCK_H=selected_block_h,
        num_warps=num_warps,
    )
    return out
