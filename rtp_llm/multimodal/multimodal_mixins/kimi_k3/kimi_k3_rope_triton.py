"""Fused 2D RoPE for Kimi-K3 MoonViT."""

import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the runtime image
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_FUSED_ROPE_ENV = "KIMI_K3_FUSED_ROPE"
# Best launch geometry for K3 MoonViT's 12-head, 128-dimension attention.
_GROUP_HEADS = 2


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_qk_rope_kernel(
        q_ptr,
        k_ptr,
        freqs_ptr,
        q_out_ptr,
        k_out_ptr,
        q_stride_s,
        q_stride_h,
        q_stride_d,
        k_stride_s,
        k_stride_h,
        k_stride_d,
        freqs_stride_s,
        freqs_stride_pair,
        freqs_stride_component,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        GROUP_HEADS: tl.constexpr,
        BLOCK_PAIRS: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        head_block = tl.program_id(1).to(tl.int64)
        heads = head_block * GROUP_HEADS + tl.arange(0, GROUP_HEADS)
        pairs = tl.arange(0, BLOCK_PAIRS)
        head_offsets = heads[:, None]
        pair_offsets = pairs[None, :]
        mask = (head_offsets < NUM_HEADS) & (pair_offsets < HEAD_DIM // 2)

        q_offsets = (
            token * q_stride_s
            + head_offsets * q_stride_h
            + pair_offsets * 2 * q_stride_d
        )
        k_offsets = (
            token * k_stride_s
            + head_offsets * k_stride_h
            + pair_offsets * 2 * k_stride_d
        )
        freq_offsets = (
            token * freqs_stride_s + pair_offsets * freqs_stride_pair
        )

        cos = tl.load(freqs_ptr + freq_offsets, mask=mask, other=0.0)
        sin = tl.load(
            freqs_ptr + freq_offsets + freqs_stride_component,
            mask=mask,
            other=0.0,
        )
        q_real = tl.load(q_ptr + q_offsets, mask=mask, other=0.0).to(tl.float32)
        q_imag = tl.load(
            q_ptr + q_offsets + q_stride_d, mask=mask, other=0.0
        ).to(tl.float32)
        k_real = tl.load(k_ptr + k_offsets, mask=mask, other=0.0).to(tl.float32)
        k_imag = tl.load(
            k_ptr + k_offsets + k_stride_d, mask=mask, other=0.0
        ).to(tl.float32)

        out_offsets = (
            token * NUM_HEADS * HEAD_DIM
            + head_offsets * HEAD_DIM
            + pair_offsets * 2
        )
        tl.store(q_out_ptr + out_offsets, q_real * cos - q_imag * sin, mask=mask)
        tl.store(
            q_out_ptr + out_offsets + 1,
            q_real * sin + q_imag * cos,
            mask=mask,
        )
        tl.store(k_out_ptr + out_offsets, k_real * cos - k_imag * sin, mask=mask)
        tl.store(
            k_out_ptr + out_offsets + 1,
            k_real * sin + k_imag * cos,
            mask=mask,
        )


def _is_supported(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> bool:
    return bool(
        _TRITON_AVAILABLE
        and xq.is_cuda
        and xk.is_cuda
        and freqs_cis.is_cuda
        and xq.device == xk.device == freqs_cis.device
        and xq.ndim == 3
        and xq.shape == xk.shape
        and xq.dtype == xk.dtype
        and xq.dtype in (torch.float16, torch.bfloat16)
        and xq.stride(-1) == 1
        and xk.stride(-1) == 1
        and xq.shape[-1] % 2 == 0
        and xq.shape[-1] <= 256
        and freqs_cis.dtype == torch.complex64
        and freqs_cis.shape == (xq.shape[0], xq.shape[-1] // 2)
        and freqs_cis.is_contiguous()
        and not freqs_cis.is_conj()
        and not xq.requires_grad
        and not xk.requires_grad
    )


def maybe_fused_apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Apply fused Q/K RoPE using one allocation for both contiguous outputs."""
    if os.environ.get(_FUSED_ROPE_ENV, "0") != "1":
        return None
    if not _is_supported(xq, xk, freqs_cis):
        return None

    qk_out = torch.empty((2, *xq.shape), dtype=xq.dtype, device=xq.device)
    q_out, k_out = qk_out.unbind(0)
    if xq.numel() == 0:
        return q_out, k_out

    freqs = torch.view_as_real(freqs_cis)
    head_dim = xq.shape[-1]
    block_pairs = triton.next_power_of_2(head_dim // 2)
    grid = (xq.shape[0], triton.cdiv(xq.shape[1], _GROUP_HEADS))
    _fused_qk_rope_kernel[grid](
        xq,
        xk,
        freqs,
        q_out,
        k_out,
        xq.stride(0),
        xq.stride(1),
        xq.stride(2),
        xk.stride(0),
        xk.stride(1),
        xk.stride(2),
        freqs.stride(0),
        freqs.stride(1),
        freqs.stride(2),
        NUM_HEADS=xq.shape[1],
        HEAD_DIM=head_dim,
        GROUP_HEADS=_GROUP_HEADS,
        BLOCK_PAIRS=block_pairs,
        num_warps=4,
    )
    return q_out, k_out
