"""Fused Triton merge for LSA partial attentions (SWA/SFA and split-KV parts).

The torch-level merge (:func:`...swa_dense_triton.merge_attention`) walks the
full ``[T, H, DV]`` partial tensors several times and dominated the composed
prefill latency (0.526 ms on B300 for t2048). This module fuses the same
online-softmax rescaling into a single read+write pass:

    M      = max_i lse_i
    w_i    = exp(lse_i - M)        (0 where lse_i == -inf)
    out    = sum_i o_i * w_i / sum_i w_i
    lse    = M + log(sum_i w_i)

Parts must be *normalized* partials (``out_i`` already divided by ``l_i``), which
is what both :func:`swa_dense_fwd` and :func:`sparse_mla_fwd` return. The merge
is associative, so N parts reduce through ``N - 1`` pairwise kernel launches.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

_NEG_INF = float("-inf")

_OUT_DTYPES = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


@triton.jit
def _merge2_kernel(
    O0,
    L0,
    O1,
    L1,
    OUT,
    LSE,
    T,
    H: tl.constexpr,
    DV: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One ``(token, head)`` row per program over the whole ``DV`` width."""
    pid = tl.program_id(0)
    t = pid // H
    head = pid % H
    row = t * H + head

    l0 = tl.load(L0 + row).to(tl.float32)
    l1 = tl.load(L1 + row).to(tl.float32)
    m = tl.maximum(l0, l1)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    w0 = tl.where(l0 == float("-inf"), 0.0, tl.exp(l0 - m_safe))
    w1 = tl.where(l1 == float("-inf"), 0.0, tl.exp(l1 - m_safe))
    denom = w0 + w1
    denom_safe = tl.where(denom > 0, denom, 1.0)

    offs = tl.arange(0, DV)
    o0 = tl.load(O0 + row * DV + offs).to(tl.float32)
    o1 = tl.load(O1 + row * DV + offs).to(tl.float32)
    merged = (o0 * w0 + o1 * w1) / denom_safe
    tl.store(OUT + row * DV + offs, merged.to(OUT_DTYPE))

    lse = tl.where(denom > 0, m + tl.log(denom_safe), float("-inf"))
    tl.store(LSE + row, lse)


def merge_attention_triton(
    parts: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    out_dtype: Optional[torch.dtype] = None,
    num_warps: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge normalized partial attentions with a fused pairwise kernel.

    Args:
        parts: sequence of ``(out_i [T, H, DV], lse_i [T, H])`` normalized
            partial results over disjoint slot sets of the same query rows.
        out_dtype: output dtype; defaults to ``parts[0][0].dtype``.
        num_warps: warps per merge program.

    Returns:
        ``(out, lse)`` with the same semantics as ``merge_attention``.
    """
    if not parts:
        raise ValueError("merge_attention_triton requires at least one part")
    outs = [part[0].contiguous() for part in parts]
    lses = [part[1].contiguous().float() for part in parts]
    t_count, heads, dv = outs[0].shape
    for out_i, lse_i in zip(outs, lses):
        if out_i.shape != (t_count, heads, dv):
            raise ValueError("all partial outputs must share one shape")
        if lse_i.shape != (t_count, heads):
            raise ValueError(f"partial lse must be [T, H], got {tuple(lse_i.shape)}")
    if dv & (dv - 1):
        raise ValueError("DV must be a power of two for the fused merge")
    resolved_out_dtype = out_dtype or outs[0].dtype
    if resolved_out_dtype not in _OUT_DTYPES:
        raise ValueError(f"unsupported out_dtype {resolved_out_dtype}")

    current_out, current_lse = outs[0], lses[0]
    for next_out, next_lse in zip(outs[1:], lses[1:]):
        merged_out = torch.empty(
            (t_count, heads, dv), device=current_out.device, dtype=resolved_out_dtype
        )
        merged_lse = torch.empty(
            (t_count, heads), device=current_out.device, dtype=torch.float32
        )
        _merge2_kernel[(t_count * heads,)](
            current_out,
            current_lse,
            next_out,
            next_lse,
            merged_out,
            merged_lse,
            t_count,
            H=heads,
            DV=dv,
            OUT_DTYPE=_OUT_DTYPES[resolved_out_dtype],
            num_warps=num_warps,
        )
        current_out, current_lse = merged_out, merged_lse
    return current_out, current_lse