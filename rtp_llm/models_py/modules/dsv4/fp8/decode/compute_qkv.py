"""DSv4 decode Q/KV projection — extracted from ``AttentionFP8._forward_decode_body``.

Mirrors ``AttentionFP8._prefill_compute_qkv`` for the decode path
(per-request batched, including target-verify ``q_len > 1``).

The Attention module is passed in as ``attn`` and treated as a bag of
weights + tiny helpers (``_lin`` / ``_rmsnorm_weighted``) — this matches
the existing free-function pattern in :mod:`.decode_attn_metadata`, and
keeps ``attention.py`` thin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
    fused_rmsnorm_rope,
)
from rtp_llm.models_py.modules.dsv4.fp8._fused_qnorm_quant_triton import (
    can_fuse_qnorm_fp8_ue8m0_quant,
    fused_rmsnorm_fp8_ue8m0_quant,
)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU/fallback environments
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _gather_freqs_cis_kernel(
        freqs_ptr,
        pos_ptr,
        out_ptr,
        stride_freq_row,
        stride_pos,
        n_rows,
        F: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        tok = tl.program_id(0)
        row = tl.load(pos_ptr + tok * stride_pos).to(tl.int64)
        row = tl.maximum(tl.minimum(row, n_rows - 1), 0)
        offs = tl.arange(0, BLOCK_F)
        mask = offs < F
        vals = tl.load(freqs_ptr + row * stride_freq_row + offs, mask=mask)
        tl.store(out_ptr + tok * F + offs, vals, mask=mask)


def _gather_freqs_cis_by_pos(
    freqs_table: torch.Tensor, pos: torch.Tensor
) -> torch.Tensor:
    """Gather ``freqs_table[pos]`` into contiguous ``[T, F]`` without int64 cast."""
    if (
        _HAS_TRITON
        and freqs_table.is_cuda
        and pos.is_cuda
        and freqs_table.dtype == torch.complex64
        and freqs_table.ndim == 2
        and freqs_table.stride(1) == 1
        and pos.numel() > 0
        and freqs_table.shape[0] > 0
    ):
        t_len = int(pos.numel())
        n_cols = int(freqs_table.shape[1])
        out = torch.empty(
            (t_len, n_cols), dtype=freqs_table.dtype, device=freqs_table.device
        )
        block_f = triton.next_power_of_2(n_cols)
        num_warps = 1 if block_f <= 32 else (2 if block_f <= 128 else 4)
        _gather_freqs_cis_kernel[(t_len,)](
            freqs_table.view(torch.int64),
            pos,
            out.view(torch.int64),
            int(freqs_table.stride(0)),
            int(pos.stride(0)),
            int(freqs_table.shape[0]),
            F=n_cols,
            BLOCK_F=block_f,
            num_warps=num_warps,
        )
        return out
    idx = pos if pos.dtype == torch.long else pos.to(torch.long)
    freqs_cis = freqs_table.index_select(0, idx)
    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()
    return freqs_cis


if TYPE_CHECKING:  # pragma: no cover - typing only
    from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8


class DecodeQKV(NamedTuple):
    """Q/KV intermediate produced by :func:`decode_compute_qkv`.

    ``qr``  — ``[B, S, q_lora_rank]`` bf16 — fed to the CSA indexer.
    ``q``   — ``[B, S, H, D]`` bf16 — dense Q for sparse attn.
    ``kv``  — ``[B, S, D]`` bf16 — single MQA head, written to SWA pool.
    ``freqs_cis`` — ``[T, freqs_dim]`` — per-token RoPE table
        lookup, reused by the output-proj inverse-RoPE.
    """

    qr: Optional[torch.Tensor]
    q: torch.Tensor
    kv: torch.Tensor
    freqs_cis: torch.Tensor


def decode_compute_qkv(
    attn: "AttentionFP8",
    x: torch.Tensor,  # [B, S, dim] bf16
    position_ids: torch.Tensor,  # [T] int32 absolute position per token
    return_freqs_cis: bool = True,
    return_qr: Optional[bool] = None,
) -> DecodeQKV:
    """Decode Q/KV path — RMSNorm + LoRA Q + KV linear + fused RMSNorm-RoPE.

    ``position_ids`` is flat over the token-major ``[B, S]`` layout. For
    normal decode ``S == 1``; target verify passes the full verify span.
    """
    rd = attn.rope_head_dim
    pos = position_ids.reshape(-1)
    if pos.device != attn.freqs_cis.device:
        pos = pos.to(device=attn.freqs_cis.device)
    if pos.dtype not in (torch.int32, torch.int64):
        pos = pos.to(dtype=torch.int32)

    attn_dict = getattr(attn, "__dict__", {})
    lin_overridden = "_lin" in attn_dict
    rmsnorm_overridden = "_rmsnorm_weighted" in attn_dict
    can_reuse_x_quant = (
        not lin_overridden
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.numel() > 0
        and callable(getattr(attn, "_can_reuse_qkv_input_quant", None))
        and attn._can_reuse_qkv_input_quant() is True
    )

    if can_reuse_x_quant:
        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        x_fp8, x_scale = attn.wq_a.quantize_input(x_2d)
        q_proj_2d = attn.wq_a.forward_quantized(x_fp8, x_scale)
        q_proj = q_proj_2d.reshape(*x.shape[:-1], -1)
        kv_in = attn.wkv.forward_quantized(x_fp8, x_scale).reshape(*x.shape[:-1], -1)
    else:
        q_proj = attn._lin(attn.wq_a, x)
        q_proj_2d = q_proj.reshape(-1, q_proj.shape[-1])
        kv_in = attn._lin(attn.wkv, x)

    wq_b = getattr(attn, "wq_b", None)
    can_fuse_q = (
        not lin_overridden
        and not rmsnorm_overridden
        and callable(getattr(wq_b, "forward_quantized", None))
        and getattr(wq_b, "scale_ue8m0", False) is True
        and can_fuse_qnorm_fp8_ue8m0_quant(q_proj_2d, attn.q_norm, group_size=128)
    )
    need_qr = (
        bool(return_qr)
        if return_qr is not None
        else (getattr(attn, "compress_ratio", 4) == 4)
    )

    # Q path: fuse q_norm + wq_b FP8 UE8M0 quantization, skipping BF16 qr emit
    # unless the caller (CSA indexer) consumes qkv.qr.
    if can_fuse_q:
        out_norm = (
            q_proj_2d
            if (need_qr and q_proj_2d.is_contiguous() and q_proj_2d.numel() > 0)
            else None
        )
        qr_2d, (qr_fp8, qr_scale) = fused_rmsnorm_fp8_ue8m0_quant(
            q_proj_2d,
            attn.q_norm,
            eps=attn.eps,
            group_size=128,
            emit_norm=need_qr,
            out_norm=out_norm,
        )
        qr = qr_2d.reshape(*x.shape[:-1], -1) if qr_2d is not None else None
        q = wq_b.forward_quantized(qr_fp8, qr_scale).reshape(
            *x.shape[:-1], attn.n_heads, attn.head_dim
        )
    else:
        qr = attn._rmsnorm_weighted(q_proj, attn.q_norm)  # [B, S, q_lora_rank]
        q = attn._lin(attn.wq_b, qr).unflatten(
            -1, (attn.n_heads, attn.head_dim)
        )  # [B, S, H, D]

    q_inplace = bool(q.is_cuda and q.is_contiguous() and q.numel() > 0)
    q = fused_rmsnorm_rope(
        q,
        None,
        attn.freqs_cis,
        rd,
        eps=attn.eps,
        position_ids=pos,
        inplace=q_inplace,
    )

    # KV path (single MQA head) — per-token RoPE using inline table lookup.
    kv_inplace = bool(kv_in.is_cuda and kv_in.is_contiguous() and kv_in.numel() > 0)
    kv = fused_rmsnorm_rope(
        kv_in,
        attn.kv_norm,
        attn.freqs_cis,
        rd,
        eps=attn.eps,
        position_ids=pos,
        inplace=kv_inplace,
    )

    if return_freqs_cis:
        freqs_cis = _gather_freqs_cis_by_pos(attn.freqs_cis, pos)
    else:
        freqs_cis = attn.freqs_cis

    return DecodeQKV(qr=qr, q=q, kv=kv, freqs_cis=freqs_cis)
