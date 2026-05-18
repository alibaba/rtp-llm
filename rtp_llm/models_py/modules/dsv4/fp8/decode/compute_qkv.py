"""DSv4 decode Q/KV projection — extracted from ``AttentionFP8._forward_decode_body``.

Mirrors ``AttentionFP8._prefill_compute_qkv`` for the decode path
(per-request batched, including target-verify ``q_len > 1``).

The Attention module is passed in as ``attn`` and treated as a bag of
weights + tiny helpers (``_lin`` / ``_rmsnorm_weighted``) — this matches
the existing free-function pattern in :mod:`.decode_attn_metadata`, and
keeps ``attention.py`` thin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope

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

    qr: torch.Tensor
    q: torch.Tensor
    kv: torch.Tensor
    freqs_cis: torch.Tensor


def decode_compute_qkv(
    attn: "AttentionFP8",
    x: torch.Tensor,  # [B, S, dim] bf16
    position_ids: torch.Tensor,  # [T] int32 absolute position per token
) -> DecodeQKV:
    """Decode Q/KV path — RMSNorm + LoRA Q + KV linear + fused RMSNorm-RoPE.

    ``position_ids`` is flat over the token-major ``[B, S]`` layout. For
    normal decode ``S == 1``; target verify passes the full verify span.
    """
    rd = attn.rope_head_dim
    position_ids = position_ids.reshape(-1).to(
        device=attn.freqs_cis.device, dtype=torch.long
    )
    freqs_cis = attn.freqs_cis.index_select(0, position_ids).contiguous()

    # Q path
    qr = attn._rmsnorm_weighted(
        attn._lin(attn.wq_a, x), attn.q_norm
    )  # [B, 1, q_lora_rank]
    q_linear = attn._lin(attn.wq_b, qr).unflatten(
        -1, (attn.n_heads, attn.head_dim)
    )  # [B, S, H, D]
    if getattr(attn, "_debug_attn_dump_enabled", False):
        attn._debug_q_linear[: x.size(0)].copy_(q_linear.to(torch.bfloat16))
    q = fused_rmsnorm_rope(q_linear, None, freqs_cis, rd, eps=attn.eps)

    # KV path (single MQA head) — per-token RoPE using the same table lookup.
    kv_linear = attn._lin(attn.wkv, x)
    if getattr(attn, "_debug_attn_dump_enabled", False):
        attn._debug_kv_linear[: x.size(0)].copy_(kv_linear.to(torch.bfloat16))
    kv = fused_rmsnorm_rope(
        kv_linear,
        attn.kv_norm,
        freqs_cis,
        rd,
        eps=attn.eps,
    )

    return DecodeQKV(qr=qr, q=q, kv=kv, freqs_cis=freqs_cis)
