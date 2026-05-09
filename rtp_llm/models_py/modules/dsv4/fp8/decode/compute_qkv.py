"""DSv4 decode Q/KV projection — extracted from ``AttentionFP8._forward_decode_body``.

Mirrors ``AttentionFP8._prefill_compute_qkv`` for the decode path
(per-request batched, ``q_len == 1``).

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

    ``qr``  — ``[B, 1, q_lora_rank]`` bf16 — fed to the CSA indexer.
    ``q``   — ``[B, 1, H, D]`` bf16 — dense Q for sparse attn.
    ``kv``  — ``[B, 1, D]`` bf16 — single MQA head, written to SWA pool.
    ``freqs_cis_per_req`` — ``[B, freqs_dim]`` — per-request RoPE table
        lookup, reused by the output-proj inverse-RoPE.
    """

    qr: torch.Tensor
    q: torch.Tensor
    kv: torch.Tensor
    freqs_cis_per_req: torch.Tensor


def decode_compute_qkv(
    attn: "AttentionFP8",
    x: torch.Tensor,  # [B, 1, dim] bf16
    start_pos: torch.Tensor,  # [B] int32 per-request absolute position
) -> DecodeQKV:
    """Decode Q/KV path — RMSNorm + LoRA Q + KV linear + fused RMSNorm-RoPE.

    Per-request batched: each row of ``x`` has its own ``start_pos`` so the
    RoPE table is gathered via ``freqs_cis[start_pos.long()]``;
    ``fused_rmsnorm_rope`` applies the partial RoPE in-kernel. KV is a
    single MQA head.
    """
    rd = attn.rope_head_dim
    freqs_cis_per_req = attn.freqs_cis[start_pos.long()]  # [B, freqs_dim]

    # Q path
    qr = attn._rmsnorm_weighted(
        attn._lin(attn.wq_a, x), attn.q_norm
    )  # [B, 1, q_lora_rank]
    q = attn._lin(attn.wq_b, qr).unflatten(
        -1, (attn.n_heads, attn.head_dim)
    )  # [B, 1, H, D]
    q = fused_rmsnorm_rope(q, None, freqs_cis_per_req, rd, eps=attn.eps)

    # KV path (single MQA head) — per-request RoPE using the same table lookup.
    kv = fused_rmsnorm_rope(
        attn._lin(attn.wkv, x),
        attn.kv_norm,
        freqs_cis_per_req,
        rd,
        eps=attn.eps,
    )

    return DecodeQKV(qr=qr, q=q, kv=kv, freqs_cis_per_req=freqs_cis_per_req)
