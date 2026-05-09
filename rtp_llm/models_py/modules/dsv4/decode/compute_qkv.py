"""DSv4 decode Q/KV projection — extracted from ``AttentionBF16VLLM._forward_decode_body``.

Mirrors the per-request batched (``q_len == 1``) Q/KV path of the
monolithic decode body. The Attention module is passed in as ``attn`` and
treated as a bag of weights + tiny helpers (``_lin`` / ``_rmsnorm_weighted``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.dsv4.attention_bf16_vllm import (
        AttentionBF16VLLM as Attention,
    )


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
    attn: "Attention",
    x: torch.Tensor,
    start_pos: torch.Tensor,
    dbg_tag: str = "",
) -> DecodeQKV:
    """Decode Q/KV path — RMSNorm + LoRA Q + KV linear + fused RMSNorm-RoPE.

    Per-request batched: each row of ``x`` has its own ``start_pos`` so the
    RoPE table is gathered via ``freqs_cis[start_pos.long()]``;
    ``fused_rmsnorm_rope`` applies the partial RoPE in-kernel. KV is a single
    MQA head.

    ``dbg_tag`` — optional ``record_function`` prefix (per-layer tag used by
    the no-graph profiler); empty string disables tagging.
    """
    rd = attn.rope_head_dim
    freqs_cis_per_req = attn.freqs_cis[start_pos.long()]

    with torch.profiler.record_function(f"{dbg_tag}/q_proj" if dbg_tag else "q_proj"):
        qr = attn._rmsnorm_weighted(attn._lin(attn.wq_a, x), attn.q_norm)
        q = attn._lin(attn.wq_b, qr).unflatten(-1, (attn.n_heads, attn.head_dim))
        q = fused_rmsnorm_rope(q, None, freqs_cis_per_req, rd, eps=attn.eps)

    with torch.profiler.record_function(f"{dbg_tag}/kv_proj" if dbg_tag else "kv_proj"):
        kv = fused_rmsnorm_rope(
            attn._lin(attn.wkv, x),
            attn.kv_norm,
            freqs_cis_per_req,
            rd,
            eps=attn.eps,
        )

    return DecodeQKV(qr=qr, q=q, kv=kv, freqs_cis_per_req=freqs_cis_per_req)
