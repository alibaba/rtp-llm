"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.modules.dsv4.fp8.decode.compute_qkv import (
    DecodeQKV,
)
from typing import TYPE_CHECKING, NamedTuple
import torch
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.attention import AttentionFP8

def decode_compute_qkv(
    attn: "AttentionFP8",
    x: torch.Tensor,  # [B, S, dim] bf16
    position_ids: torch.Tensor,  # [T] int32 absolute position per token
) -> DecodeQKV:
    """Decode Q/KV path — RMSNorm + LoRA Q + KV linear + fused RMSNorm-RoPE.

    ``position_ids`` is flat over the token-major ``[B, S]`` layout. For
    normal decode ``S == 1``; target verify passes the full verify span.
    """
    freqs_cis = decode_select_freqs(attn, position_ids)
    qr = decode_compute_q_a(attn, x)
    q = decode_compute_q_b(attn, qr, freqs_cis)
    kv = decode_compute_kv(attn, x, freqs_cis)
    return DecodeQKV(qr=qr, q=q, kv=kv, freqs_cis=freqs_cis)

def decode_select_freqs(attn, position_ids):
    position_ids = position_ids.reshape(-1).to(
        device=attn.freqs_cis.device, dtype=torch.long
    )
    return attn.freqs_cis.index_select(0, position_ids).contiguous()

def decode_compute_q_a(attn, x):
    return attn._rmsnorm_weighted(
        attn._lin(attn.wq_a, x), attn.q_norm
    )  # [B, 1, q_lora_rank]

def decode_compute_q_b(attn, qr, freqs_cis):
    q = attn._lin(attn.wq_b, qr).unflatten(
        -1, (attn.n_heads, attn.head_dim)
    )  # [B, S, H, D]
    return fused_rmsnorm_rope(q, None, freqs_cis, attn.rope_head_dim, eps=attn.eps)

def decode_compute_kv(attn, x, freqs_cis):
    return fused_rmsnorm_rope(
        attn._lin(attn.wkv, x),
        attn.kv_norm,
        freqs_cis,
        attn.rope_head_dim,
        eps=attn.eps,
    )
