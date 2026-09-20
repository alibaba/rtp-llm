"""DSv4 decode Q/KV projection — extracted from ``AttentionFP8._forward_decode_body``.

Mirrors ``AttentionFP8._prefill_compute_qkv`` for the decode path
(per-request batched, including target-verify ``q_len > 1``).

The Attention module is passed in as ``attn`` and treated as a bag of
weights + tiny helpers (``_lin`` / ``_rmsnorm_weighted``) — this matches
the existing free-function pattern in :mod:`.decode_attn_metadata`, and
keeps ``attention.py`` thin.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, NamedTuple

import torch

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope
from rtp_llm.models_py.modules.dsv4._rope_only_triton import rope_only_inplace
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb

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


def _fused_v41_q_rope_supported(q, freqs_cis, rope_dim: int) -> bool:
    """Shape-only gate: replay keeps positions/frequencies on the device."""
    return (
        os.environ.get("DSV41_FUSED_DECODE_Q_ROPE", "1") == "1"
        and q.is_cuda
        and q.dtype == torch.bfloat16
        and q.ndim == 4
        and q.shape[-1] == 512
        and rope_dim == 64
        and q.is_contiguous()
        and freqs_cis.device == q.device
        and freqs_cis.dtype == torch.complex64
        and freqs_cis.shape == (q.shape[0] * q.shape[1], rope_dim // 2)
    )


def _apply_v41_q_rope(q, freqs_cis, rope_dim: int) -> None:
    # V4.1 intentionally omits the post-projection Q norm. Preserve the
    # in-place BF16 tail store; the non-RoPE part must remain untouched.
    if _fused_v41_q_rope_supported(q, freqs_cis, rope_dim):
        rope_only_inplace(q[..., -rope_dim:], freqs_cis)
    else:
        apply_rotary_emb(q[..., -rope_dim:], freqs_cis)


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

    fused_qkv = attn._try_fused_qr_kv(x)
    # Q path; fused KV remains a view until its strided RMSNorm/RoPE below.
    if fused_qkv is None:
        qr = attn._rmsnorm_weighted(attn._lin(attn.wq_a, x), attn.q_norm)
    else:
        qr = fused_qkv[0]
    q = attn._lin(attn.wq_b, qr).unflatten(
        -1, (attn.n_heads, attn.head_dim)
    )  # [B, S, H, D]
    if getattr(attn, "skip_post_q_norm", False):
        _apply_v41_q_rope(q, freqs_cis, rd)
    else:
        q = fused_rmsnorm_rope(q, None, freqs_cis, rd, eps=attn.eps)

    # KV path (single MQA head) — per-token RoPE using the same table lookup.
    kv = fused_rmsnorm_rope(
        attn._lin(attn.wkv, x) if fused_qkv is None else fused_qkv[1],
        attn.kv_norm,
        freqs_cis,
        rd,
        eps=attn.eps,
    )

    return DecodeQKV(qr=qr, q=q, kv=kv, freqs_cis=freqs_cis)
