"""DSv4 decode output projection — extracted from ``Attention._forward_decode_body``.

Mirrors :meth:`rtp_llm.models_py.modules.dsv4.attention.Attention._prefill_output_proj`
for the decode path (per-request batched ``[B, q_len, H, D]``).

Pipeline:
    1. Inverse partial RoPE (unrotate the ``[rope_head_dim:]`` slice).
    2. Per-group per-token FP8 quant (UE8M0 scale layout).
    3. ``wo_a`` grouped einsum via ``deep_gemm.fp8_einsum``.
    4. ``wo_b`` linear.
    5. Optional TP ``all_reduce``.

The CUDA fast path fuses steps 1+2 into a single Triton kernel
(``fused_inv_rope_fp8_quant``). The eager fallback (CPU / empty) mirrors
the original per-op path, kept for warmup / unit tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4.qlinear import _fp8_dequant_to_fp32
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8


def decode_output_proj(
    attn: "AttentionFP8",
    o: torch.Tensor,  # [B, q_len, H, D]
    freqs_cis: torch.Tensor,  # [B, freqs_dim] or flat [B*q_len, freqs_dim]
    bsz: int,
    q_len: int,
) -> torch.Tensor:
    """Inverse-RoPE + grouped ``wo_a`` + ``wo_b`` + (TP) all-reduce.

    Fast path (CUDA, non-empty): ``fused_inv_rope_fp8_quant`` collapses
    ``apply_rotary_emb_batched`` (5 launches) + per-group
    ``per_token_group_quant_fp8`` (G launches) into ONE Triton kernel
    emitting ``(fp8 [M, G, K], scale [M, G, K/512])`` in the UE8M0 layout
    ``deep_gemm.fp8_einsum`` consumes. Matches vLLM
    ``deepseek_v4_attention.py``.

    Eager fallback (CPU / empty): explicit inv-rotary + bf16-dequant +
    ``einsum("bsgd,grd->bsgr")``, preserved for framework warmup forwards.
    """
    rd = attn.rope_head_dim

    if o.is_cuda and o.numel() > 0:
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            freqs_cis,
            n_groups=attn.n_groups,
            heads_per_group=attn.n_heads // attn.n_groups,
            nope_dim=attn.head_dim - attn.rope_head_dim,
            rope_head_dim=attn.rope_head_dim,
        )
        o = attn._wo_a_einsum_from_fp8(o_fp8, o_scale, bsz, q_len)
    else:
        if freqs_cis.dim() == 2 and int(freqs_cis.shape[0]) == bsz:
            apply_rotary_emb_batched(o[..., -rd:], freqs_cis, inverse=True)
        else:
            apply_rotary_emb(
                o[..., -rd:],
                freqs_cis.reshape(-1, freqs_cis.shape[-1]).contiguous(),
                inverse=True,
            )
        o = o.reshape(bsz, q_len, attn.n_groups, -1)
        wo_a_bf16 = _fp8_dequant_to_fp32(attn.wo_a_w, attn.wo_a_s).to(o.dtype)
        wo_a = wo_a_bf16.view(attn.n_groups, attn.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    out = attn._lin(attn.wo_b, o.flatten(2))
    if attn.tp_size > 1:
        from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

        all_reduce(out, Group.TP)
    return out
