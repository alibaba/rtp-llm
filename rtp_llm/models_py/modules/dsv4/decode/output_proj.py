"""DSv4 decode output projection — extracted from ``AttentionVLLM._forward_decode_body``.

Pipeline:
    1. Inverse partial RoPE (unrotate the ``[rope_head_dim:]`` slice).
    2. Per-group per-token FP8 quant (UE8M0 scale layout) for ``wo_a`` GEMM.
    3. ``wo_a`` grouped einsum via ``deep_gemm.fp8_einsum``.
    4. ``wo_b`` linear.
    5. Optional TP ``all_reduce``.

The CUDA fast path fuses steps 1+2 into a single Triton kernel
(``fused_inv_rope_fp8_quant``). The eager fallback (CPU / empty) mirrors
the original per-op path, kept for warmup / unit tests. Activation/output
are BF16; the FP8 quant feeds ``wo_a``'s FP8-weight GEMM only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4.qlinear import _fp8_dequant_to_fp32
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb_batched

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.dsv4.attention_vllm import AttentionVLLM as Attention


def decode_output_proj(
    attn: "Attention",
    o: torch.Tensor,
    freqs_cis_per_req: torch.Tensor,
    bsz: int,
    q_len: int,
    dbg_tag: str = "",
) -> torch.Tensor:
    """Inverse-RoPE + grouped ``wo_a`` + ``wo_b`` + (TP) all-reduce.

    Fast path (CUDA, non-empty): ``fused_inv_rope_fp8_quant`` collapses
    ``apply_rotary_emb_batched`` (5 launches) + per-group
    ``per_token_group_quant_fp8`` (G launches) into ONE Triton kernel
    emitting ``(fp8 [M, G, K], scale [M, G, K/512])`` in the UE8M0 layout
    ``deep_gemm.fp8_einsum`` consumes.

    Eager fallback (CPU / empty): explicit inv-rotary + bf16-dequant +
    ``einsum("bsgd,grd->bsgr")``, preserved for framework warmup forwards.
    """
    rd = attn.rope_head_dim

    with torch.profiler.record_function(
        f"{dbg_tag}/output_proj" if dbg_tag else "output_proj"
    ):
        if o.is_cuda and o.numel() > 0:
            o_fp8, o_scale = fused_inv_rope_fp8_quant(
                o,
                freqs_cis_per_req,
                n_groups=attn.n_groups,
                heads_per_group=attn.n_heads // attn.n_groups,
                nope_dim=attn.head_dim - attn.rope_head_dim,
                rope_head_dim=attn.rope_head_dim,
            )
            o = attn._wo_a_einsum_from_fp8(o_fp8, o_scale, bsz, q_len)
        else:
            apply_rotary_emb_batched(o[..., -rd:], freqs_cis_per_req, inverse=True)
            o = o.reshape(bsz, q_len, attn.n_groups, -1)
            wo_a_bf16 = _fp8_dequant_to_fp32(attn.wo_a_w, attn.wo_a_s).to(o.dtype)
            wo_a = wo_a_bf16.view(attn.n_groups, attn.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        out = attn._lin(attn.wo_b, o.flatten(2))
        if attn.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            all_reduce(out, Group.TP)
        return out
