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

from rtp_llm.models_py.modules.dsv41._fused_rmsnorm_rope_triton import (
    fused_rmsnorm_rope,
)
from rtp_llm.models_py.modules.dsv41._rope_only_triton import rope_only_inplace
from rtp_llm.models_py.modules.dsv41.rope import apply_rotary_emb

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rtp_llm.models_py.modules.dsv41.fp8.attention import AttentionFP8


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
    qr_quantized: tuple[torch.Tensor, torch.Tensor] | None = None
    swa_written: bool = False
    # Strided view into the same GEMM output as Main Q.
    index_q: torch.Tensor | None = None
    head_weights: torch.Tensor | None = None
    index_prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    global_slots: torch.Tensor | None = None
    flashmla_pdl: bool = False


def _fused_v41_q_rope_supported(q, freqs_cis, rope_dim: int) -> bool:
    """Shape-only gate: replay keeps positions/frequencies on the device."""
    return (
        q.is_cuda
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
    freqs_cis: torch.Tensor | None = None,
    *,
    swa_pool: torch.Tensor | None = None,
    swa_slots: torch.Tensor | None = None,
    mega_attention=None,
    shared_input_quant=None,
    qb_pdl: bool = False,
    ready_plan=None,
) -> DecodeQKV:
    """Decode Q/KV path — RMSNorm + LoRA Q + KV linear + fused RMSNorm-RoPE.

    ``position_ids`` is flat over the token-major ``[B, S]`` layout. For
    normal decode ``S == 1``; target verify passes the full verify span.
    """
    rd = attn.rope_head_dim
    if freqs_cis is None:
        position_ids = position_ids.reshape(-1).to(
            device=attn.freqs_cis.device, dtype=torch.long
        )
        freqs_cis = attn.freqs_cis.index_select(0, position_ids).contiguous()

    from rtp_llm.models_py.modules.dsv41._v41_query_quant import (
        try_project_qkv_to_cache,
        try_project_quantized_qkv,
    )

    from rtp_llm.models_py.modules.dsv41.fp8._v41_front_gemm import is_reindex, project

    # Restrict our fusions to Reindex, including when FULL overlap is disabled
    # or falls back. FULL keeps its original projection and PDL consumer.
    reindex = is_reindex(attn)
    front = (
        project(attn, x, quantized_input=shared_input_quant, pdl=qb_pdl)
        if reindex
        else None
    )
    head_weights = None if front is None else front[3]
    native_qkv = None if front is not None else try_project_qkv_to_cache(
        attn, x, freqs_cis, swa_pool, swa_slots, shared_input_quant
    )
    quantized_qkv = (
        front[:3] if front is not None else native_qkv
        if front is not None or native_qkv is not None
        else try_project_quantized_qkv(
            attn,
            x,
            quantized_input=shared_input_quant,
            pdl=qb_pdl,
        )
    )
    qr_quantized = None if quantized_qkv is None else quantized_qkv[2]
    if quantized_qkv is not None:
        fused_qkv = quantized_qkv[:2]
    elif shared_input_quant is not None:
        fused_qkv = attn._try_fused_qr_kv(x, shared_input_quant)
    else:
        fused_qkv = attn._try_fused_qr_kv(x)
    # Q path; fused KV remains a view until its strided RMSNorm/RoPE below.
    if fused_qkv is None:
        qr = attn._rmsnorm_weighted(attn._lin(attn.wq_a, x), attn.q_norm)
    else:
        qr = fused_qkv[0]
    if (
        ready_plan is not None
        and mega_attention is not None
        and qr_quantized is not None
        and fused_qkv is not None
        and not reindex
        and getattr(attn, "skip_post_q_norm", False)
        and rd == 64
        and (attn.n_heads, attn.head_dim, mega_attention.padded_heads) == (64, 512, 64)
    ):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_qb_kv import try_project_kv_index

        result = try_project_kv_index(
            mega_attention,
            *qr_quantized,
            fused_qkv[1],
            attn.kv_norm,
            position_ids,
            attn.freqs_cis,
            swa_pool,
            swa_slots,
            ready_plan,
            eps=attn.eps,
            wait_pdl=qb_pdl,
            trigger_pdl=ready_plan.pdl,
        )
        if result is not None:
            main_q, kv, slots = result
            return DecodeQKV(
                qr,
                main_q.unflatten(-1, (64, 512)),
                kv,
                freqs_cis,
                qr_quantized,
                True,
                global_slots=slots,
                flashmla_pdl=ready_plan.pdl,
            )
    index_q = None
    index_prepared = None
    merged = getattr(mega_attention, "merged_qb", None)
    if (
        os.environ.get("DSV41_FUSED_QB_KV", "0") == "1"
        and not reindex
        and merged is not None
        and qr_quantized is not None
        and fused_qkv is not None
        and getattr(attn, "skip_post_q_norm", False)
        and rd == 64
        and (attn.n_heads, attn.head_dim, getattr(mega_attention, "padded_heads", None))
        == (64, 512, 64)
    ):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_qb_kv import try_project_kv

        result = try_project_kv(
            merged,
            *qr_quantized,
            fused_qkv[1],
            attn.kv_norm,
            position_ids,
            attn.freqs_cis,
            swa_pool,
            swa_slots,
            eps=attn.eps,
            wait_pdl=qb_pdl,
        )
        if result is not None:
            main_q, index_q, kv = result
            return DecodeQKV(
                qr,
                main_q.unflatten(-1, (attn.n_heads, attn.head_dim)),
                kv,
                freqs_cis,
                qr_quantized,
                True,
                index_q,
            )
    if merged is not None and qr_quantized is not None:
        from rtp_llm.models_py.modules.dsv41._profiler import record_function_range

        with record_function_range("dsv41.decode.qb.merged"):
            prepared = (
                merged.try_forward_prepared(
                    *qr_quantized, head_weights, attn.freqs_cis, position_ids
                )
                # The custom consumer does not opt into PDL. Preserve the
                # official wait-enabled DeepGEMM consumer for a PDL producer.
                if reindex and rd == 64 and not qb_pdl else None
            )
            if prepared is None:
                q, index_q = merged.forward_quantized(
                    *qr_quantized, **({"wait_pdl": True} if qb_pdl else {})
                )
            else:
                q, index_prepared = prepared
        q = q.unflatten(-1, (attn.n_heads, attn.head_dim))
    elif mega_attention is not None:
        # The caller selected the complete mega path before preparing Q.
        q = mega_attention.project_q(attn.wq_b, qr, qr_quantized).view(
            *qr.shape[:-1], attn.n_heads, attn.head_dim
        )
    else:
        q = (
            attn._lin(attn.wq_b, qr)
            if qr_quantized is None
            else attn.wq_b.forward_quantized(*qr_quantized)
        ).unflatten(
            -1, (attn.n_heads, attn.head_dim)
        )  # [B, S, H, D]
    raw_kv = attn._lin(attn.wkv, x) if fused_qkv is None else fused_qkv[1]
    if native_qkv is not None:
        if mega_attention is None:
            _apply_v41_q_rope(q, freqs_cis, rd)
        return DecodeQKV(qr, q, raw_kv, freqs_cis, qr_quantized, True, index_q)
    if (
        getattr(attn, "skip_post_q_norm", False)
        and rd == 64
        and swa_pool is not None
        and swa_slots is not None
    ):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_qkv_rope_cache_triton import (
            try_fused_qkv_rope_cache,
        )

        fused = try_fused_qkv_rope_cache(
            q,
            raw_kv,
            attn.kv_norm,
            position_ids,
            attn.freqs_cis,
            swa_pool,
            swa_slots,
            eps=attn.eps,
            freqs_cis=freqs_cis,
            **(
                {"apply_q_rope": False, "pad_query_heads": mega_attention.padded_heads}
                if mega_attention is not None
                else {}
            ),
        )
        if fused is not None:
            return DecodeQKV(
                qr, fused.q, fused.kv, fused.freqs_cis, qr_quantized, True, index_q, head_weights,
                index_prepared
            )
    if mega_attention is None:
        if getattr(attn, "skip_post_q_norm", False):
            _apply_v41_q_rope(q, freqs_cis, rd)
        else:
            q = fused_rmsnorm_rope(q, None, freqs_cis, rd, eps=attn.eps)

    # KV path (single MQA head) — per-token RoPE using the same table lookup.
    kv = fused_rmsnorm_rope(
        raw_kv,
        attn.kv_norm,
        freqs_cis,
        rd,
        eps=attn.eps,
    )

    return DecodeQKV(
        qr=qr,
        q=q,
        kv=kv,
        freqs_cis=freqs_cis,
        qr_quantized=qr_quantized,
        index_q=index_q,
        head_weights=head_weights,
        index_prepared=index_prepared,
    )
