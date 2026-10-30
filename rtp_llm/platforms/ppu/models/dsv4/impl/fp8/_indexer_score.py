"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    _sm100_needs_triton_fallback,
    deepgemm_supports_block_kv,
    supports_paged_mqa_logits,
    supports_mqa_logits,
    has_fp8_paged_mqa_logits,
    has_fp8_mqa_logits,
    _sched_cache,
    _num_sms_cache,
    _get_num_sms,
)
from typing import Optional
import torch
from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)
from rtp_llm.platforms.ppu.models.dsv4.impl.platform_provider import run_dsv4_fp8_mqa_logits

try:
    import deep_gemm as _deep_gemm

    # DeepGEMM 2.8 renamed the group-128 MQA logits entry points to the
    # fp8_fp4_* family and switched q to a (tensor, sf|None) tuple.
    _HAS_DEEP_GEMM = hasattr(_deep_gemm, "fp8_paged_mqa_logits") and hasattr(
        _deep_gemm, "get_paged_mqa_logits_metadata"
    )
    _HAS_DEEP_GEMM_MQA = hasattr(_deep_gemm, "fp8_mqa_logits")
    # DeepGEMM 2.8.1 tightens the SM100 logits contracts: paged MQA requires
    # varlen indices and non-paged MQA requires per-element MX q scale
    # factors, so the group-128 legacy calls (q without SF) are rejected on
    # SM100 there and must fall back to the Triton kernel. 2.8.0 still
    # accepts them, and the legacy entry points exist in both versions.
    import re as _re

    _match = _re.match(r"(\d+)\.(\d+)\.(\d+)", getattr(_deep_gemm, "__version__", ""))
    _DEEP_GEMM_SM100_LEGACY_OK = bool(_match) and tuple(
        int(x) for x in _match.groups()
    ) < (2, 8, 1)
except ImportError:
    _deep_gemm = None
    _HAS_DEEP_GEMM = False
    _HAS_DEEP_GEMM_MQA = False
    _DEEP_GEMM_SM100_LEGACY_OK = False

def fp8_paged_indexer_score(
    q_fp8: torch.Tensor,  # [B, next_n, H, D] float8_e4m3fn
    w_fold: torch.Tensor,  # [B*next_n, H]    fp32
    kv_pool_uint8: torch.Tensor,  # [total_slots, 132] uint8 — flat pool view
    block_table: torch.Tensor,  # [B, max_blocks] int32 — logical→physical
    context_lens: torch.Tensor,  # [B, next_n] int32 — live K length per row
    block_size: int,  # tokens per cache block
    max_ctx_len: int,  # output T dim
    *,
    query_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """One-shot FP8 paged indexer logits via DeepGEMM.

    Returns ``[B*next_n, max_ctx_len] fp32`` — feed straight to topk.
    Padded columns past per-row ``context_lens[b, n]`` are left as
    whatever DeepGEMM writes (use ``clean_logits=True`` if the
    downstream topk needs ``-inf`` there; default False to save the
    extra mask).
    """
    assert q_fp8.dtype == torch.float8_e4m3fn, f"q_fp8 dtype={q_fp8.dtype}"
    assert q_fp8.dim() == 4 and q_fp8.shape[-1] == INDEXER_HEAD_DIM
    assert w_fold.dtype == torch.float32 and w_fold.dim() == 2
    assert kv_pool_uint8.dtype == torch.uint8
    assert kv_pool_uint8.shape[-1] == INDEXER_ENTRY_BYTES
    assert block_table.dtype == torch.int32 and block_table.dim() == 2
    assert context_lens.dtype == torch.int32 and context_lens.dim() == 2
    if query_chunk_size is not None:
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        batch, next_n, heads, _ = q_fp8.shape
        if next_n > query_chunk_size:
            # Only providers with a bounded next_n ABI request this path.
            # Preserve batch-major order across speculative query slices.
            weights = w_fold.view(batch, next_n, heads)
            parts = []
            for start in range(0, next_n, query_chunk_size):
                end = min(start + query_chunk_size, next_n)
                part = fp8_paged_indexer_score(
                    q_fp8[:, start:end].contiguous(),
                    weights[:, start:end].contiguous().view(-1, heads),
                    kv_pool_uint8,
                    block_table,
                    context_lens[:, start:end].contiguous(),
                    block_size,
                    max_ctx_len,
                )
                parts.append(part.view(batch, end - start, max_ctx_len))
            return torch.cat(parts, dim=1).view(batch * next_n, max_ctx_len)
    # DeepGEMM kv_cache shape: [num_blocks, block_size, 1, D+4] uint8.
    # Our pool is a flat [total_slots, 132] view; reshape into the 4D
    # layout (no copy — just a metadata change).
    total_slots = kv_pool_uint8.shape[0]
    assert (
        total_slots % block_size == 0
    ), f"total_slots={total_slots} not divisible by block_size={block_size}"
    num_blocks = total_slots // block_size
    kv_4d = kv_pool_uint8.view(num_blocks, block_size, 1, INDEXER_ENTRY_BYTES)

    if not supports_paged_mqa_logits(q_fp8.device, block_size):
        from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
            legacy_fp8_paged_mqa_logits,
        )

        return legacy_fp8_paged_mqa_logits(
            q_fp8, kv_4d, w_fold, context_lens, block_table, max_ctx_len
        )
    num_sms = _get_num_sms(q_fp8.device)
    schedule = _deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, block_size, num_sms
    )
    return _deep_gemm.fp8_paged_mqa_logits(
        q_fp8.contiguous(),
        kv_4d,
        w_fold.contiguous(),
        context_lens,
        block_table,
        schedule,
        max_ctx_len,
    )

def fp8_mqa_indexer_score(
    q_fp8: torch.Tensor,  # [M, H, D] float8_e4m3fn
    w_fold: torch.Tensor,  # [M, H]    fp32
    k_quant: torch.Tensor,  # [N, D]    float8_e4m3fn
    k_scale: torch.Tensor,  # [N]       float32
    cu_seqlen_ks: torch.Tensor,  # [M]       int32
    cu_seqlen_ke: torch.Tensor,  # [M]       int32
    *,
    clean_logits: bool = False,
    max_seqlen_k: int = 0,
    platform_provider=None,
) -> torch.Tensor:
    """One-shot non-paged FP8 indexer logits via DeepGEMM.

    Returns ``[M, N] fp32`` (M = total Q tokens this chunk; N = total K
    tokens in the gathered workspace). Caller reshapes back to ``[B, S, T]``.

    ``clean_logits=False`` matches what we want — entries past
    ``cu_seqlen_ke[m]`` are left untouched; the topk-with-causal-mask path
    in :class:`Indexer.forward` re-applies its own ``q_pos`` causal cap.
    """
    assert q_fp8.dtype == torch.float8_e4m3fn and q_fp8.dim() == 3
    assert q_fp8.shape[-1] == INDEXER_HEAD_DIM
    assert w_fold.dtype == torch.float32 and w_fold.dim() == 2
    assert w_fold.shape[0] == q_fp8.shape[0]
    assert k_quant.dtype == torch.float8_e4m3fn and k_quant.dim() == 2
    assert k_quant.shape[-1] == INDEXER_HEAD_DIM
    assert k_scale.dtype == torch.float32 and k_scale.dim() == 1
    assert k_scale.shape[0] == k_quant.shape[0]
    assert cu_seqlen_ks.dtype == torch.int32 and cu_seqlen_ks.dim() == 1
    assert cu_seqlen_ke.dtype == torch.int32 and cu_seqlen_ke.dim() == 1
    assert cu_seqlen_ks.shape[0] == q_fp8.shape[0]
    assert cu_seqlen_ke.shape[0] == q_fp8.shape[0]

    if not supports_mqa_logits(q_fp8.device):
        from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
            legacy_fp8_mqa_logits,
        )

        return legacy_fp8_mqa_logits(
            q_fp8,
            (k_quant, k_scale),
            w_fold,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=clean_logits,
            max_seqlen_k=max_seqlen_k,
        )

    return run_dsv4_fp8_mqa_logits(
        _deep_gemm.fp8_mqa_logits,
        q_fp8.contiguous(),
        (k_quant.contiguous(), k_scale.contiguous()),
        w_fold.contiguous(),
        cu_seqlen_ks.contiguous(),
        cu_seqlen_ke.contiguous(),
        clean_logits,
        max_seqlen_k,
        platform_provider=platform_provider,
    )
