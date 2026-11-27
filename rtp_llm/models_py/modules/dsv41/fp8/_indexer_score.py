"""DSv4 Indexer FP8-paged score path via DeepGEMM.

Wraps ``deep_gemm.fp8_paged_mqa_logits`` so it can drop into the indexer
decode loop in place of the bf16 ``v4_indexer_score`` Triton kernel
when the cache is FP8 packed (132B/slot).

The FP8 score wrappers share dsv4's DeepGEMM and Triton dispatch. The FP4
variants below keep their own DeepGEMM path.

End-to-end shape contract:

  q_fp8       [B, next_n, H, D]            float8_e4m3fn  (per-(t,h) quant)
  w_fold      [B*next_n, H]                fp32 or bf16   (legacy folded or
                                                            MX raw weights)
  kv_cache    [num_blocks, block_size, 1, D+4]  uint8     (132B per slot:
                                                            128 FP8 K + 4B scale)
  context_lens[B, next_n]                  int32          (live K length per row)
  block_table [B, max_blocks]              int32          (logical→physical block id)

Returns ``[B*next_n, max_ctx_len] fp32`` logits — same semantics as
``v4_indexer_score``: each row is the per-K-token score after fused
einsum + ReLU + per-head weighted sum.

Caller is responsible for FP8 quantizing Q via
:func:`indexer_q_fp8_quant_fold`, building the block_table, and the
2D context_lens shape DeepGEMM requires.
"""

from __future__ import annotations

from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    fp8_mqa_indexer_score as _shared_fp8_mqa_indexer_score,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    fp8_paged_indexer_score as _shared_fp8_paged_indexer_score,
)

# JIT paths are configured centrally by jit_cache_env before loading DeepGEMM.


try:
    import deep_gemm as _deep_gemm

    _HAS_DEEP_GEMM_FP4_MQA = hasattr(_deep_gemm, "fp8_fp4_mqa_logits")
    _HAS_DEEP_GEMM_FP4_PAGED_MQA = hasattr(
        _deep_gemm, "fp8_fp4_paged_mqa_logits"
    ) and hasattr(_deep_gemm, "get_paged_mqa_logits_metadata")
except (ImportError, RuntimeError) as error:
    if isinstance(error, RuntimeError) and torch.cuda.is_available():
        raise
    _deep_gemm = None
    _HAS_DEEP_GEMM_FP4_MQA = False
    _HAS_DEEP_GEMM_FP4_PAGED_MQA = False


def has_fp8_paged_mqa_logits() -> bool:
    # The shared FP8 scorer also serves the original 132B cache through Triton
    # when the installed DeepGEMM only accepts MX scales on SM100.
    return True


def has_fp8_mqa_logits() -> bool:
    return True


def has_fp8_fp4_mqa_logits() -> bool:
    return _HAS_DEEP_GEMM_FP4_MQA


def has_fp8_fp4_paged_mqa_logits() -> bool:
    return _HAS_DEEP_GEMM_FP4_PAGED_MQA


# ---------------------------------------------------------------------------
# V4.1 FP4 (MX mode) prefill indexer wrapper around
# ``deep_gemm.fp8_fp4_mqa_logits``.
#
# Shape contract:
#   q_payload  [M, H, 64]  int8   (packed e2m1; logical head_dim 128)
#   q_sf       [M, H]      int32  (packed UE8M0, one int32 per token/head)
#   k_payload  [N, 64]     int8
#   k_sf       [N]         int32
#   weights    [M, H]      fp32   (raw head weights; scales live in the SFs)
#
# Returns ``[M, N] fp32`` logits — same semantics as the FP8 path: each row
# is the per-K-token score after fused einsum + ReLU + per-head weighted sum.
# ---------------------------------------------------------------------------


def fp8_fp4_mqa_indexer_score(
    q_payload: torch.Tensor,  # [M, H, 64] int8
    q_sf: torch.Tensor,  # [M, H] int32
    k_payload: torch.Tensor,  # [N, 64] int8
    k_sf: torch.Tensor,  # [N] int32
    weights: torch.Tensor,  # [M, H] fp32
    cu_seqlen_ks: torch.Tensor,  # [M] int32 (K start, inclusive)
    cu_seqlen_ke: torch.Tensor,  # [M] int32 (K end, exclusive)
    *,
    clean_logits: bool = True,
    max_seqlen_k: int = 0,
) -> torch.Tensor:
    """MX-mode FP4 prefill indexer logits via DeepGEMM.

    Both scale factors use DeepGEMM's packed-UE8M0 int32 form (group 32 along
    the head dim), matching the V4.1-Flash official indexer quantization.
    """
    from rtp_llm.models_py.modules.dsv4.fp8._mqa_layout import normalize_mqa_logits

    # Upstream SM100 now consumes BF16 head weights and emits request-relative
    # BF16 logits. Selectors keep their FP32 tensor ABI, but no longer rely on
    # the removed clean_logits argument or uninitialized tail values.
    width = max_seqlen_k if max_seqlen_k > 0 else k_payload.shape[0]
    scores = _deep_gemm.fp8_fp4_mqa_logits(
        (q_payload.contiguous(), q_sf.contiguous()),
        (k_payload.contiguous(), k_sf.contiguous()),
        weights.to(torch.bfloat16).contiguous(),
        cu_seqlen_ks.contiguous(),
        cu_seqlen_ke.contiguous(),
        width,
    )
    return normalize_mqa_logits(
        scores, cu_seqlen_ks, cu_seqlen_ke, width, relative=max_seqlen_k > 0
    )


_num_sms_cache: int = 0


def _get_num_sms(device: torch.device) -> int:
    global _num_sms_cache
    if _num_sms_cache == 0:
        _num_sms_cache = torch.cuda.get_device_properties(device).multi_processor_count
    return _num_sms_cache


# ---------------------------------------------------------------------------
# V4.1 FP4 (MX mode) decode paged indexer wrapper around
# ``deep_gemm.fp8_fp4_paged_mqa_logits``.
#
# The fused cache is the INDEX_K pool itself: the planar per-block layout
# (payload plane then packed-UE8M0 scale plane) is exactly DeepGEMM's fused
# paged form — [num_blocks, block_kv, 1, 68] uint8 with per-block stride
# ``block_kv * 68``. DeepGEMM derives the payload/SF views from it.
#
# Shape contract:
#   q_payload  [B, next_n, H, 64]  int8   (packed e2m1; logical head_dim 128)
#   q_sf       [B, next_n, H]      int32  (packed UE8M0, one int32 per token/head)
#   kv_pool    [num_blocks, block_kv, 68] uint8 (planar per block)
#   weights    [B*next_n, H]       fp32   (raw head weights; scales live in SFs)
#
# Returns ``[B*next_n, max_ctx_len] fp32`` logits — same semantics as the
# FP8 paged path (columns >= context_lens are left as DeepGEMM writes them;
# the caller compacts and masks).
# ---------------------------------------------------------------------------


def fp8_fp4_paged_indexer_score(
    q_payload: torch.Tensor,  # [B, next_n, H, 64] int8 packed e2m1
    q_sf: torch.Tensor,  # [B, next_n, H] int32 packed UE8M0
    kv_pool_uint8: torch.Tensor,  # [num_blocks, block_kv, 68] uint8 — planar pool
    weights: torch.Tensor,  # [B*next_n, H] fp32
    block_table: torch.Tensor,  # [B, max_blocks] int32 — logical→physical
    context_lens: torch.Tensor,  # [B, next_n] int32 — live K length per row
    block_size: int,  # entries per cache block (= pool.shape[1])
    max_ctx_len: int,  # output T dim (physical capacity)
) -> torch.Tensor:
    """MX-mode FP4 paged indexer logits via DeepGEMM.

    Both scale factors use DeepGEMM's packed-UE8M0 int32 form (group 32 along
    the head dim), matching the V4.1-Flash official indexer quantization.
    """
    from rtp_llm.models_py.modules.dsv4.fp8._mqa_layout import normalize_mqa_logits

    # SM100's current API uses one query per row plus explicit request ids.
    # Expand only the small block-table metadata for speculative next_n.
    batch, next_n, heads, packed_dim = q_payload.shape
    rows = batch * next_n
    request_ids = torch.arange(
        batch, dtype=torch.int32, device=q_payload.device
    ).repeat_interleave(next_n)
    q = q_payload.reshape(rows, 1, heads, packed_dim).contiguous()
    sf = q_sf.reshape(rows, 1, heads).contiguous()
    lengths = context_lens.reshape(rows, 1).to(torch.int32).contiguous()
    tables = block_table.repeat_interleave(next_n, dim=0).contiguous()
    schedule = _deep_gemm.get_paged_mqa_logits_metadata(
        lengths, block_size, _get_num_sms(q_payload.device), indices=request_ids
    )
    scores = _deep_gemm.fp8_fp4_paged_mqa_logits(
        (q, sf),
        kv_pool_uint8.unsqueeze(2),
        weights.to(torch.bfloat16).contiguous(),
        lengths,
        tables,
        schedule,
        max_ctx_len,
        indices=request_ids,
    )
    ends = lengths.view(-1)
    return normalize_mqa_logits(
        scores, torch.zeros_like(ends), ends, max_ctx_len, relative=True
    )


def fp8_paged_indexer_score(
    q_fp8: torch.Tensor,  # [B, next_n, H, D] float8_e4m3fn
    w_fold: torch.Tensor,  # [B*next_n, H]    fp32
    kv_pool_uint8: torch.Tensor,  # [total_slots, 132] uint8 — flat pool view
    block_table: torch.Tensor,  # [B, max_blocks] int32 — logical→physical
    context_lens: torch.Tensor,  # [B, next_n] int32 — live K length per row
    block_size: int,  # tokens per cache block
    max_ctx_len: int,  # output T dim
    *,
    q_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-shot FP8 paged indexer logits via DeepGEMM.

    Returns ``[B*next_n, max_ctx_len] fp32`` — feed straight to topk.
    Padded columns past per-row ``context_lens[b, n]`` are ignored by the
    downstream TopK, which masks each row to its live context length.

    Uses dsv4's shared DeepGEMM or Triton dispatch for the active cache ABI.
    """
    if not q_fp8.is_cuda:
        raise RuntimeError(
            "dsv41 fp8_paged_indexer_score requires a CUDA tensor, got "
            f"{q_fp8.device}, block_kv={block_size}"
        )
    return _shared_fp8_paged_indexer_score(
        q_fp8,
        w_fold,
        kv_pool_uint8,
        block_table,
        context_lens,
        block_size,
        max_ctx_len,
        q_sf=q_sf,
    )


# ---------------------------------------------------------------------------
# Prefill (non-paged) wrapper around ``deep_gemm.fp8_mqa_logits``.
#
# Shape contract:
#   q_fp8        [M, H, D]    float8_e4m3fn  (M = total query tokens)
#   w_fold       [M, H]       fp32           (per-(token, head) Q scale folded in)
#   k_quant      [N, D]       float8_e4m3fn  (N = total key tokens — gathered
#                                              contiguous from the FP8 cache)
#   k_scale      [N]          float32
#   cu_seqlen_ks [M]          int32          (K start, inclusive)
#   cu_seqlen_ke [M]          int32          (K end,   exclusive)
#
# Returns ``[M, N] fp32`` logits — same semantics as ``v4_indexer_score``
# but laid out flat over total query tokens (the indexer prefill caller
# reshapes back to ``[B, S, T]``).
# ---------------------------------------------------------------------------


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
    q_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-shot non-paged FP8 indexer logits via DeepGEMM.

    Returns ``[M, N] fp32`` (M = total Q tokens this chunk; N = total K
    tokens in the gathered workspace). Caller reshapes back to ``[B, S, T]``.

    ``clean_logits=False`` matches what we want — entries past
    ``cu_seqlen_ke[m]`` are left untouched; the topk-with-causal-mask path
    in :class:`Indexer.forward` re-applies its own ``q_pos`` causal cap.

    Uses dsv4's shared DeepGEMM or Triton dispatch for the active cache ABI.
    """
    if not q_fp8.is_cuda:
        raise RuntimeError(
            f"dsv41 fp8_mqa_indexer_score requires a CUDA tensor, got {q_fp8.device}"
        )
    return _shared_fp8_mqa_indexer_score(
        q_fp8,
        w_fold,
        k_quant,
        k_scale,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
        max_seqlen_k=max_seqlen_k,
        q_sf=q_sf,
    )
