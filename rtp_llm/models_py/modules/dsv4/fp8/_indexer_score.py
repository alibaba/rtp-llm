"""DSv4 Indexer FP8-paged score path via DeepGEMM.

Wraps ``deep_gemm.fp8_paged_mqa_logits`` so it can drop into the indexer
decode loop in place of the bf16 ``v4_indexer_score`` Triton kernel
when the cache is FP8 packed (132B/slot).

End-to-end shape contract:

  q_fp8       [B, next_n, H, D]            float8_e4m3fn  (per-(t,h) quant)
  w_fold      [B*next_n, H]                fp32           (per-token Q
                                                            scale folded in)
  kv_cache    [num_blocks, block_size, 1, D+4]  uint8     (132B per slot:
                                                            128 FP8 K + 4B fp32 scale)
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

import os
from typing import Optional

import torch

# DeepGEMM JIT writes ``kernel.cu`` under ``$HOME/.deep_gemm/tmp/<id>/``
# and shells out to NVCC; if ``HOME`` is unset (bazel test sandbox does
# not propagate it by default) ``os.path.expanduser("~")`` returns ``~``
# unchanged and DeepGEMM falls back to the relative path
# ``.deep_gemm/tmp/<id>/``.  NVCC's child cc1plus then runs in a
# different CWD and reports ``fatal error: .deep_gemm/tmp/.../kernel.cu:
# No such file or directory``.  Pin a writable absolute fallback before
# DeepGEMM is imported so the JIT cache lands at ``/tmp/.deep_gemm/``.
# ``setdefault`` is a no-op in production (real user HOME is set).
os.environ.setdefault("HOME", "/tmp")

from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)

try:
    import deep_gemm as _deep_gemm

    _HAS_DEEP_GEMM = hasattr(_deep_gemm, "fp8_paged_mqa_logits") and hasattr(
        _deep_gemm, "get_paged_mqa_logits_metadata"
    )
    _HAS_DEEP_GEMM_MQA = hasattr(_deep_gemm, "fp8_mqa_logits")
except ImportError:
    _deep_gemm = None
    _HAS_DEEP_GEMM = False
    _HAS_DEEP_GEMM_MQA = False


def has_fp8_paged_mqa_logits() -> bool:
    return _HAS_DEEP_GEMM


def has_fp8_mqa_logits() -> bool:
    return _HAS_DEEP_GEMM_MQA


_sched_cache: Optional[torch.Tensor] = None
_num_sms_cache: int = 0


def _get_num_sms(device: torch.device) -> int:
    global _num_sms_cache
    if _num_sms_cache == 0:
        _num_sms_cache = torch.cuda.get_device_properties(device).multi_processor_count
    return _num_sms_cache


def fp8_paged_indexer_score(
    q_fp8: torch.Tensor,  # [B, next_n, H, D] float8_e4m3fn
    w_fold: torch.Tensor,  # [B*next_n, H]    fp32
    kv_pool_uint8: torch.Tensor,  # [total_slots, 132] uint8 — flat pool view
    block_table: torch.Tensor,  # [B, max_blocks] int32 — logical→physical
    context_lens: torch.Tensor,  # [B, next_n] int32 — live K length per row
    block_size: int,  # tokens per cache block
    max_ctx_len: int,  # output T dim
    *,
    region_offset: int = 0,
) -> torch.Tensor:
    """One-shot FP8 paged indexer logits via DeepGEMM.

    Returns ``[B*next_n, max_ctx_len] fp32`` — feed straight to topk.
    Padded columns past per-row ``context_lens[b, n]`` are left as
    whatever DeepGEMM writes (use ``clean_logits=True`` if the
    downstream topk needs ``-inf`` there; default False to save the
    extra mask).

    ``region_offset`` (M08 §4.4): constexpr block-base for the
    asymmetric ``bps > 1`` future variant. Under DSV4 today (bps=1)
    this MUST be 0 — caller is expected to pre-slice the pool view to
    the INDEXER_KV region. Ignored on the kernel pointer math path
    today; reserved so PR-2 plumbing does not churn the signature when
    F02 ratios change.
    """
    assert _HAS_DEEP_GEMM, "deep_gemm.fp8_paged_mqa_logits not available"
    assert q_fp8.dtype == torch.float8_e4m3fn, f"q_fp8 dtype={q_fp8.dtype}"
    assert q_fp8.dim() == 4 and q_fp8.shape[-1] == INDEXER_HEAD_DIM
    assert w_fold.dtype == torch.float32 and w_fold.dim() == 2
    assert kv_pool_uint8.dtype == torch.uint8
    assert kv_pool_uint8.shape[-1] == INDEXER_ENTRY_BYTES
    assert block_table.dtype == torch.int32 and block_table.dim() == 2
    assert context_lens.dtype == torch.int32 and context_lens.dim() == 2
    if __debug__:
        # M08 §4.4 + §10.9 — DeepGEMM I5 zero-copy reshape contract.
        # block_table must be contiguous int32; pool slot axis must
        # factor cleanly; region_offset==0 today (bps=1).
        assert block_table.is_contiguous(), (
            "fp8_paged_indexer_score: block_table must be contiguous "
            "(DeepGEMM zero-copy reshape contract — M08 §10.9)"
        )
        assert region_offset == 0, (
            "fp8_paged_indexer_score: region_offset > 0 reserved for "
            "future bps>1 asymmetric variant (M08 §4.4); today must be 0"
        )
    # DeepGEMM kv_cache shape: [num_blocks, block_size, 1, D+4] uint8.
    # Our pool is a flat [total_slots, 132] view; reshape into the 4D
    # layout (no copy — just a metadata change).
    total_slots = kv_pool_uint8.shape[0]
    assert (
        total_slots % block_size == 0
    ), f"total_slots={total_slots} not divisible by block_size={block_size}"
    num_blocks = total_slots // block_size
    kv_4d = kv_pool_uint8.view(num_blocks, block_size, 1, INDEXER_ENTRY_BYTES)

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
) -> torch.Tensor:
    """One-shot non-paged FP8 indexer logits via DeepGEMM.

    Returns ``[M, N] fp32`` (M = total Q tokens this chunk; N = total K
    tokens in the gathered workspace). Caller reshapes back to ``[B, S, T]``.

    ``clean_logits=False`` matches what we want — entries past
    ``cu_seqlen_ke[m]`` are left untouched; the topk-with-causal-mask path
    in :class:`Indexer.forward` re-applies its own ``q_pos`` causal cap.
    """
    assert _HAS_DEEP_GEMM_MQA, "deep_gemm.fp8_mqa_logits not available"
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

    return _deep_gemm.fp8_mqa_logits(
        q_fp8.contiguous(),
        (k_quant.contiguous(), k_scale.contiguous()),
        w_fold.contiguous(),
        cu_seqlen_ks.contiguous(),
        cu_seqlen_ke.contiguous(),
        clean_logits,
        max_seqlen_k,
    )
