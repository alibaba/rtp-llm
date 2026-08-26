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

# ``get_paged_mqa_logits_metadata`` accepts 32/64/128, but the execution
# kernel is stricter across architectures: 64 is the portable block size
# (32 is SM100-only and 128 is rejected by the attention kernel).
_PORTABLE_PAGED_BLOCK_KV = (64,)


def _supported_paged_block_kv(device: torch.device) -> tuple[int, ...]:
    # DeepGEMM's 32-entry execution path is specific to SM100. Keep the
    # existing 128-token owner (32 compressed entries) working there, while
    # selecting the portable 64-entry row for all other architectures.
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 10:
        return (32, 64)
    return _PORTABLE_PAGED_BLOCK_KV


def _get_num_sms(device: torch.device) -> int:
    global _num_sms_cache
    if _num_sms_cache == 0:
        _num_sms_cache = torch.cuda.get_device_properties(device).multi_processor_count
    return _num_sms_cache


def validate_indexer_paged_layout(
    kv_pool: torch.Tensor,
    kernel_block_table: torch.Tensor,
    kernel_entries_per_block: int,
    owner_tokens_per_block: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Validate and expose the framework's INDEXER_KV kernel-row layout.

    The framework may subdivide one physical cache block into several kernel
    blocks. DeepGEMM requires each exposed kernel row itself to use a supported
    ``block_kv``; this function does not merge or remap rows. Kernel ids are
    already expanded by the framework as
    ``physical_id * blocks_per_owner + subblock`` and are passed through.
    """
    kernel_eb = int(kernel_entries_per_block)
    owner_tpb = int(owner_tokens_per_block)
    ratio = int(compress_ratio)
    if kernel_eb <= 0 or owner_tpb <= 0 or ratio <= 0:
        raise RuntimeError(
            "DSV4 indexer paged layout requires positive kernel entries, "
            f"owner tokens and compression ratio; got {kernel_eb}, "
            f"{owner_tpb}, {ratio}"
        )
    if owner_tpb % ratio != 0:
        raise RuntimeError(
            f"DSV4 indexer owner block size {owner_tpb} is not divisible by "
            f"compression ratio {ratio}"
        )
    owner_eb = owner_tpb // ratio
    if owner_eb % kernel_eb != 0:
        raise RuntimeError(
            f"DSV4 indexer owner entries {owner_eb} are not divisible by "
            f"kernel entries {kernel_eb}"
        )
    blocks_per_owner = owner_eb // kernel_eb
    if kernel_block_table.dim() != 2 or kernel_block_table.dtype != torch.int32:
        raise RuntimeError(
            "DSV4 indexer kernel block table must be 2D int32, got "
            f"shape={tuple(kernel_block_table.shape)}, "
            f"dtype={kernel_block_table.dtype}"
        )
    if int(kernel_block_table.shape[1]) % blocks_per_owner != 0:
        raise RuntimeError(
            "DSV4 indexer kernel block-table width must contain complete "
            f"physical owners: width={int(kernel_block_table.shape[1])}, "
            f"kernel_blocks_per_owner={blocks_per_owner}"
        )
    supported_block_kv = _supported_paged_block_kv(kv_pool.device)
    if kernel_eb not in supported_block_kv:
        raise RuntimeError(
            "DSV4 indexer kernel row is not supported by DeepGEMM: "
            f"kernel_block_kv={kernel_eb}, "
            f"physical_block_kv={owner_eb}, supported="
            f"{supported_block_kv}"
        )
    deepgemm_eb = kernel_eb
    if kv_pool.dim() == 3:
        if (
            int(kv_pool.shape[1]) != kernel_eb
            or int(kv_pool.shape[2]) != INDEXER_ENTRY_BYTES
        ):
            raise RuntimeError(
                "DSV4 indexer pool shape does not match kernel geometry: "
                f"shape={tuple(kv_pool.shape)}, kernel_entries={kernel_eb}"
            )
        if int(kv_pool.shape[0]) % blocks_per_owner != 0:
            raise RuntimeError(
                "DSV4 indexer kernel pool rows must contain complete physical "
                f"owners: rows={int(kv_pool.shape[0])}, "
                f"kernel_blocks_per_owner={blocks_per_owner}"
            )
        # Preserve the block-row stride: framework opaque pools may append
        # shared-pool padding after the useful entries in every kernel row.
        pool_for_kernel = kv_pool
    elif kv_pool.dim() == 2 and int(kv_pool.shape[1]) == INDEXER_ENTRY_BYTES:
        total_entries = int(kv_pool.shape[0])
        if total_entries % deepgemm_eb != 0:
            raise RuntimeError(
                f"DSV4 indexer pool entries {total_entries} are not divisible "
                f"by DeepGEMM block_kv={deepgemm_eb}"
            )
        kernel_rows = total_entries // deepgemm_eb
        if kernel_rows % blocks_per_owner != 0:
            raise RuntimeError(
                "DSV4 indexer flattened pool rows must contain complete "
                f"physical owners: rows={kernel_rows}, "
                f"kernel_blocks_per_owner={blocks_per_owner}"
            )
        pool_for_kernel = kv_pool
    else:
        raise RuntimeError(
            "DSV4 indexer pool must be [blocks, entries, 132] or "
            f"[slots, 132], got {tuple(kv_pool.shape)}"
        )

    return pool_for_kernel, kernel_block_table, deepgemm_eb


def fp8_paged_indexer_score(
    q_fp8: torch.Tensor,  # [B, next_n, H, D] float8_e4m3fn
    w_fold: torch.Tensor,  # [B*next_n, H]    fp32
    kv_pool_uint8: torch.Tensor,  # [blocks, block, 132] (may have row padding)
    block_table: torch.Tensor,  # [B, max_blocks] int32 — logical→physical
    context_lens: torch.Tensor,  # [B, next_n] int32 — live K length per row
    block_size: int,  # tokens per cache block
    max_ctx_len: int,  # output T dim
) -> torch.Tensor:
    """One-shot FP8 paged indexer logits via DeepGEMM.

    Returns ``[B*next_n, max_ctx_len] fp32`` — feed straight to topk.
    Padded columns past per-row ``context_lens[b, n]`` are left as
    whatever DeepGEMM writes (use ``clean_logits=True`` if the
    downstream topk needs ``-inf`` there; default False to save the
    extra mask).
    """
    assert _HAS_DEEP_GEMM, "deep_gemm.fp8_paged_mqa_logits not available"
    assert q_fp8.dtype == torch.float8_e4m3fn, f"q_fp8 dtype={q_fp8.dtype}"
    assert q_fp8.dim() == 4 and q_fp8.shape[-1] == INDEXER_HEAD_DIM
    assert w_fold.dtype == torch.float32 and w_fold.dim() == 2
    assert kv_pool_uint8.dtype == torch.uint8
    assert kv_pool_uint8.shape[-1] == INDEXER_ENTRY_BYTES
    assert block_table.dtype == torch.int32 and block_table.dim() == 2
    assert context_lens.dtype == torch.int32 and context_lens.dim() == 2
    # DeepGEMM kv_cache shape: [num_blocks, block_size, 1, D+4] uint8.
    if kv_pool_uint8.dim() == 3:
        assert kv_pool_uint8.shape[1:] == (block_size, INDEXER_ENTRY_BYTES)
        # ``unsqueeze`` retains a possibly padded dim-0 stride while exposing
        # DeepGEMM's [block, token, head=1, bytes] contract without a copy.
        kv_4d = kv_pool_uint8.unsqueeze(2)
    else:
        # Legacy tightly-packed flat pool.
        total_slots = kv_pool_uint8.shape[0]
        assert (
            total_slots % block_size == 0
        ), f"total_slots={total_slots} not divisible by block_size={block_size}"
        num_blocks = total_slots // block_size
        kv_4d = kv_pool_uint8.view(
            num_blocks, block_size, 1, INDEXER_ENTRY_BYTES
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
