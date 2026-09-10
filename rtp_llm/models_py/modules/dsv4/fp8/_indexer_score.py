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

_BAND_BOUNDS_CT = [0]  # S4 engagement proof (DSV4_DIAG, first 3 fires)

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

# M5-B (Sep 7): the SM120 bf16 fallback below predates nv_dev DeepGEMM's
# sm120_fp8_mqa_logits (built at DeepGEMM/build/lib...). DSV4_INDEXER_FP8_DEEPGEMM=1
# lets cap-12 devices call deep_gemm.fp8_mqa_logits on the FP8 operands directly
# instead of dequanting to bf16 and re-running the triton kernel. Offline race
# (bench/p2_m5b_fp8_score.py): 6.6x at the 32K shape, rel-diff p50 0.0005 /
# max 0.36%, Jaccard@512 p50 1.0 (min 0.988). Requires the launcher PYTHONPATH
# shadow (RF wheel bundles an OLD deep_gemm without the SM120 kernel).
_FP8_DEEPGEMM_ON_SM120 = os.environ.get("DSV4_INDEXER_FP8_DEEPGEMM", "0") == "1"

# (d) SM120 paged indexer -> deep_gemm paged flip (default off).
#
# The shipping cap-12 path is the per-(b,n) python fallback below. The pinned
# deep_gemm (+ee6161b) now ships sm120_fp8_paged_mqa_logits, but its host side
# hard-asserts block_kv == 64 for fp8 on SM120 (fp4 allows 32/64; SM100 allows
# 32/64/128). This engine's INDEXER_KV entries_per_block is 32
# (kernel_seq_size_per_block 128 -> 128/4), so enabling the flag at the shipped
# geometry would DG_HOST_ASSERT-crash the engine on the first decode call.
# Therefore the flag only takes effect when block_size == 64; otherwise we keep
# the fallback and say so once, so an A/B can never silently crash.
_INDEXER_FP8_DEEPGEMM_PAGED = (
    os.environ.get("DSV4_INDEXER_FP8_DEEPGEMM_PAGED", "0") == "1"
)
_PAGED_GEOM_WARNED = [False]

# [IDXDIAG] shape marker + timing-ablation arm for the SM120 paged fallback.
# Decode runs under CUDA-graph replay, so host timings here only see the capture
# pass; DSV4_INDEXER_SM120_NULL=1 removes the loop entirely so the ENGINE-level
# cost can be priced by A/B on TPOT (the captured graph then holds none of the
# gather/cast/einsum kernels). Both default off:
#   DSV4_INDEXER_SM120_DIAG=1  -> print up to 500 shape lines per process
#   DSV4_INDEXER_SM120_NULL=1  -> ablation arm (same allocation, no work)
_INDEXER_SM120_DIAG = os.environ.get("DSV4_INDEXER_SM120_DIAG", "0") == "1"
_INDEXER_SM120_NULL = os.environ.get("DSV4_INDEXER_SM120_NULL", "0") == "1"
_IDX_DIAG_BUDGET = [500 if _INDEXER_SM120_DIAG else 0]
_IDX_PAGED_DIAG = [200]


def sm120_paged_deepgemm_ready(block_size: int) -> bool:
    """True when the SM120 paged deep_gemm path may be used for this geometry."""
    if not _INDEXER_FP8_DEEPGEMM_PAGED:
        return False
    if block_size != 64:
        if not _PAGED_GEOM_WARNED[0]:
            _PAGED_GEOM_WARNED[0] = True
            import sys

            print(
                "[IDX-PAGED] DSV4_INDEXER_FP8_DEEPGEMM_PAGED=1 but INDEXER_KV "
                "entries_per_block=%d != 64 (the only block size the SM120 fp8 "
                "paged kernel supports); keeping the SM120 fallback." % block_size,
                file=sys.stderr,
                flush=True,
            )
        return False
    if _HAS_DEEP_GEMM and _IDX_PAGED_DIAG[0] > 0:
        _IDX_PAGED_DIAG[0] -= 1
        import sys

        print(
            "[IDXPAGED] SM120 paged deep_gemm ENGAGED block_size=%d" % block_size,
            file=sys.stderr,
            flush=True,
        )
    return _HAS_DEEP_GEMM


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
) -> torch.Tensor:
    """One-shot FP8 paged indexer logits via DeepGEMM.

    Returns ``[B*next_n, max_ctx_len] fp32`` — feed straight to topk.
    Padded columns past per-row ``context_lens[b, n]`` are left as
    whatever DeepGEMM writes (use ``clean_logits=True`` if the
    downstream topk needs ``-inf`` there; default False to save the
    extra mask).
    """
    if (
        q_fp8.is_cuda
        and torch.cuda.get_device_capability(q_fp8.device)[0] == 12
        and not sm120_paged_deepgemm_ready(block_size)
    ):
        return _fp8_paged_indexer_score_sm120(
            q_fp8, w_fold, kv_pool_uint8, block_table, context_lens,
            block_size, max_ctx_len,
        )
    assert _HAS_DEEP_GEMM, "deep_gemm.fp8_paged_mqa_logits not available"
    assert q_fp8.dtype == torch.float8_e4m3fn, f"q_fp8 dtype={q_fp8.dtype}"
    assert q_fp8.dim() == 4 and q_fp8.shape[-1] == INDEXER_HEAD_DIM
    assert w_fold.dtype == torch.float32 and w_fold.dim() == 2
    assert kv_pool_uint8.dtype == torch.uint8
    assert kv_pool_uint8.shape[-1] == INDEXER_ENTRY_BYTES
    assert block_table.dtype == torch.int32 and block_table.dim() == 2
    assert context_lens.dtype == torch.int32 and context_lens.dim() == 2
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


def _fp8_paged_indexer_score_sm120(
    q_fp8: torch.Tensor,
    w_fold: torch.Tensor,
    kv_pool_uint8: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_ctx_len: int,
) -> torch.Tensor:
    B, next_n, H, D = q_fp8.shape
    rows = B * next_n
    if _IDX_DIAG_BUDGET[0] > 0:
        _IDX_DIAG_BUDGET[0] -= 1
        import sys

        print(
            "[IDXDIAG] B=%d next_n=%d H=%d T_max=%d block=%d cap=%d null=%d"
            % (
                B, next_n, H, max_ctx_len, block_size,
                torch.cuda.is_current_stream_capturing(),
                _INDEXER_SM120_NULL,
            ),
            file=sys.stderr,
            flush=True,
        )
    if _INDEXER_SM120_NULL:
        # ablation arm: same allocation, none of the per-(b,n) work
        return torch.full(
            (rows, max_ctx_len), float("-inf"), dtype=torch.float32,
            device=q_fp8.device,
        )
    out = torch.full(
        (rows, max_ctx_len), float("-inf"), dtype=torch.float32,
        device=q_fp8.device,
    )
    q = q_fp8.float().view(rows, H, D)
    weights = w_fold.float().view(rows, H)
    byte_pool = kv_pool_uint8.reshape(-1, block_size * INDEXER_ENTRY_BYTES)
    capturing = torch.cuda.is_current_stream_capturing()
    for b in range(B):
        for n in range(next_n):
            row = b * next_n + n
            length = max_ctx_len if capturing else min(
                int(context_lens[b, n].item()), max_ctx_len
            )
            if length <= 0:
                continue
            pos = torch.arange(length, device=q_fp8.device, dtype=torch.long)
            block_ids = block_table[b].long().index_select(
                0, pos // block_size
            ).clamp_min_(0)
            block_rows = byte_pool.index_select(0, block_ids)
            offsets = pos.remainder(block_size)
            k_cols = offsets[:, None] * D + torch.arange(D, device=q_fp8.device)
            k_fp8 = block_rows.gather(1, k_cols).contiguous().view(torch.float8_e4m3fn).float()
            s_cols = block_size * D + offsets[:, None] * 4 + torch.arange(4, device=q_fp8.device)
            k_scale = block_rows.gather(1, s_cols).contiguous().view(torch.float32).view(-1)
            k = k_fp8 * k_scale[:, None]
            per_head = torch.einsum("hd,td->ht", q[row], k).relu_()
            score = torch.einsum("h,ht->t", weights[row], per_head)
            if capturing:
                out[row] = torch.where(pos < context_lens[b, n], score, out[row])
            else:
                out[row, :length] = score
    return out
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
    if (
        q_fp8.is_cuda
        and torch.cuda.get_device_capability(q_fp8.device)[0] == 12
        and not _FP8_DEEPGEMM_ON_SM120
    ):
        from rtp_llm.models_py.modules.dsv4._indexer_score_triton import (
            v4_indexer_score,
        )
        M_rows = q_fp8.shape[0]
        N_cols = k_quant.shape[0]
        banded = (
            os.environ.get("DSV4_INDEXER_BANDED", "0") == "1"
            and M_rows > 0
            and N_cols > 0
        )
        if banded:
            # L3 causal-band fix (Sep 3): the dense SM120 fallback scores the
            # FULL [M, N] axis even though row m only ever reads columns
            # [ks[m], ke[m]) (the vendored topk never touches out-of-window
            # entries and clean_logits re-masks anyway). Prefill rows ascend
            # with position, so ke grows monotonically down the chunk: split
            # rows into bands and score each against only its column range.
            # Microbench (bench/indexer_score_microbench.py): dense scaling is
            # exactly quadratic (x4.0 per ISL doubling; 20.7 ms @32K shape)
            # and band_frac = 0.5 → ~2x saving at 32K, growing with ISL.
            band_rows = int(os.environ.get("DSV4_INDEXER_BAND_ROWS", "1024"))
            q_bf16 = q_fp8.to(torch.bfloat16).unsqueeze(0).contiguous()
            k_bf16 = (k_quant.float() * k_scale.float()[:, None]).to(
                torch.bfloat16
            ).unsqueeze(0).contiguous()
            w_f32 = w_fold.float().unsqueeze(0).contiguous()
            # ONE small DtoH per call (S4 fix, Sep 3): band bounds are
            # reduced on-device per band, and only [nb, 2] int32 crosses to
            # the host (~64 B). The previous form copied the FULL
            # cu_seqlen_ke/ks arrays as int64 (2 x 64 KiB at 32K ISL — the
            # per-layer 65536-B DtoH class in every trace since q2prof) plus
            # the int64-widening elementwise kernels, only to take per-band
            # max/min on the host. Values are identical (max/min over the
            # same rows; the partial tail band is reduced on its real
            # slice), so the scored region is unchanged.
            n_bands = (M_rows + band_rows - 1) // band_rows
            n_full = M_rows // band_rows
            ke_parts, ks_parts = [], []
            if n_full:
                ke_parts.append(
                    cu_seqlen_ke[: n_full * band_rows]
                    .view(n_full, band_rows)
                    .max(dim=1)
                    .values
                )
                ks_parts.append(
                    cu_seqlen_ks[: n_full * band_rows]
                    .view(n_full, band_rows)
                    .min(dim=1)
                    .values
                )
            rem = M_rows - n_full * band_rows
            if rem:
                ke_parts.append(cu_seqlen_ke[n_full * band_rows :].max().view(1))
                ks_parts.append(cu_seqlen_ks[n_full * band_rows :].min().view(1))
            band_bounds = torch.stack(
                [torch.cat(ke_parts), torch.cat(ks_parts)], dim=1
            ).to("cpu", torch.int64)
            ke_host_b = band_bounds[:, 0].tolist()
            ks_host_b = band_bounds[:, 1].tolist()
            if os.environ.get("DSV4_DIAG") and _BAND_BOUNDS_CT[0] < 3:
                _BAND_BOUNDS_CT[0] += 1
                import sys
                print("[S4-BAND] rank=%d M=%d bands=%d packed_dtoh_bytes=%d" % (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized() else -1,
                    M_rows, n_bands,
                    band_bounds.numel() * band_bounds.element_size()),
                    file=sys.stderr, flush=True)
            out = torch.empty(
                (M_rows, N_cols), dtype=torch.float32, device=q_fp8.device)
            for b_i, r0 in enumerate(range(0, M_rows, band_rows)):
                r1 = min(r0 + band_rows, M_rows)
                ke_max = min(int(ke_host_b[b_i]), N_cols)
                ks_min = max(int(ks_host_b[b_i]), 0)
                if ke_max <= ks_min:
                    continue
                sub = v4_indexer_score(
                    q_bf16[:, r0:r1].contiguous(),
                    k_bf16[:, ks_min:ke_max].contiguous(),
                    w_f32[:, r0:r1].contiguous(),
                ).squeeze(0)
                out[r0:r1, ks_min:ke_max] = sub
            if clean_logits:
                positions = torch.arange(N_cols, device=q_fp8.device).unsqueeze(0)
                valid = (positions >= cu_seqlen_ks.long().unsqueeze(1)) & (
                    positions < cu_seqlen_ke.long().unsqueeze(1)
                )
                out.masked_fill_(~valid, float("-inf"))
            return out
        q_bf16 = q_fp8.to(torch.bfloat16).unsqueeze(0).contiguous()
        k_bf16 = (k_quant.float() * k_scale.float()[:, None]).to(
            torch.bfloat16
        ).unsqueeze(0).contiguous()
        out = v4_indexer_score(
            q_bf16, k_bf16, w_fold.float().unsqueeze(0).contiguous()
        ).squeeze(0)
        rows, cols = out.shape
        if clean_logits:
            positions = torch.arange(cols, device=q_fp8.device).unsqueeze(0)
            valid = (positions >= cu_seqlen_ks.long().unsqueeze(1)) & (
                positions < cu_seqlen_ke.long().unsqueeze(1)
            )
            out.masked_fill_(~valid, float("-inf"))
        return out
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
