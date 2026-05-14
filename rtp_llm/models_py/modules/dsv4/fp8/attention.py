"""DeepSeek-V4 Attention with HCA / CSA / SWA-only path selection.

Direct port of `inference/model.py:Attention` (BF16-only, mock per-layer
KV cache via register_buffer). Skips Hadamard rotate / FP4 / FP8 quant.

Layer schedule via `compress_ratio`:
  0   -> SWA-only (no Compressor, no Indexer)
  4   -> CSA (Compressor with overlap=True + Indexer for sparse top-k)
  128 -> HCA (Compressor with overlap=False, dense compressed MQA)

Sparse attention reference uses `gather`-based PyTorch implementation —
slow but correct. M6 will swap in FlashMLA sparse impl.
"""

import os
from typing import Any, Dict, NamedTuple, Optional, Union

# P3 (audit §3.5 / §7.4 P0): wo_a batched output projection.
# Replaces the per-group ``for g in range(G)`` loop (G launches of
# ``fp8_gemm_nt``) with one ``deep_gemm.fp8_einsum("bhr,hdr->bhd", ...)``
# call.  Matches vLLM's ``deepseek_v4_attention.py:325`` exactly (same
# API, same recipe).  Validated by ``test/test_wo_a_batched_vs_loop.py``
# — bit-identical output + 3.5-4.4× speedup at decode B∈{1..256}.
import deep_gemm  # noqa: E402
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_gemm.utils.layout import (  # noqa: E402
    get_mn_major_tma_aligned_packed_ue8m0_tensor,
)

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)

# Audit §7.4 P0 (row 1) + §7.3.4: fused RMSNorm + partial RoPE, single
# Triton launch.  Covers every Q/KV decode + prefill site.  Standalone
# (no-RoPE) RMSNorm sites (``_rmsnorm_weighted``) use the framework C++
# ``rtp_llm_ops.rmsnorm`` (matches vLLM — bf16 weight).
# Validated by test_fused_rmsnorm_rope.py (bf16 <=1-ULP + 1.25-1.75x).
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    build_cp_full_prefill_positions,
    cp_all_gather_full,
    cp_freqs_cis_local,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8
from rtp_llm.models_py.modules.dsv4.qlinear import _fp8_dequant_to_fp32
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb, precompute_freqs_cis
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.ops.compute_ops import KVCacheRegionName, rtp_llm_ops


# int attn_type id → pybind11 ``KVCacheRegionName`` enum.  The
# ``KVCache::getLayerCache`` pybind11 overload takes ``(int, KVCacheRegionName)``
# — passing a raw Python int for the second arg raises TypeError which the
# ``except`` clauses in ``_pool_view`` / ``_pool_entries_per_block`` swallow,
# collapsing the pool read to None (and eventually the continuation-prefill
# ``_gather_kv_cache_dense_from_pool`` assert).  Map ints → enums up front so
# the pybind dispatch succeeds.  Mirrors
# ``prefill/forward.py::DSv4WriteCacheStoreOp._ATTN_TYPE_ENUM_BY_INT``.
def _build_attn_type_enum_map() -> Dict[int, "KVCacheRegionName"]:
    from rtp_llm.models_py.modules.dsv4.attn_type import (
        CSA_KV,
        CSA_STATE,
        HCA_KV,
        HCA_STATE,
        INDEXER_KV,
        INDEXER_STATE,
        SWA_KV,
    )

    return {
        CSA_KV: KVCacheRegionName.CSA_KV,
        HCA_KV: KVCacheRegionName.HCA_KV,
        INDEXER_KV: KVCacheRegionName.INDEXER_KV,
        INDEXER_STATE: KVCacheRegionName.INDEXER_STATE,
        CSA_STATE: KVCacheRegionName.CSA_STATE,
        HCA_STATE: KVCacheRegionName.HCA_STATE,
        SWA_KV: KVCacheRegionName.SWA_KV,
    }


_ATTN_TYPE_ENUM_BY_INT: Dict[int, KVCacheRegionName] = _build_attn_type_enum_map()


# Phase E1 (dsv4_kvcache_native_refactor_plan.md §9): route prefill
# continuation reads through the framework BlockPool instead of the
# register_buffer mirror.  Phase B dual-write keeps the pool fresh each
# forward, so ``self.kv_cache[:bsz]`` and the pool gather are byte-equal
# on all well-defined positions (sentinel / uninitialized slots return
# zero in both paths).  Default ON; ``DSV4_READ_FROM_POOL=0`` restores
# the legacy register_buffer read for regression bisection.
def _use_read_from_pool() -> bool:
    return os.environ.get("DSV4_READ_FROM_POOL", "1") != "0"


# Phase-1 varlen migration kill-switch. While the per-builder bodies are
# being switched from B==1 scalar plumbing to ``cu_seqlens``/``position_ids``
# per-request plumbing (Phase 2 SWA, Phase 3a/3b CSA·HCA), each new code
# path checks this flag at the dispatch point. Default ON once a phase
# lands; ``DSV4_VARLEN_PREFILL=0`` forces the legacy B==1 path so a
# regression can be bisected without reverting the patch series.
def _use_varlen_prefill() -> bool:
    return os.environ.get("DSV4_VARLEN_PREFILL", "1") != "0"


def _flat_1d(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).contiguous()


_V4_FP8_BLOCK_CFG = Fp8BlockWiseQuantConfig()

_DSV4_FP8_KV_ENTRY_BYTES = 584
_DSV4_FP8_INDEXER_ENTRY_BYTES = 132


def _repack_v4_fp8_scale_to_int32(scale: torch.Tensor) -> torch.Tensor:
    """V4 ckpt UE8M0 ``[N/128, K/128]`` → DeepGEMM ``[N, K/128]`` UE8M0
    int32-packed TMA-aligned scale.  Row-repeats by 128 along N so each
    weight row gets its own scale row, then hands off to DeepGEMM's
    ``get_mn_major_tma_aligned_packed_ue8m0_tensor`` (column-major,
    int32-packed).  Must be called on-device (DeepGEMM helper is CUDA)."""
    assert scale.dtype == torch.float8_e8m0fnu, f"unexpected scale dtype {scale.dtype}"
    assert scale.dim() == 2, f"unexpected scale dim {scale.dim()}"
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

    N_blk, _ = scale.shape
    N = N_blk * 128
    idx = torch.arange(N, device=scale.device) // 128
    scale_rep = scale.float().index_select(-2, idx)
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)


def _prepare_wo_a_stacked(
    weight_fp8: torch.Tensor,
    scale_raw: torch.Tensor,
    G: int,
    R: int,
    K: int,
) -> tuple:
    """Stack V4 wo_a ckpt (``[G*R, K]`` fp8 + ``[G*R/128, K/128]`` e8m0fnu)
    into the ``fp8_einsum``-expected layout, computed once at init:

    - weight: ``[G, R, K]`` fp8 contiguous (free view)
    - scale : ``[G, R, K/512]`` int32 UE8M0 packed, MN-major TMA-aligned
      (stride ``(K/512 * tma_R, 1, tma_R)``)

    Cast e8m0fnu → fp32, row-repeat by 128 along R so
    ``get_mn_major_tma_aligned_packed_ue8m0_tensor`` (which operates on
    fp32 ``[*, mn, k/128]``) sees the full [G, R, K/128] grid.  The helper
    floor-log2 bitcasts each block scale and packs 4 UE8M0 bytes per
    int32; output shape ``[G, R, K/512]`` matches
    ``deep_gemm.fp8_einsum(..., recipe=(1, 1, 128))`` expectations."""
    w_stk = weight_fp8.view(G, R, K).contiguous()
    scale_fp32 = scale_raw.float().view(G, R // 128, K // 128)
    idx = torch.arange(R, device=scale_raw.device) // 128
    scale_rep = scale_fp32.index_select(-2, idx).contiguous()  # [G, R, K/128]
    s_stk = get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)
    return w_stk, s_stk


def _v4_fp8_linear(w: torch.Tensor, s: torch.Tensor):
    """Build a CudaFp8DeepGEMMLinear from raw V4 FP8 weight + scale tensors.

    Repacks the UE8M0 ``float8_e8m0fnu`` scale into DeepGEMM's int32
    TMA-aligned packed layout when needed. Framework descriptor path may
    deliver the scale already packed (dtype int32) — we no-op then."""
    assert s is not None, "expected non-null FP8 scale"
    if s.dtype == torch.float8_e8m0fnu:
        s = _repack_v4_fp8_scale_to_int32(s)
    # LinearFactory.create_linear_from_weights consumes a (weights_dict,
    # weight_key, scale_key) triple — feed it a one-shot dict so the
    # factory plumbing is unchanged.
    local = {"_w": w, "_s": s}
    return LinearFactory.create_linear_from_weights(
        local,
        "_w",
        "_s",
        quant_config=_V4_FP8_BLOCK_CFG,
    )


def _v4_fp8_linear_from_dict(weights: dict, weight_key: str, scale_key: str):
    """Backwards-compat bridge over ``_v4_fp8_linear`` for callers that
    still pass a flat dict + keys.  Mutates ``weights[scale_key]`` to the
    packed form so subsequent callers don't repack."""
    w = weights[weight_key]
    s = weights[scale_key]
    if s.dtype == torch.float8_e8m0fnu:
        s = _repack_v4_fp8_scale_to_int32(s)
        weights[scale_key] = s
    return _v4_fp8_linear(w, s)


def _get_window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int, device
) -> torch.Tensor:
    """Returns int64 [bsz, seqlen, window_size] with linear absolute KV indices."""
    if start_pos > 0 and seqlen > 1:
        # Continuation prefill uses a dense absolute SWA view reconstructed
        # from the pool. Per-position topk therefore stays in absolute token
        # coordinates, not final ring slots; otherwise earlier suffix tokens
        # read ring entries overwritten by later suffix tokens.
        base = torch.arange(start_pos, start_pos + seqlen, device=device).unsqueeze(1)
        offs = torch.arange(window_size, device=device)
        window_start = (base - window_size + 1).clamp_min(0)
        matrix = window_start + offs
        matrix = torch.where(matrix > base, -1, matrix)
    elif start_pos >= window_size - 1:
        sp = start_pos % window_size
        matrix = torch.cat(
            [
                torch.arange(sp + 1, window_size, device=device),
                torch.arange(0, sp + 1, device=device),
            ],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1, device=device),
            (0, window_size - start_pos - 1),
            value=-1,
        )
    else:
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size), device=device
        )
        matrix = torch.where(matrix > base, -1, matrix)
        if matrix.size(1) < window_size:
            matrix = F.pad(matrix, (0, window_size - matrix.size(1)), value=-1)
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _get_window_topk_idxs_cp(
    window_size: int,
    bsz: int,
    seq_len_total: int,
    global_positions: torch.Tensor,
    use_ring_layout: bool = False,
) -> torch.Tensor:
    """CP-prefill variant: each rank-local Q token at local index i sits
    at GLOBAL position g = global_positions[i].  Its sliding window
    reads KV at global positions [max(0, g-win+1), g+1).

    Fresh CP prefill uses the all-gathered linear ``kv_full`` layout.  CP
    continuation prefill reconstructs the same linear absolute view from the
    paged SWA pool; ``use_ring_layout`` is kept for legacy ring callers.

    Returns [bsz, S_local, window_size] int64 with valid indices at the
    START of each row (slots 0..k-1) and -1 padding at the END — matching
    the non-CP ``_get_window_topk_idxs`` layout.  This slot ordering is
    LOAD-BEARING for sparse-attn numerical equivalence: the TileLang
    sparse_attn kernel's per-block fp32 reductions are not invariant to
    the position of valid vs masked slots within a 64-wide block, so any
    deviation from the non-CP layout leaks ~1 BF16 ULP per layer of
    noise that compounds across 43 layers and shifts greedy decode onto
    OOV vocab indices.  See `project_dsv4_cp_ep_wrong_output` memory.

    Entries beyond each row's valid window are -1 so the sparse_attn
    kernel masks them out.
    """
    device = global_positions.device
    S_local = int(global_positions.shape[0])
    if use_ring_layout:
        offsets = torch.arange(window_size, device=device)  # [win]
        base = global_positions.unsqueeze(1)  # [S_local, 1]
        idxs = (base % window_size + 1 + offsets.unsqueeze(0)) % window_size
        valid_count = torch.clamp(global_positions + 1, max=window_size)
        invalid = offsets.unsqueeze(0) < (window_size - valid_count.unsqueeze(1))
        matrix = torch.where(invalid, torch.full_like(idxs, -1), idxs)
        return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()

    W = min(window_size, max(seq_len_total, 1))
    # Per-row window start, clamped to 0 for early Q positions whose
    # left edge would be negative.  Matches `_get_window_topk_idxs`'s
    # `(base - window_size + 1).clamp(0)` at start_pos == 0.
    base = global_positions.unsqueeze(1)  # [S_local, 1]
    window_start = (base - W + 1).clamp_min(0)  # [S_local, 1]
    offs = torch.arange(W, device=device)  # [W]
    matrix = window_start + offs  # [S_local, W] — valid kv indices left-aligned
    # Mask out positions that are causally future (matrix > g) or land
    # past the unpadded sequence end (matrix >= seq_len_total, which
    # protects rank-local padding slots whose global_position is beyond end).
    invalid = (matrix > base) | (matrix >= seq_len_total)
    matrix = torch.where(invalid, torch.full_like(matrix, -1), matrix)
    if W < window_size:
        # Tail-pad with -1 so the topk slot count matches the non-CP path.
        pad = torch.full(
            (S_local, window_size - W), -1, dtype=matrix.dtype, device=device
        )
        matrix = torch.cat([matrix, pad], dim=1)
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _get_window_topk_idxs_varlen(
    window_size: int,
    cu_seqlens: torch.Tensor,  # [B+1] int32 — flat-axis req boundaries
    position_ids: torch.Tensor,  # [T_total] int32/int64 — global abs pos per token
    prefix_lengths: torch.Tensor,  # [B] int32 — per-req sp (≥0)
    req_id_per_token: torch.Tensor,  # [T_total] int32 — req each token belongs to
) -> torch.Tensor:
    """Returns ``[T_total, window_size]`` **int32** flat-KV indices.

    For request b, token t at local pos p = position_ids[t] - prefix_lengths[b]:
      * window covers local positions [max(0, p - win + 1), p]
      * flat indices = cu_seqlens[b] + (those local positions), all within
        [cu_seqlens[b], cu_seqlens[b] + S_b)
      * tail slots beyond the valid window get -1 (kernel masks them out)

    Only consumed by the cold ``_attn_fp8_swa_via_kv_full`` path. Continuation
    prefill uses ``combined_indices`` (workspace coordinates, M*batch_idx+slot)
    via ``_attn_fp8_swa_via_concat``, so prefix-tail KV does NOT need to be
    represented here.

    **CP alignment:** under cp_size > 1 the caller passes GLOBAL per-request
    positions for each rank-local token plus the full per-request
    ``cu_seqlens`` view. The formula therefore emits row indices into the
    all-gathered ``kv_full[seq_len_full]`` while preserving request
    boundaries for B>=1.

    Mirrors the right-pad slot ordering of ``_get_window_topk_idxs`` (sparse_attn
    kernel block reductions are not invariant to slot ordering — see comment
    on ``_get_window_topk_idxs_cp`` line 281+).

    **Dtype:** internal math is int32 — every value (T_total, max_seq_len,
    batch_size, accumulated cu_seqlens) is bounded by max_seq_len * batch_size
    which is well within int32 (≪ 2^31). Only ``req_id_per_token`` is cast
    to int64 because ``torch.gather`` requires int64 indices. The downstream
    ``_attn_fp8_swa_via_kv_full`` casts to int32 anyway, so int32 here saves
    a 64MB → 32MB allocation at T=16K, win=512.
    """
    position_ids = _flat_1d(position_ids)
    req_id_per_token = _flat_1d(req_id_per_token)
    cu_seqlens = _flat_1d(cu_seqlens)
    prefix_lengths = _flat_1d(prefix_lengths)
    assert position_ids.numel() == req_id_per_token.numel(), (
        "position_ids / req_id_per_token must have matching token counts: "
        f"{position_ids.numel()} vs {req_id_per_token.numel()}"
    )
    device = position_ids.device

    # gather() requires int64 indices — this is the only mandatory long cast.
    req_id_idx = req_id_per_token.to(device=device, dtype=torch.long)

    # Source tensors normalized to int32 (their natural framework dtype).
    cu_seqlens_i32 = cu_seqlens.to(device=device, dtype=torch.int32)
    prefix_lengths_i32 = prefix_lengths.to(device=device, dtype=torch.int32)
    positions_i32 = position_ids.to(device=device, dtype=torch.int32)

    # Per-token broadcast of the request-level scalars: each token learns its
    # own request's prefix length and flat-axis start offset.
    prefix_per_token = prefix_lengths_i32.gather(0, req_id_idx)  # [T_total] int32
    req_start_in_flat = cu_seqlens_i32.gather(0, req_id_idx)  # [T_total] int32

    # local_query_pos[t] = position of token t within ITS OWN request
    # (0-indexed). For request b's i-th new token: positions[t] = prefix[b] + i
    # so local_query_pos[t] = i.
    local_query_pos = positions_i32 - prefix_per_token  # [T_total] int32

    # Build the [T_total, window_size] grid of LOCAL kv positions covered by
    # each token's window: [max(0, p - win + 1), max(0, p - win + 1) + win).
    # ``window_local_start`` is per-token; ``window_offset`` is shared across
    # all tokens. Right-pad with -1 where the offset overshoots the query's
    # own position (causal mask) — that ordering is the sparse_attn kernel's
    # numerical-equivalence requirement.
    window_offset = torch.arange(
        window_size, device=device, dtype=torch.int32
    )  # [W] int32
    query_pos_col = local_query_pos.unsqueeze(1)  # [T_total, 1] int32
    window_local_start = (query_pos_col - window_size + 1).clamp_min(0)
    window_local_idx = window_local_start + window_offset  # [T_total, W] int32

    is_causal_pad = window_local_idx > query_pos_col  # right-pad mask
    window_flat_idx = req_start_in_flat.unsqueeze(1) + window_local_idx
    return torch.where(
        is_causal_pad,
        torch.full_like(window_flat_idx, -1),
        window_flat_idx,
    ).contiguous()


class SwaPrefillMeta(NamedTuple):
    """FP8 prefill metadata bundle — built once per ``_prefill_common_setup``
    call for **all FP8 KV-cache layers** (compress_ratio 0/4/128 alike).

    Two field groups with different lifecycles:

    1. **FP8 KV cache write metadata** (used by ``_prefill_write_swa_fp8_paged``)
       — built for every FP8 layer regardless of ``compress_ratio``,
       because BF16 ``_prefill_write_swa_to_pool`` is also unconditional
       across compress_ratio. CSA/HCA layers still need to populate the
       SWA pool for downstream decode reads.

       Fields: ``slot_mapping``, ``query_start_loc``, ``combined_seq_lens``.
       ``None`` on warmup forward (``self._kv_cache is None``).

    2. **SWA-only attention metadata** (used by ``_attn_fp8_swa_via_kv_full``
       / ``_attn_fp8_swa_via_concat``) — built only when
       ``compress_ratio == 0``. CSA/HCA layers don't read the SWA pool
       directly during attention (they go through compressor/indexer).

       ``topk_length_kv_full`` is cache-independent so it's also set on
       warmup. ``cache_*`` / ``combined_indices`` etc. are skipped on
       warmup or non-SWA-only layers.
    """

    # Group 1: FP8 KV cache write meta (all FP8 layers)
    slot_mapping: Optional[torch.Tensor]  # [num_tokens] int64; -1 = skip
    query_start_loc: Optional[torch.Tensor]  # [B+1] int32
    combined_seq_lens: Optional[torch.Tensor]  # [B] int32 — sp + S per req

    # Group 2: SWA-only attention meta (compress_ratio == 0 only)
    topk_length_kv_full: Optional[torch.Tensor]  # [num_tokens] int32
    combined_gather_lens: Optional[torch.Tensor]  # [B] int32 — P + S per req
    combined_gather_len_max: int
    M: int  # workspace stride
    cache_seq_lens: Optional[torch.Tensor]  # [B] int32 — sp per req
    cache_gather_lens: Optional[torch.Tensor]  # [B] int32 — P = min(sp, win-1)
    prefix_len_max: int
    combined_indices: Optional[torch.Tensor]  # [num_tokens, combined_topk] int32
    combined_lens: Optional[torch.Tensor]  # [num_tokens] int32
    # Pre-computed per-token scatter target into the flat [B*M, D] workspace
    # view (= ``req_id * M + min(prefix, win-1) + local_pos``). Consumed by
    # ``_attn_fp8_swa_via_concat`` step-2 ``index_copy_``. Populated only on
    # the varlen continuation branch where via_concat actually runs; legacy
    # B==1 single-slice copy doesn't need it.
    slot_in_flat: Optional[torch.Tensor]  # [num_tokens] int64


class WorkspaceMeta(NamedTuple):
    """Static index/dim metadata for the vLLM-style workspace + dual-gather
    + ``combine_topk_swa_indices`` + ``flash_mla_sparse_fwd`` flow used by
    both CSA and HCA paths. Built once per (forward, ratio) by
    :meth:`Attention._build_workspace_meta`.

    Workspace layout under varlen B>=1 — **N_max-padded** so the per-request
    compressed and SWA regions land at the same column offset across the
    batch (lets ``combine_topk_swa_indices`` keep its scalar ``M`` / ``N``
    contract). For each request b in ``workspace[b, :, :]``:

        ``[0,             N_b              )`` — request b compressed
        ``[N_b,           N_max            )`` — zero pad
        ``[N_max,         N_max + gather_b )`` — request b SWA stream
                  (first ``P_b = min(sp_b, win-1)`` rows are prefix tail
                  dequant'd from pool; next ``S_b`` rows are overwritten
                  by fresh BF16 new K via ``new_k_slot_in_flat``)
        ``[N_max+gather_b, M               )`` — zero pad

    with ``N_max = max_b N_b``, ``gather_len_max = max_b gather_b``,
    ``M = N_max + gather_len_max``. For B==1 / DSV4_VARLEN_PREFILL=0 the
    formulas collapse: ``N_max = N``, ``gather_len_max = gather_len``,
    ``M = N + gather_len``.

    Every elementwise operation needed by ``_attn_via_workspace`` is
    pre-baked here so the hot path stays kernel-only (dequant ×2 +
    ``index_copy_`` + ``combine_topk`` + ``flash_mla_sparse_fwd``).
    """

    M: int  # workspace stride per batch (= N_max + gather_len_max)
    N: int  # compressed region stride (= N_max under varlen)
    swa_eb: int
    cmp_eb: int
    swa_bt_int32: torch.Tensor  # [B, max_blocks_per_seq] int32 contig
    cmp_bt_int32: torch.Tensor
    swa_seq_lens: torch.Tensor  # [B] int32 — total SWA stream length per req
    cmp_seq_lens: torch.Tensor  # [B] int32 — per-req compressed pool length
    swa_gather_lens: torch.Tensor  # [B] int32 — per-req workspace SWA length (P + S)
    swa_cache_seq_lens: torch.Tensor  # [B] int32 — cached prefix length (sp)
    swa_cache_gather_lens: torch.Tensor  # [B] int32 — cached prefix tail length (P)
    qsl: torch.Tensor  # [B+1] int32 — query_start_loc (== cu_seqlens)
    # HCA only: precomputed dense compressed topk [T_total, N_max] int32
    # (each row = arange(N_max)). The ``_combine_topk_swa_indices_kernel``
    # masks each row down to ``min((pos+1)//ratio, TOP_K)`` per token from
    # ``COMPRESS_RATIO`` + per-request ``seq_lens`` / ``query_start_loc``,
    # so even tokens whose request has ``N_b < N_max`` only see valid
    # compressed slots in ``[0, (pos+1)//ratio) ⊆ [0, N_b)``. CSA gets
    # None; runtime indexer output is fed to combine_topk instead.
    dense_cmp_topk: Optional[torch.Tensor]
    # Per-token scatter target into ``workspace.view(B*M, D)`` for the new
    # K BF16 overlay. Pre-baked here so the hot path is a single
    # ``index_copy_`` regardless of B / mixed prefix.
    #   slot_in_flat[t] = req_id_per_token[t] * M
    #                     + N           # = N_max ⇒ start of SWA region
    #                     + min(prefix_lengths[req(t)], win - 1)
    #                     + (position_ids[t] - prefix_lengths[req(t)])
    # B==1 collapses to a contiguous ``arange(seqlen) + (N + P)`` so the
    # ``index_copy_`` is bit-equal to the legacy ``copy_`` slice write.
    new_k_slot_in_flat: torch.Tensor  # [T_total] int64


class CsaPrefillMeta(NamedTuple):
    """CSA-layer prefill metadata (compress_ratio == 4). Carries the
    nested indexer metadata + the main CSA compressor write metadata so
    the per-layer ``_forward_prefill_csa`` is kernel-only."""

    indexer_meta: Any  # IndexerFP8PrefillMeta — sparse topk source
    compressor_meta: Any  # CompressorMeta — main CSA pool write
    workspace_meta: Optional[WorkspaceMeta]


class HcaPrefillMeta(NamedTuple):
    """HCA-layer prefill metadata (compress_ratio == 128). Carries the
    main HCA compressor write metadata. HCA generates dense compressed
    indices in-line so no indexer is involved."""

    compressor_meta: Any  # CompressorMeta — main HCA pool write
    workspace_meta: Optional[WorkspaceMeta]


class PrefillMeta(NamedTuple):
    """Per-call prefill metadata, layer-invariant within a
    ``compress_ratio`` bucket. Built once per (forward, ratio) by
    :meth:`Attention._build_shared_prefill_meta` and broadcast to every
    same-ratio layer via :meth:`Attention._set_prefill_meta_shared`.

    The three sub-metadata fields are mutually exclusive — exactly one
    is non-None per layer, gated by ``compress_ratio``:
      * ``compress_ratio == 0``   → ``swa_meta`` only (SWA-only path)
      * ``compress_ratio == 4``   → ``swa_meta`` + ``csa_meta``
      * ``compress_ratio == 128`` → ``swa_meta`` + ``hca_meta``

    ``swa_meta`` is set on every FP8 layer because all paths still
    write the SWA pool for downstream decode.
    """

    seqlen: int
    seqlen_full: int  # CP-aware (== seqlen when CP off); reused by compress / pool
    rd: int
    device: torch.device
    cp_ctx: Optional[CPContext]
    cp_on: bool
    freqs_cis: torch.Tensor
    topk_idxs: torch.Tensor
    sp_int: int
    any_cont: bool
    row_seqlens_full: torch.Tensor  # [1] long — for SWA pool helpers
    # Phase-1 varlen plumbing — populated by upper-layer broadcast builder.
    # All ``None`` on the standalone (B==1) path so existing logic that
    # consults ``sp_int`` / ``seqlen`` continues to work bit-equally;
    # varlen-aware code paths added in later phases (attention / indexer /
    # compressor) prefer these tensors when present.
    #
    #   batch_size       : B — number of requests packed in this prefill
    #   cu_seqlens       : [B+1] int32/int64 — per-request token offsets
    #                      into the flat [T_total] axis
    #   input_lengths    : [B] int32 — per-request new-token count (S_b)
    #   prefix_lengths   : [B] int32 — per-request prior context length
    #                      (== absolute start_pos of each request, i.e.
    #                      ``sp_per_req``); kept under both names because
    #                      framework plumbing uses ``prefix_lengths`` and
    #                      DSV4 internals use ``sp_per_req``
    #   sp_per_req       : alias of ``prefix_lengths`` cast to int64 on
    #                      device — what the existing SWA pool helpers
    #                      consume
    #   position_ids     : [T_total] int64 — per-token global absolute
    #                      position (RoPE input)
    #   req_id_per_token : [T_total] int32 — request id each token
    #                      belongs to (== ``searchsorted(cu_seqlens, t,
    #                      right=True) - 1``); shared by indexer.prepare
    #                      / compressor.prepare_metadata as ``b_idx`` and
    #                      by the workspace path's per-request slot offsets
    #   max_seqlen_q     : max(input_lengths) — workspace sizing hint
    # ``use_varlen`` is the single varlen / legacy switch for every prefill
    # downstream consumer. ``True`` ⇒ the per-request tensors below are
    # populated by the upper-layer broadcast and may be used directly (no
    # ``is not None`` checks needed). ``False`` ⇒ ``DSV4_VARLEN_PREFILL=0``
    # bisect channel — the legacy ``sp_int`` / ``seqlen`` scalars are the
    # only valid source. Set once in ``_build_shared_prefill_meta`` after
    # the single varlen-tensor presence assert.
    use_varlen: bool = False
    sp_per_req: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    batch_size: int = 1
    input_lengths: Optional[torch.Tensor] = None
    prefix_lengths: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    req_id_per_token: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    swa_meta: Optional[SwaPrefillMeta] = None
    csa_meta: Optional[CsaPrefillMeta] = None
    hca_meta: Optional[HcaPrefillMeta] = None


class PrefillQKV(NamedTuple):
    """Q/KV intermediate produced by ``_prefill_compute_qkv``.

    ``qr`` is fed to the indexer (CSA layers); ``q`` is the dense Q.
    ``kv_full`` is the all-gathered KV under CP; equals ``kv`` otherwise.
    The CP-aware sequence length lives on ``PrefillMeta.seqlen_full``.
    """

    qr: torch.Tensor
    q: torch.Tensor
    kv_full: torch.Tensor


class AttentionFP8(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        q_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        o_lora_rank: int,
        o_groups: int,
        window_size: int,
        compress_ratio: int,
        compress_rope_theta: float,
        rope_theta: float,
        rope_factor: float,
        beta_fast: int,
        beta_slow: int,
        original_seq_len: int,
        max_batch_size: int,
        max_seq_len: int,
        # Indexer
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        norm_eps: float = 1e-6,
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        """``layer_weights`` is the framework's per-layer dict
        (``ModelWeights.weights[layer_id]``) keyed by ``W.v4_*`` enum.
        Reads ``W.v4_attn_*`` for dense attention weights, ``W.v4_compressor_*``
        for the outer compressor, ``W.v4_indexer_*`` (forwarded) for the
        indexer."""
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.eps = norm_eps
        self.softmax_scale = head_dim**-0.5
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        # Per-rank head + group counts (S7a). Sharding only kicks in when
        # tp_size > 1; tp_size==1 keeps everything bit-exact unchanged.
        assert (
            n_heads % tp_size == 0
        ), f"n_heads={n_heads} not divisible by tp_size={tp_size}"
        assert (
            o_groups % tp_size == 0
        ), f"o_groups={o_groups} not divisible by tp_size={tp_size}"
        self.n_heads = n_heads // tp_size
        self.n_groups = o_groups // tp_size

        # Slices used to carve TP-local tensors out of the full ckpt.
        n_heads_local = self.n_heads
        n_groups_local = self.n_groups
        wq_b_row_slice = slice(
            tp_rank * n_heads_local * head_dim, (tp_rank + 1) * n_heads_local * head_dim
        )
        wo_a_row_slice = slice(
            tp_rank * n_groups_local * o_lora_rank,
            (tp_rank + 1) * n_groups_local * o_lora_rank,
        )
        wo_b_col_slice = slice(
            tp_rank * n_groups_local * o_lora_rank,
            (tp_rank + 1) * n_groups_local * o_lora_rank,
        )
        attn_sink_slice = slice(tp_rank * n_heads_local, (tp_rank + 1) * n_heads_local)

        from rtp_llm.utils.model_weight import W

        # Q / KV / O — FP8 linears go through LinearFactory →
        # CudaFp8DeepGEMMLinear → DeepGEMM fp8_gemm_nt.
        def _fp8_w_s(w_tag, s_tag, row_slice=None, col_slice=None):
            """Pull (weight, scale) by W tag with optional TP slicing,
            then build a CudaFp8DeepGEMMLinear via ``_v4_fp8_linear``.

            Scale slicing depends on layout:
              * legacy raw UE8M0 [N//128, K//128]: row/col strides are
                block-128 → ``slice.start // 128``.
              * framework packed int32 [N, K//128//4]: N is fully
                expanded (slice by full N stride) and K is packed 4×
                (slice by ``slice.start // 512``)."""
            w = layer_weights[w_tag]
            s = layer_weights[s_tag]
            scale_is_packed_int32 = s.dtype == torch.int32
            if row_slice is not None:
                w = w[row_slice]
                if scale_is_packed_int32:
                    s = s[row_slice]
                else:
                    s = s[row_slice.start // 128 : row_slice.stop // 128]
            if col_slice is not None:
                w = w[:, col_slice]
                if scale_is_packed_int32:
                    assert col_slice.start % 512 == 0 and col_slice.stop % 512 == 0, (
                        f"col_slice {col_slice} not aligned to 512 for "
                        f"packed int32 scale; framework path requires "
                        f"K-slices on 512-byte boundaries"
                    )
                    s = s[:, col_slice.start // 512 : col_slice.stop // 512]
                else:
                    s = s[:, col_slice.start // 128 : col_slice.stop // 128]
            if row_slice is not None or col_slice is not None:
                w = w.contiguous()
                s = s.contiguous()
            return _v4_fp8_linear(w, s)

        self.wq_a = _fp8_w_s(W.v4_attn_wq_a_w, W.v4_attn_wq_a_s)
        # wq_b is row-split along N (n_heads * head_dim)
        self.wq_b = _fp8_w_s(
            W.v4_attn_wq_b_w,
            W.v4_attn_wq_b_s,
            row_slice=wq_b_row_slice if tp_size > 1 else None,
        )
        # MQA single KV head — replicate.
        self.wkv = _fp8_w_s(W.v4_attn_wkv_w, W.v4_attn_wkv_s)

        # wo_a grouped projection: row-split along (n_groups*o_lora_rank).
        # Stored as plain ``[N, K]`` fp8 weight + UE8M0 scale tensors;
        # the ``fp8_einsum`` production path uses the pre-stacked
        # ``_wo_a_stk_w`` / ``_wo_a_stk_s`` buffers below, the BF16
        # fallback path inline-dequants from these via
        # ``_fp8_dequant_to_fp32``.
        assert (n_heads * head_dim) % o_groups == 0
        wo_a_w = layer_weights[W.v4_attn_wo_a_w]
        wo_a_s = layer_weights[W.v4_attn_wo_a_s]
        if tp_size > 1:
            wo_a_w = wo_a_w[wo_a_row_slice].contiguous()
            if wo_a_s.dtype == torch.int32:
                # framework path: scale is already (N, K//128//4) int32
                wo_a_s = wo_a_s[wo_a_row_slice].contiguous()
            else:
                wo_a_s = wo_a_s[
                    wo_a_row_slice.start // 128 : wo_a_row_slice.stop // 128
                ].contiguous()
        self.wo_a_w = wo_a_w
        self.wo_a_s = wo_a_s
        K_local = n_heads_local * head_dim // n_groups_local
        _stk_w, _stk_s = _prepare_wo_a_stacked(
            wo_a_w, wo_a_s, n_groups_local, o_lora_rank, K_local
        )
        self.register_buffer("_wo_a_stk_w", _stk_w, persistent=False)
        self.register_buffer("_wo_a_stk_s", _stk_s, persistent=False)

        # wo_b row-split along K (cols), all_reduce after forward
        self.wo_b = _fp8_w_s(
            W.v4_attn_wo_b_w,
            W.v4_attn_wo_b_s,
            col_slice=wo_b_col_slice if tp_size > 1 else None,
        )

        # Non-quantized norm weights — plain BF16 tensors (loader cast
        # via compute_dtype).  BF16 dtype is required by
        # ``rtp_llm_ops.rmsnorm`` (silent NaN with fp32).  attn_sink loads
        # as fp32 via descriptor data_type.
        self.q_norm = layer_weights[W.v4_attn_q_norm]
        self.kv_norm = layer_weights[W.v4_attn_kv_norm]
        attn_sink_full = layer_weights[W.v4_attn_sink]
        self.attn_sink = (
            attn_sink_full[attn_sink_slice].contiguous()
            if tp_size > 1
            else attn_sink_full
        )

        assert (n_heads * head_dim) % o_groups == 0

        # Compressor + Indexer (only for compressed layers)
        if compress_ratio:
            outer_cmp_weights = {
                "ape": layer_weights[W.v4_compressor_ape],
                "wkv": layer_weights[W.v4_compressor_wkv],
                "wgate": layer_weights[W.v4_compressor_wgate],
                "norm": layer_weights[W.v4_compressor_norm],
            }
            from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8

            self.compressor = CompressorFP8(
                dim=dim,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                compress_ratio=compress_ratio,
                max_batch_size=max_batch_size,
                norm_eps=norm_eps,
                compressor_weights=outer_cmp_weights,
            )

            # Phase E5: Compressor.kv_cache is self-managed (was an alias
            # into ``Attention.kv_cache[:, win:]``).  Configure shape here
            # because the Compressor doesn't know ``max_seq_len``.
            self.compressor.configure_kv_cache_shape(max_seq_len // compress_ratio)
            # #50 standalone / warmup fallback: nested indexer compressor
            # shares the INDEXER_KV pool with Indexer; when pool context is
            # absent (warmup, unit tests), ``_bind_kv_cache_from_pool``
            # needs the T hint to allocate the ephemeral zero tensor.
            if compress_ratio == 4:
                from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8

                self.indexer = IndexerFP8(
                    dim=dim,
                    q_lora_rank=q_lora_rank,
                    index_n_heads=index_n_heads,
                    index_head_dim=index_head_dim,
                    rope_head_dim=rope_head_dim,
                    index_topk=index_topk,
                    compress_ratio=compress_ratio,
                    max_batch_size=max_batch_size,
                    max_seq_len=max_seq_len,
                    norm_eps=norm_eps,
                    layer_weights=layer_weights,
                )

                # Configure nested indexer compressor shape hint so warmup
                # (where pool context is absent) can still allocate an
                # ephemeral zero kv_cache for the write at the tail of
                # ``_forward_scalar_impl``.
                if self.indexer.compressor is not None:
                    self.indexer.compressor.configure_kv_cache_shape(
                        max_seq_len // compress_ratio
                    )
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        # Phase E5b: kv_cache register_buffer retired.  KV storage now lives
        # exclusively in the framework BlockPools (SWA_KV + CSA_KV / HCA_KV);
        # prefill reads via ``_gather_kv_cache_dense_from_pool``, prefill
        # writes via ``_prefill_write_swa_to_pool`` + ``_prefill_paged_write_compressed``,
        # decode reads/writes via ``forward_decode``'s paged paths.
        kv_cache_size = window_size + (
            max_seq_len // compress_ratio if compress_ratio else 0
        )

        # Per-layer freqs_cis: SWA-only uses base rope_theta with no yarn,
        # CSA/HCA uses compress_rope_theta with yarn (when original_seq_len > 0).
        # Store scalars so we can re-compute after `to_empty`(meta) — otherwise
        # the buffer ends up all zeros.
        if compress_ratio:
            self._rope_base = compress_rope_theta
            self._rope_o_seq_len = original_seq_len
        else:
            self._rope_base = rope_theta
            self._rope_o_seq_len = 0
        self._rope_factor = rope_factor
        self._rope_beta_fast = beta_fast
        self._rope_beta_slow = beta_slow
        self._rope_dim = rope_head_dim
        self._rope_max_seq_len = max_seq_len
        freqs_cis = precompute_freqs_cis(
            rope_head_dim,
            max_seq_len,
            self._rope_o_seq_len,
            self._rope_base,
            rope_factor,
            beta_fast,
            beta_slow,
        )
        # Phase G: plain attr (not register_buffer).  `reset_rope_cache(device)`
        # recomputes + moves to the real device after meta-to-device
        # materialization — that's the authoritative placement path; no
        # automatic `.to(device)` semantics needed.
        self.freqs_cis = freqs_cis

        # Iter2: persistent FP8 sparse decode op. Carries the FlashMLA
        # ``sched_meta`` cache (per single/dual-pool flag) so the planner
        # only runs once per layer-type per process — vs the iter1' default
        # of one planner setup per layer per step. Built lazily on the
        # first FP8 decode call so BF16-KV runs don't pay the import cost.
        self._fp8_decode_op: Optional[Any] = None

        # CP context bound per-forward by V4Transformer.  None = no CP.
        self._cp_ctx: Optional[CPContext] = None

        # Shared prefill meta (compress_ratio bucket; not layer-specific).
        # V4Transformer.forward_layers builds one per ratio in use, sets it
        # on every layer's attention via ``_set_prefill_meta_shared``, and
        # clears at end-of-forward. None ⇒ standalone path / decode →
        # ``_prefill_common_setup`` falls back to per-call build.
        self._prefill_meta_shared: Optional["PrefillMeta"] = None

        # qwen3-style: framework KVCache handle + per-request block tables
        # flow in as kwargs on ``forward`` / ``forward_decode``; these two
        # attrs are stashed only for the duration of a single forward call
        # (try/finally clears them), so pool views resolve via
        # ``self._kv_cache.get_layer_cache(layer_id, attn_type).kv_cache_base``
        # inside helpers without threading the handle through every method.
        self._kv_cache: Optional[Any] = None
        self._block_tables_by_type: Optional[Dict[int, torch.Tensor]] = None

        # Precomputed pool specs per attn_type — (vec_dtype, vec_dim) used
        # to reinterpret the raw ``[num_blocks, stride_elems]`` framework
        # pool tensor as a typed ``[total_slots, vec_dim]`` flat view.
        from rtp_llm.models_py.modules.dsv4.attn_type import (
            CSA_KV,
            CSA_STATE,
            HCA_KV,
            HCA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
            SWA_KV,
        )

        idx_hd = index_head_dim
        coff_csa = 2  # CSA overlap=True
        coff_idx = 2  # Indexer's nested Compressor overlap=True
        # HCA uses coff=1 (overlap=False) so HCA_STATE vec_dim = 2*head_dim.
        kv_spec = (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES)
        indexer_kv_spec = (torch.uint8, _DSV4_FP8_INDEXER_ENTRY_BYTES)
        self._pool_spec: Dict[int, tuple] = {
            SWA_KV: kv_spec,
            CSA_KV: kv_spec,
            HCA_KV: kv_spec,
            INDEXER_KV: indexer_kv_spec,
            CSA_STATE: (torch.float32, 2 * coff_csa * head_dim),
            HCA_STATE: (torch.float32, 2 * head_dim),
            INDEXER_STATE: (torch.float32, 2 * coff_idx * idx_hd),
        }

    def set_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        """Bind CP context.  When active on a prefill call, ``forward``
        does rank-local Q × FULL-KV attention: RoPE uses global
        positions; the rank-local KV is all-gathered + padding-stripped
        so every rank sees the same full sliding-window KV; the
        sliding-window + compressed topk indices are computed relative
        to that full-KV layout; sparse_attn runs on rank-local Q rows
        only so the output is ``[B, chunk_length, H, D]`` — the frame-
        work then all-gathers across ranks and strips padding."""
        self._cp_ctx = cp_ctx

    def _pool_view(self, attn_type: int) -> Optional[torch.Tensor]:
        """Return a flat ``[total_slots, vec_dim]`` typed view of the
        framework BlockPool for this layer + attn_type, or ``None`` if
        the pool isn't allocated (e.g. SWA-only layer has no CSA/HCA
        pool).  Delegates to ``KVCache.get_layer_cache(layer_id,
        attn_type)`` — no Python-side descriptor cache."""
        if self._kv_cache is None:
            return None
        spec = self._pool_spec.get(attn_type)
        if spec is None:
            return None
        attn_type_enum = _ATTN_TYPE_ENUM_BY_INT.get(attn_type)
        if attn_type_enum is None:
            return None
        # Polymorphic probe: build_paged_pool_specs sweeps every attn_type
        # across every layer.  C++ raises "Layer X does not own attention
        # type Y" for layers that don't own this region — catching it tells
        # the caller to skip.  Not defensive bloat.
        try:
            layer_kv = self._kv_cache.get_layer_cache(self.layer_id, attn_type_enum)
        except RuntimeError:
            return None
        base = layer_kv.kv_cache_base
        if base is None or base.numel() == 0 or base.dim() != 2:
            return None
        vec_dtype, vec_dim = spec
        # base: [num_blocks, stride_elems].  stride_bytes = stride_elems *
        # element_size.  entries_per_block = stride_bytes // bytes_per_entry.
        stride_bytes = int(base.shape[1]) * int(base.element_size())
        bytes_per_entry = vec_dim * vec_dtype.itemsize
        if bytes_per_entry <= 0 or stride_bytes < bytes_per_entry:
            return None
        eb = stride_bytes // bytes_per_entry
        useful_bytes = eb * bytes_per_entry
        # Reinterpret as uint8 [num_blocks, stride_bytes] so we can slice
        # exact useful-byte span, then cast to vec_dtype + flatten to
        # [total_slots, vec_dim].
        raw_u8 = base.view(torch.uint8)
        if raw_u8.shape[1] < useful_bytes:
            return None
        # FP8 pools may have C++-side TMA padding (stride_bytes >
        # useful_bytes), making the slice non-viewable. Callers that
        # need this layout must use ``_pool_view_3d_fp8`` instead.
        if vec_dtype == torch.uint8 and stride_bytes > useful_bytes:
            return None
        return raw_u8[:, :useful_bytes].view(vec_dtype).view(-1, vec_dim)

    def _pool_view_3d_fp8(self, attn_type: int) -> Optional[torch.Tensor]:
        """Return ``[num_blocks, eb, ENTRY_BYTES]`` uint8 view of an FP8 KV
        pool, respecting C++-side TMA padding (per-block stride may exceed
        ``eb * ENTRY_BYTES``). The flat 2D form ``_pool_view`` returns is
        invalid here because the slice it produces is non-contiguous and
        can't be ``.view()``'d through the dtype/shape chain.
        """
        if self._kv_cache is None:
            return None
        spec = self._pool_spec.get(attn_type)
        if spec is None:
            return None
        attn_type_enum = _ATTN_TYPE_ENUM_BY_INT.get(attn_type)
        if attn_type_enum is None:
            return None
        try:
            layer_kv = self._kv_cache.get_layer_cache(self.layer_id, attn_type_enum)
        except RuntimeError:
            return None
        base = layer_kv.kv_cache_base
        if base is None or base.numel() == 0 or base.dim() != 2:
            return None
        vec_dtype, vec_dim = spec
        if vec_dtype != torch.uint8:
            return None
        stride_bytes = int(base.shape[1]) * int(base.element_size())
        bytes_per_entry = vec_dim
        if bytes_per_entry <= 0 or stride_bytes < bytes_per_entry:
            return None
        eb = stride_bytes // bytes_per_entry
        raw_u8 = base.view(torch.uint8)
        num_blocks = int(raw_u8.shape[0])
        # as_strided: dim-0 stride = stride_bytes (jump over per-block
        # padding); dim-1 stride = bytes_per_entry; dim-2 stride = 1.
        return raw_u8.as_strided(
            (num_blocks, eb, bytes_per_entry),
            (stride_bytes, bytes_per_entry, 1),
        )

    def _pool_entries_per_block(self, attn_type: int) -> int:
        """Derive ``entries_per_block`` from the framework pool tensor for
        this layer + attn_type.  Returns 0 if pool unavailable."""
        if self._kv_cache is None:
            return 0
        spec = self._pool_spec.get(attn_type)
        if spec is None:
            return 0
        attn_type_enum = _ATTN_TYPE_ENUM_BY_INT.get(attn_type)
        if attn_type_enum is None:
            return 0
        # Polymorphic probe — see _pool_view for rationale.
        try:
            layer_kv = self._kv_cache.get_layer_cache(self.layer_id, attn_type_enum)
        except RuntimeError:
            return 0
        base = layer_kv.kv_cache_base
        if base is None or base.numel() == 0 or base.dim() != 2:
            return 0
        vec_dtype, vec_dim = spec
        stride_bytes = int(base.shape[1]) * int(base.element_size())
        bytes_per_entry = vec_dim * vec_dtype.itemsize
        if bytes_per_entry <= 0:
            return 0
        return stride_bytes // bytes_per_entry

    def _prefill_paged_write_kv(
        self,
        attn_type: int,
        source_buf: torch.Tensor,  # [bsz, T, vec_dim]
        bsz: int,
    ) -> None:
        """Phase F generic dual-write: mirror ``source_buf[:bsz, :T]`` into
        the framework BlockPool of ``attn_type``. No-op when no KVCache
        handle / block table bound, or the pool isn't allocated for this
        layer. Sentinel block_id ≤ 0 entries are skipped via
        ``mask_negative=True``.

        Supports ``bsz >= 1``.  ``self._block_tables_by_type[attn_type]``
        must carry at least ``bsz`` rows; each row's block_id list addresses
        that request's own pool slots.  slot_mapping is built as ``[B, T]``
        via double-axis ``bt[b_idx, block_in_seq]`` so per-row block
        assignment is respected.  bsz==1 produces byte-equal slot_mapping
        to the historical scalar-row implementation."""
        if self._kv_cache is None or self._block_tables_by_type is None:
            return
        from rtp_llm.models_py.modules.dsv4.fp8.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        bt = self._block_tables_by_type.get(attn_type)
        if bt is None or bt.numel() == 0:
            return
        pool_view = self._pool_view(attn_type)
        eb = self._pool_entries_per_block(attn_type)
        if pool_view is None or eb <= 0:
            return
        T = int(source_buf.shape[1])
        D = int(source_buf.shape[2])
        if T == 0:
            return
        device = source_buf.device
        max_blocks = bt.shape[1]
        # Pool capacity = max_blocks * eb. When source_buf has more rows than
        # pool capacity (e.g. HCA compressor.kv_state has coff*ratio=128 rows
        # but HCA_STATE pool only provisions 2 blocks × 8 eb = 16 rows), the
        # deleted ``_scatter_state_pool`` stopped at ``max_blks * eb``; we
        # mirror that here by sentinel-masking positions past pool capacity
        # so write_kv_to_pool(mask_negative=True) skips them instead of
        # clamp-collapsing them into the last block (which silently
        # overwrites valid rows on every excess position).
        pool_capacity = max_blocks * eb
        pos = torch.arange(T, device=device, dtype=torch.long)  # [T]
        in_capacity_row = pos < pool_capacity  # [T]
        safe_pos = torch.where(in_capacity_row, pos, torch.zeros_like(pos))
        block_in_seq = safe_pos // eb  # [T]
        in_block = safe_pos % eb  # [T]
        bt_long = bt.to(torch.long)
        # Double-axis gather: block_id[b, t] = bt_long[b, block_in_seq[t]].
        b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)  # [B,1]
        block_id = bt_long[:bsz][b_idx, block_in_seq.unsqueeze(0)]  # [B, T]
        in_capacity = in_capacity_row.unsqueeze(0).expand(bsz, -1)  # [B, T]
        # Mirror ``_scatter_kv_pool``'s ``if bid <= 0: continue`` sentinel:
        # unallocated blocks (bid <= 0) and over-capacity rows both → -1.
        valid = (block_id > 0) & in_capacity
        slot_per = torch.where(
            valid,
            block_id * eb + in_block.unsqueeze(0),
            torch.full_like(block_id, -1),
        )  # [B, T]
        slot_mapping = slot_per.reshape(-1)  # [B*T]
        buf_flat = source_buf[:bsz].reshape(bsz * T, D)
        write_kv_to_pool(buf_flat, slot_mapping, pool_view, mask_negative=True)

    def _prefill_paged_write_kv_range(
        self,
        attn_type: int,
        source_buf: torch.Tensor,  # [bsz, T, vec_dim]
        bsz: int,
        write_start: int,
    ) -> None:
        if self._kv_cache is None or self._block_tables_by_type is None:
            return
        from rtp_llm.models_py.modules.dsv4.fp8.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        bt = self._block_tables_by_type.get(attn_type)
        pool_view = self._pool_view(attn_type)
        eb = self._pool_entries_per_block(attn_type)
        if bt is None or bt.numel() == 0 or pool_view is None or eb <= 0:
            return
        T = int(source_buf.shape[1])
        D = int(source_buf.shape[2])
        if T <= 0:
            return
        device = source_buf.device
        pos = torch.arange(
            write_start, write_start + T, device=device, dtype=torch.long
        )
        block_in_seq = pos // eb
        in_block = pos % eb
        in_capacity = block_in_seq < int(bt.shape[1])
        safe_block = torch.where(
            in_capacity, block_in_seq, torch.zeros_like(block_in_seq)
        )
        bt_long = bt[:bsz].to(device=device, dtype=torch.long)
        b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
        block_id = bt_long[b_idx, safe_block.unsqueeze(0).expand(bsz, -1)]
        valid = in_capacity.unsqueeze(0) & (block_id > 0)
        slot_per = torch.where(
            valid,
            block_id * eb + in_block.unsqueeze(0),
            torch.full_like(block_id, -1),
        )
        write_kv_to_pool(
            source_buf[:bsz].reshape(bsz * T, D),
            slot_per.reshape(-1),
            pool_view,
            mask_negative=True,
        )

    def _prefill_write_swa_to_pool(
        self,
        bsz: int,
        kv_full: torch.Tensor,
        sp: Union[int, torch.Tensor],
        row_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Phase E5b: direct SWA write to the framework SWA_KV pool
        (replaces the retired ``self.kv_cache[:bsz, :win]`` register
        _buffer ring-write + ``_prefill_paged_write_swa`` mirror pair).

        For each of a row's valid new tokens at local index i:
            abs_pos[b] = sp[b] + i
            slot[b]    = bt[b, abs_pos[b] // eb] * eb + (abs_pos[b] % eb)

        The framework SWA_KV pool is addressed by absolute token position,
        not by ``pos % window_size``.  Decode translates its sliding-window
        absolute positions through the same block table; continuation prefill
        does the same when reconstructing a dense linear KV view.

        Supports ``bsz >= 1``.  ``sp`` accepts scalar (legacy) or ``[B]``
        int64 tensor.  ``row_seqlens`` (optional ``[B]``) marks the valid
        new-token count per row; ``None`` means ``kv_full.shape[1]`` for
        every row.  Per-row rows with ``seq_t[b] < max_n_write`` emit
        sentinel ``-1`` slots past their valid range so write_kv_to_pool
        skips them.
        """
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        if self._kv_cache is None or self._block_tables_by_type is None:
            return
        bt = self._block_tables_by_type.get(SWA_KV)
        if bt is None or bt.numel() == 0:
            return
        eb = self._pool_entries_per_block(SWA_KV)
        if eb <= 0:
            return
        # FP8 SWA pool uses TMA per-block padding which makes the 2D
        # ``_pool_view`` slice non-viewable (RuntimeError). Skip the 2D
        # view for FP8; the FP8 write branch below uses a 3D as_strided
        # view directly. BF16 path keeps the existing 2D view contract.
        pool_view = None

        max_seqlen = int(kv_full.shape[1])
        if max_seqlen == 0:
            return
        device = kv_full.device
        max_blocks = bt.shape[1]

        # Normalize sp -> [B] int64 tensor.
        if isinstance(sp, torch.Tensor):
            sp_t = sp.to(device=device, dtype=torch.long)
            if sp_t.dim() == 0:
                sp_t = sp_t.unsqueeze(0)
            if sp_t.numel() == 1 and bsz > 1:
                sp_t = sp_t.expand(bsz)
        else:
            sp_t = torch.full((bsz,), int(sp), device=device, dtype=torch.long)

        # Per-row seqlens (valid new-token count).  None => max_seqlen for all.
        if row_seqlens is None:
            seq_t = torch.full((bsz,), max_seqlen, device=device, dtype=torch.long)
        else:
            seq_t = row_seqlens.to(device=device, dtype=torch.long)
            if seq_t.dim() == 0:
                seq_t = seq_t.unsqueeze(0)
            if seq_t.numel() == 1 and bsz > 1:
                seq_t = seq_t.expand(bsz)

        # Write every token from this prefill that maps to a currently
        # allocated SWA block-table entry.  Attention itself only consumes a
        # sliding window, but prefix-cache reuse may stop at an earlier block
        # boundary than this request's final token.  Those boundary states need
        # the full request tail physical blocks populated, not just the final
        # ``window_size`` rows.
        n_write_per_row = seq_t  # [B]
        max_n_write = int(n_write_per_row.max().item()) if bsz > 0 else 0
        if max_n_write == 0:
            return

        j = torch.arange(max_n_write, device=device, dtype=torch.long)  # [max_n_write]
        # Row-local validity: j < n_write[b] -> valid position.
        row_valid = j.unsqueeze(0) < n_write_per_row.unsqueeze(1)  # [B, max_n_write]
        # Global positions per (row, j): sp[b] + j.
        global_pos = sp_t.unsqueeze(1) + j.unsqueeze(0)  # [B, max_n_write]
        block_in_seq = global_pos // eb
        in_block = global_pos % eb
        bt_long = bt.to(torch.long)
        in_capacity = block_in_seq < max_blocks
        safe_in_seq = torch.where(
            in_capacity, block_in_seq, torch.zeros_like(block_in_seq)
        )
        b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)  # [B,1]
        block_id = bt_long[:bsz][b_idx, safe_in_seq]  # [B, max_n_write]
        valid = (block_id > 0) & in_capacity & row_valid
        slot = torch.where(
            valid, block_id * eb + in_block, torch.full_like(in_block, -1)
        )

        # Gather source rows: src[b, j] = kv_full[b, j].
        src_pos = j.unsqueeze(0).expand(bsz, -1)  # [B, max_n_write]
        # Clamp to [0, max_seqlen-1] for safe gather; invalid positions are
        # masked by slot == -1 so the gathered value doesn't matter.
        src_pos_safe = src_pos.clamp(min=0, max=max(max_seqlen - 1, 0))
        gather_idx = src_pos_safe.unsqueeze(-1).expand(-1, -1, self.head_dim)
        source = torch.gather(kv_full[:bsz], 1, gather_idx)  # [B, max_n_write, D]
        source_flat = source.reshape(bsz * max_n_write, self.head_dim)
        slot_mapping = slot.reshape(-1)
        # FP8 SWA pool: 584B per slot (fp8 NoPE 448 + bf16 RoPE 128 +
        # ue8m0 scale 8). 3D view honoring TMA per-block padding.
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )

        pool_3d = self._pool_view_3d_fp8(SWA_KV)
        assert (
            pool_3d is not None
        ), "FP8 SWA pool view unavailable in _prefill_write_swa_to_pool"
        quantize_and_insert_k_cache(
            source_flat.to(torch.bfloat16), pool_3d, slot_mapping
        )

    def _prefill_read_swa_from_pool(
        self,
        bsz: int,
        sp: Union[int, torch.Tensor],
        row_seqlens: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Reconstruct the dense SWA ring ``[B, window_size, head_dim]`` from
        sparse absolute-token block tables.

        ``topk_idxs`` indexes SWA by ring slot (``global_pos % window_size``),
        while allocator block tables retain only the last absolute token
        blocks.  For each ring slot we locate the latest global position that
        maps to it and gather that absolute pool slot.
        """
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV

        if self._kv_cache is None or self._block_tables_by_type is None:
            return None
        bt = self._block_tables_by_type.get(SWA_KV)
        if bt is None or bt.numel() == 0:
            return None
        pool_view = self._pool_view(SWA_KV)
        eb = self._pool_entries_per_block(SWA_KV)
        if pool_view is None or eb <= 0:
            return None

        win = self.window_size
        if win <= 0:
            return None
        device = pool_view.device
        dtype = torch.bfloat16
        max_blocks = bt.shape[1]

        if isinstance(sp, torch.Tensor):
            sp_t = sp.to(device=device, dtype=torch.long)
            if sp_t.dim() == 0:
                sp_t = sp_t.unsqueeze(0)
            if sp_t.numel() == 1 and bsz > 1:
                sp_t = sp_t.expand(bsz)
        else:
            sp_t = torch.full((bsz,), int(sp), device=device, dtype=torch.long)

        if row_seqlens is None:
            seq_t = torch.full((bsz,), win, device=device, dtype=torch.long)
        else:
            seq_t = row_seqlens.to(device=device, dtype=torch.long)
            if seq_t.dim() == 0:
                seq_t = seq_t.unsqueeze(0)
            if seq_t.numel() == 1 and bsz > 1:
                seq_t = seq_t.expand(bsz)

        last_global = sp_t + seq_t - 1  # [B]
        window_start = torch.clamp(last_global - win + 1, min=0)
        ring_pos = torch.arange(win, device=device, dtype=torch.long)  # [W]
        delta = torch.remainder(last_global.unsqueeze(1) - ring_pos.unsqueeze(0), win)
        global_pos = last_global.unsqueeze(1) - delta  # [B, W]
        valid_pos = (seq_t.unsqueeze(1) > 0) & (global_pos >= window_start.unsqueeze(1))

        block_in_seq = global_pos // eb
        in_block = global_pos % eb
        in_capacity = (block_in_seq >= 0) & (block_in_seq < max_blocks)
        safe_block = torch.where(
            in_capacity, block_in_seq, torch.zeros_like(block_in_seq)
        )
        bt_long = bt.to(torch.long)
        b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
        block_id = bt_long[:bsz][b_idx, safe_block]
        valid = valid_pos & in_capacity & (block_id > 0)
        safe_slot = torch.where(
            valid, block_id * eb + in_block, torch.zeros_like(block_id)
        )

        gathered = pool_view.index_select(0, safe_slot.reshape(-1))
        if gathered.dtype != dtype:
            gathered = gathered.to(dtype)
        zero_row = torch.zeros((), dtype=dtype, device=device)
        out = torch.where(valid.reshape(-1).unsqueeze(-1), gathered, zero_row)
        return out.view(bsz, win, self.head_dim).contiguous()

    def _prefill_read_swa_dense_abs_from_pool(
        self,
        bsz: int,
        sp: Union[int, torch.Tensor],
        row_seqlens: torch.Tensor,
        dense_len: int,
        current_kv_full: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Build a dense absolute SWA view for continuation prefill.

        Continuation prefill attention needs the SWA window as it existed at
        each query position, not the final ring after this suffix has been
        written.  The pool contains the prefix tail at entry; overlay the
        current suffix KV from ``current_kv_full`` so absolute topk indices
        can read ``[prefix tail | current suffix]`` by token position.
        """
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV

        if self._kv_cache is None or self._block_tables_by_type is None:
            return None
        bt = self._block_tables_by_type.get(SWA_KV)
        if bt is None or bt.numel() == 0:
            return None
        pool_view = self._pool_view(SWA_KV)
        eb = self._pool_entries_per_block(SWA_KV)
        if pool_view is None or eb <= 0 or dense_len <= 0:
            return None

        device = pool_view.device
        dtype = torch.bfloat16
        if isinstance(sp, torch.Tensor):
            sp_t = sp.to(device=device, dtype=torch.long)
            if sp_t.dim() == 0:
                sp_t = sp_t.unsqueeze(0)
            if sp_t.numel() == 1 and bsz > 1:
                sp_t = sp_t.expand(bsz)
        else:
            sp_t = torch.full((bsz,), int(sp), device=device, dtype=torch.long)
        seq_t = row_seqlens.to(device=device, dtype=torch.long)
        if seq_t.dim() == 0:
            seq_t = seq_t.unsqueeze(0)
        if seq_t.numel() == 1 and bsz > 1:
            seq_t = seq_t.expand(bsz)

        pos = torch.arange(dense_len, device=device, dtype=torch.long)
        block_in_seq = pos // eb
        in_block = pos % eb
        max_blocks = bt.shape[1]
        in_capacity_row = block_in_seq < max_blocks
        safe_block = torch.where(
            in_capacity_row, block_in_seq, torch.zeros_like(block_in_seq)
        )
        bt_long = bt[:bsz].to(device=device, dtype=torch.long)
        b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
        block_id = bt_long[b_idx, safe_block.unsqueeze(0).expand(bsz, -1)]
        valid = in_capacity_row.unsqueeze(0) & (block_id > 0)

        safe_slot = torch.where(
            valid, block_id * eb + in_block.unsqueeze(0), torch.zeros_like(block_id)
        )

        gathered = pool_view.index_select(0, safe_slot.reshape(-1))
        if gathered.dtype != dtype:
            gathered = gathered.to(dtype)
        zero_row = torch.zeros((), dtype=dtype, device=device)
        out = torch.where(valid.reshape(-1).unsqueeze(-1), gathered, zero_row)
        out = out.view(bsz, dense_len, self.head_dim).contiguous()
        for b in range(bsz):
            sp_b = int(sp_t[b].item())
            seq_b = int(seq_t[b].item())
            if current_kv_full is not None and seq_b > 0 and sp_b < dense_len:
                dst_end = min(sp_b + seq_b, dense_len)
                copy_len = dst_end - sp_b
                if copy_len > 0:
                    src = current_kv_full[b, :copy_len]
                    if src.dtype != dtype:
                        src = src.to(dtype)
                    out[b, sp_b:dst_end] = src
        return out.contiguous()

    def _set_compressor_pool_context(self) -> None:
        """#50: resolve CSA/HCA + INDEXER pool views + per-request block
        tables from ``self._kv_cache`` + ``self._block_tables_by_type`` and
        hand them to Compressor / Indexer via ``set_pool_context``.  Called
        once at the top of every forward/forward_decode; paired with
        :meth:`_clear_compressor_pool_context` in a try/finally so stale
        pool views don't leak across forwards."""
        from rtp_llm.models_py.modules.dsv4.attn_type import (
            CSA_KV,
            CSA_STATE,
            HCA_KV,
            HCA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
        )

        bt_by_type = self._block_tables_by_type

        if self.compressor is not None:
            if self.compress_ratio == 4:
                kv_at, state_at = CSA_KV, CSA_STATE
            elif self.compress_ratio == 128:
                kv_at, state_at = HCA_KV, HCA_STATE
            else:
                kv_at, state_at = None, None
            # FP8 KV pool has TMA per-block padding which the flat 2D
            # ``_pool_view`` cannot represent; use the 3D as_strided form
            # instead. CompressorFP8 stores whatever shape it receives
            # in ``_kv_pool_view`` and dispatches via ``_pool_view_3d``.
            if kv_at is not None:
                kv_view = self._pool_view_3d_fp8(kv_at)
            else:
                kv_view = self._pool_view(kv_at) if kv_at is not None else None
            kv_bt = (
                bt_by_type.get(kv_at)
                if (bt_by_type is not None and kv_at is not None)
                else None
            )
            kv_eb = self._pool_entries_per_block(kv_at) if kv_at is not None else 0
            state_view = self._pool_view(state_at) if state_at is not None else None
            state_bt = (
                bt_by_type.get(state_at)
                if (bt_by_type is not None and state_at is not None)
                else None
            )
            state_eb = (
                self._pool_entries_per_block(state_at) if state_at is not None else 0
            )
            self.compressor.set_pool_context(
                kv_view, kv_bt, kv_eb, state_view, state_bt, state_eb
            )

        if self.indexer is not None:
            kv_view = self._pool_view_3d_fp8(INDEXER_KV)

            kv_bt = bt_by_type.get(INDEXER_KV) if bt_by_type is not None else None
            kv_eb = self._pool_entries_per_block(INDEXER_KV)
            state_view = self._pool_view(INDEXER_STATE)
            state_bt = bt_by_type.get(INDEXER_STATE) if bt_by_type is not None else None
            state_eb = self._pool_entries_per_block(INDEXER_STATE)
            self.indexer.set_pool_context(
                kv_view, kv_bt, kv_eb, state_view, state_bt, state_eb
            )

    def _clear_compressor_pool_context(self) -> None:
        if self.compressor is not None:
            self.compressor.clear_pool_context()
        if self.indexer is not None:
            self.indexer.clear_pool_context()

    def _prefill_paged_read_kv(
        self,
        attn_type: int,
        bsz: int,
        T: int,
        vec_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Phase E1 read-path counterpart to ``_prefill_paged_write_kv``.

        Gathers a ``[bsz, T, vec_dim]`` dense tensor from the framework
        BlockPool of ``attn_type`` using the same slot_mapping formula as
        the writer, so the write-then-read round trip is byte-equal on
        valid positions.  Sentinel positions (pos ≥ pool_capacity or
        unallocated block_id) are zero-filled.

        Supports ``bsz >= 1``.  Per-row block_id lookup uses
        ``bt[b_idx, block_in_seq]`` so each row reads from its own
        block allocation.  bsz==1 produces byte-equal output to the
        historical scalar-row implementation.

        Returns ``None`` when the ctx is unbound or the pool isn't
        registered for this layer, so callers can fall back.
        """
        if self._kv_cache is None or self._block_tables_by_type is None:
            return None
        bt = self._block_tables_by_type.get(attn_type)
        if bt is None or bt.numel() == 0 or T == 0:
            return None
        eb = self._pool_entries_per_block(attn_type)
        if eb <= 0:
            return None

        # FP8 KV pool: dequantize on read. The 584B per-slot layout
        # matches vLLM's fp8_ds_mla scheme so we route the [0, T) prefix
        # through ``dequantize_and_gather_k_cache`` (gather_lens=T,
        # seq_lens=T, offset=0 ⇒ writes out[:, 0:T, :] bf16). The
        # zero-row sentinel logic the BF16 path uses for unallocated
        # blocks is replaced by the kernel reading whatever physical
        # block_id maps to — caller must pass T ≤ what was written.
        pool_3d = self._pool_view_3d_fp8(attn_type)
        if pool_3d is None or pool_3d.shape[-1] != _DSV4_FP8_KV_ENTRY_BYTES:
            return None
        from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
            dequantize_and_gather_k_cache,
        )

        HD_DEQUANT = 512
        assert (
            dtype == torch.bfloat16
        ), f"FP8 pool read returns bf16; got requested dtype={dtype}"
        out = torch.zeros((bsz, T, HD_DEQUANT), dtype=torch.bfloat16, device=device)
        seq_lens = torch.full((bsz,), T, dtype=torch.int32, device=device)
        bt_for_kernel = bt[:bsz].to(torch.int32).contiguous()
        dequantize_and_gather_k_cache(
            out, pool_3d, seq_lens, None, bt_for_kernel, eb, 0
        )
        return out

        pool_view = self._pool_view(attn_type)
        if pool_view is None:
            return None
        max_blocks = bt.shape[1]
        pool_capacity = max_blocks * eb

        pos = torch.arange(T, device=device, dtype=torch.long)  # [T]
        in_capacity_row = pos < pool_capacity  # [T]
        safe_pos = torch.where(in_capacity_row, pos, torch.zeros_like(pos))
        block_in_seq = safe_pos // eb
        in_block = safe_pos % eb
        bt_long = bt.to(torch.long)
        b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)  # [B,1]
        block_id = bt_long[:bsz][b_idx, block_in_seq.unsqueeze(0)]  # [B, T]
        in_capacity = in_capacity_row.unsqueeze(0).expand(bsz, -1)  # [B, T]
        valid = (block_id > 0) & in_capacity
        safe_slot = torch.where(
            valid,
            block_id * eb + in_block.unsqueeze(0),
            torch.zeros_like(block_id),
        )  # [B, T]
        gathered = pool_view.index_select(0, safe_slot.reshape(-1))  # [B*T, vec_dim]
        # Pool storage dtype matches source_buf dtype by construction (see
        # _prefill_paged_write_kv which writes via write_kv_to_pool →
        # index_copy_).  BF16 KV pools, fp32 STATE pools.  Enforce dtype
        # + zero-fill sentinels in one where().
        if gathered.dtype != dtype:
            gathered = gathered.to(dtype)
        zero_row = torch.zeros((), dtype=dtype, device=device)
        out_flat = torch.where(
            valid.reshape(-1).unsqueeze(-1), gathered, zero_row
        )  # [B*T, vec_dim]
        return out_flat.view(bsz, T, vec_dim).contiguous()

    def _gather_kv_cache_dense_from_pool(
        self,
        bsz: int,
        sp: Optional[Union[int, torch.Tensor]] = None,
        row_seqlens: Optional[torch.Tensor] = None,
        swa_dense_len: Optional[int] = None,
        swa_dense_override: Optional[torch.Tensor] = None,
        swa_T: Optional[int] = None,
        cmp_T: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Phase E1: reconstruct the ``[bsz, kv_cache_size, head_dim]``
        dense tensor that ``self.kv_cache[:bsz]`` presents, but sourced
        from the framework pools instead of the register_buffer mirror.

        Layout (matches register_buffer):
          ``[:, :swa_T, :]``             -- SWA_KV absolute-position stream
          ``[:, swa_T:swa_T+cmp_T, :]``  -- CSA_KV or HCA_KV compressed stream

        Returns ``None`` when ctx not bound — caller falls back to
        register_buffer.  SWA-only layers (compress_ratio == 0) get a bare
        ``[bsz, swa_T, hd]`` read.  ``swa_T`` defaults to ``window_size`` for
        decode-like ring callers; continuation prefill passes the absolute
        sequence end so every query can use linear absolute topk indices.
        """
        from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, HCA_KV, SWA_KV

        if self._kv_cache is None or self._block_tables_by_type is None:
            return None
        win = self.window_size
        hd = self.head_dim
        dtype = torch.bfloat16
        device = self.freqs_cis.device
        T_swa = int(swa_T) if swa_T is not None else win
        T_cmp = (
            int(cmp_T)
            if cmp_T is not None
            else (
                self.compressor._kv_cache_t
                if (self.compressor is not None and self.compress_ratio)
                else 0
            )
        )

        if swa_dense_override is not None:
            swa_dense = swa_dense_override
        elif swa_dense_len is not None:
            assert sp is not None
            assert row_seqlens is not None
            swa_dense = self._prefill_read_swa_dense_abs_from_pool(
                bsz, sp, row_seqlens, int(swa_dense_len)
            )
        elif sp is not None:
            swa_dense = self._prefill_read_swa_from_pool(bsz, sp, row_seqlens)
        else:
            swa_dense = self._prefill_paged_read_kv(
                SWA_KV, bsz, T_swa, hd, dtype, device
            )
        if swa_dense is None:
            return None
        if T_cmp <= 0 or self.compress_ratio == 0:
            return swa_dense
        cmp_at = CSA_KV if self.compress_ratio == 4 else HCA_KV
        cmp_dense = self._prefill_paged_read_kv(cmp_at, bsz, T_cmp, hd, dtype, device)
        if cmp_dense is None:
            # Compressed pool not bound for this layer — shouldn't happen in
            # production but keep the safe fallback so caller can use
            # register_buffer instead of a half-built view.
            return None
        return torch.cat([swa_dense, cmp_dense], dim=1)

    def reset_rope_cache(self, device=None):
        """Recompute `freqs_cis` on the actual device — MUST be called after
        `model.to_empty(device=...)` since meta-tensor construction leaves the
        cached freqs as zeros."""
        freqs_cis = precompute_freqs_cis(
            self._rope_dim,
            self._rope_max_seq_len,
            self._rope_o_seq_len,
            self._rope_base,
            self._rope_factor,
            self._rope_beta_fast,
            self._rope_beta_slow,
        )
        if device is not None:
            freqs_cis = freqs_cis.to(device)
        self.freqs_cis = freqs_cis
        # Clear compressor / indexer bound references so they rebind on next forward
        if self.compressor is not None:
            self.compressor.freqs_cis = None
        if self.indexer is not None:
            self.indexer.freqs_cis = None
            if self.indexer.compressor is not None:
                self.indexer.compressor.freqs_cis = None

    def _get_fp8_decode_op(self):
        """Lazy-build the persistent ``SparseAttnV4DecodeFp8Op`` so its
        ``sched_meta`` cache survives across decode steps.

        Iter1' instantiated the op per call inside ``_forward_decode_body``
        which threw away the FlashMLA planner state on every call (60 layers
        × per step = 60 planner setups). Caching here cuts that to one setup
        per layer-type per process.
        """
        if self._fp8_decode_op is None:
            from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
                SparseAttnV4DecodeFp8Op,
            )

            self._fp8_decode_op = SparseAttnV4DecodeFp8Op(
                n_heads=self.n_heads,
                head_dim=self.head_dim,
                softmax_scale=self.softmax_scale,
            )
        return self._fp8_decode_op

    def _rmsnorm_weighted(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Framework C++ ``rtp_llm_ops.rmsnorm`` (single launch, bf16 weight).
        # Requires 2D input — reshape/restore keeps this a drop-in.
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        out = torch.empty_like(x_2d)
        rtp_llm_ops.rmsnorm(
            out, x_2d, weight, self.eps, torch.cuda.current_stream().cuda_stream
        )
        return out.view(orig_shape)

    def _lin(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Linear call that tolerates both the legacy QuantizedLinear
        (3-D input OK via F.linear) and factory LinearBase (expects 2-D)."""
        if x.dim() > 2:
            shape = x.shape
            y = layer(x.reshape(-1, shape[-1]))
            return y.view(*shape[:-1], y.shape[-1])
        return layer(x)

    def _wo_a_einsum_from_fp8(
        self, o_fp8: torch.Tensor, o_scale: torch.Tensor, B: int, S: int
    ) -> torch.Tensor:
        """One ``fp8_einsum`` call on pre-quantized activations.

        ``fused_inv_rope_fp8_quant`` emits ``(o_fp8 [M, G, K], o_scale
        [M, G, K/512])`` in the exact layout ``deep_gemm.fp8_einsum``
        consumes, so the wo_a projection is a single einsum launch.
        Matches vLLM ``deepseek_v4_attention.py:325`` (same
        ``"bhr,hdr->bhd"`` + recipe ``(1, 1, 128)`` for SM100 UE8M0)."""
        M, G, _K = o_fp8.shape
        R = self.o_lora_rank
        out = torch.empty(M, G, R, dtype=torch.bfloat16, device=o_fp8.device)
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8, o_scale),
            (self._wo_a_stk_w, self._wo_a_stk_s),
            out,
            recipe=(1, 1, 128),
        )
        return out.view(B, S, G, R)

    def forward_decode(
        self,
        x: torch.Tensor,  # [B, 1, dim] bf16  (q_len=1 pure decode)
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
        kv_cache: Optional[Any] = None,
    ) -> torch.Tensor:
        """qwen3-style: ``kv_cache`` (framework KVCache handle) flows in
        as a kwarg. Stashed for the duration of this call so pool views
        resolve via ``self._pool_view(...)``; block tables for decode
        come from ``attn_metadata.pool_block_tables`` — stashed onto
        ``self._block_tables_by_type`` here so Compressor / Indexer pool
        context resolution shares one code path with prefill."""
        prev_kv = self._kv_cache
        prev_bt = self._block_tables_by_type
        if kv_cache is not None:
            self._kv_cache = kv_cache
        if attn_metadata.pool_block_tables is not None:
            self._block_tables_by_type = attn_metadata.pool_block_tables
        try:
            self._set_compressor_pool_context()
            try:
                return self._forward_decode_body(x, attn_metadata)
            finally:
                self._clear_compressor_pool_context()
        finally:
            self._kv_cache = prev_kv
            self._block_tables_by_type = prev_bt

    def _forward_decode_body(
        self,
        x: torch.Tensor,  # [B, 1, dim] bf16  (q_len=1 pure decode)
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """Decode attention body — thin dispatcher mirroring ``_forward_prefill``.

        Pipeline:
          1. Q/KV + per-request partial RoPE (``decode_compute_qkv``).
          2. FP8 SWA pool write (``decode_write_swa_fp8``).
          3. Per-``compress_ratio`` body:
             * ``0``   → :meth:`_forward_decode_swa_only`
             * ``4``   → :meth:`_forward_decode_csa`   (indexer + compressor)
             * ``128`` → :meth:`_forward_decode_hca`   (compressor, dense idx)
          4. Output projection (``decode_output_proj``).

        Compressor / Indexer freqs_cis is bound lazily on first call
        (pool context is set in :meth:`forward_decode`'s try/finally).
        """
        from rtp_llm.models_py.modules.dsv4.fp8.decode.compute_qkv import (
            decode_compute_qkv,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.output_proj import (
            decode_output_proj,
        )

        bsz, q_len, _ = x.size()
        assert q_len == 1, "decode: q_len==1 only (MTP/spec-decode is later)"
        start_pos = attn_metadata.start_pos[:bsz]  # [bsz] int32

        self._ensure_freqs_cis_bound()

        qkv = decode_compute_qkv(self, x, start_pos)
        self._decode_write_swa_fp8(qkv.kv, bsz, q_len, attn_metadata)

        if self.compress_ratio == 0:
            o = self._forward_decode_swa_only(qkv.q, bsz, q_len, attn_metadata)
        elif self.compress_ratio == 4:
            o = self._forward_decode_csa(x, qkv, bsz, q_len, start_pos, attn_metadata)
        elif self.compress_ratio == 128:
            o = self._forward_decode_hca(x, qkv, bsz, q_len, start_pos, attn_metadata)
        else:
            raise AssertionError(f"unknown compress_ratio={self.compress_ratio}")

        return decode_output_proj(self, o, qkv.freqs_cis_per_req, bsz, q_len)

    # ------------------------------------------------------------------
    # Decode per-path bodies + shared epilogue
    # ------------------------------------------------------------------
    def _decode_write_swa_fp8(
        self,
        kv: torch.Tensor,  # [B, q_len, head_dim] bf16
        bsz: int,
        q_len: int,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
    ) -> None:
        """Write newly computed SWA KV into the FP8 584B/slot pool.

        Mirrors :meth:`_prefill_write_swa_fp8_paged` for decode — uses
        the framework-populated ``pool_write_slot_mappings[SWA_KV]``
        plus the CUDA ``concat_and_cache_mla("fp8_model1_mla", ...)``
        kernel dispatched by ``quantize_v4_kv_decode``.
        """
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8.decode.write_swa import (
            decode_write_swa_fp8,
        )

        slot_mapping = attn_metadata.pool_write_slot_mappings.get(SWA_KV)
        swa_pool_3d = self._pool_view_3d_fp8(SWA_KV)
        decode_write_swa_fp8(
            kv=kv,
            slot_mapping=slot_mapping,
            swa_pool_3d=swa_pool_3d,
            bsz=bsz,
            q_len=q_len,
            head_dim=self.head_dim,
        )

    def _forward_decode_swa_only(
        self,
        q: torch.Tensor,  # [B, 1, H, D]
        bsz: int,
        q_len: int,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """SWA-only layer (compress_ratio == 0) — one FlashMLA call over
        the FP8 SWA pool using per-request global slot ids translated
        from ``swa_abs_idx`` through the SWA block table."""
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8.decode.attention_kernels import (
            attn_fp8_swa_paged,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
            get_or_build_sched_meta,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
            build_req_id_per_token,
            translate_local_to_global_slots,
        )

        swa_pool_3d = self._pool_view_3d_fp8(SWA_KV)
        swa_pool_bt = (
            attn_metadata.pool_block_tables.get(SWA_KV)
            if attn_metadata.pool_block_tables
            else None
        )
        assert (
            swa_pool_3d is not None
            and swa_pool_bt is not None
            and swa_pool_bt.numel() > 0
            and attn_metadata.swa_abs_idx is not None
        ), (
            f"[DSV4 decode SWA-only] FP8 pool + block table + swa_abs_idx "
            f"required (layer={self.layer_id}); "
            f"kv_cache_bound={self._kv_cache is not None}, "
            f"pool_3d_ok={swa_pool_3d is not None}, "
            f"pool_block_tables_keys={list(attn_metadata.pool_block_tables.keys()) if attn_metadata.pool_block_tables else None}, "
            f"swa_pool_bt_shape={tuple(swa_pool_bt.shape) if swa_pool_bt is not None else None}, "
            f"swa_abs_idx_shape={tuple(attn_metadata.swa_abs_idx.shape) if attn_metadata.swa_abs_idx is not None else None}"
        )
        win = self.window_size
        T = bsz * q_len
        # FlashMLA's sparse FP8 kernel reads the packed pool via direct
        # ``pool[indices[i]]`` addressing — no block-table indirection on
        # the kernel side — so ``indices`` MUST be global slot ids, not
        # abs positions. Mirrors the CSA/HCA dual-pool path below.
        if attn_metadata.swa_global_slots is not None:
            swa_global = attn_metadata.swa_global_slots[:T]
        else:
            req_id = build_req_id_per_token(bsz, q_len, swa_pool_bt.device)
            swa_local = attn_metadata.swa_abs_idx[:bsz].reshape(T, win)
            swa_global = translate_local_to_global_slots(
                req_id,
                swa_pool_bt[:bsz],
                swa_local,
                self._pool_entries_per_block(SWA_KV),
            )
        swa_topk_3d = swa_global.view(bsz, q_len, win).contiguous()

        sched_meta = get_or_build_sched_meta(
            attn_metadata,
            batch_size=bsz,
            q_len=q_len,
            num_heads=self.n_heads,
            topk=self.window_size,
            extra_attn_type=None,
        )
        return attn_fp8_swa_paged(
            q=q,
            swa_pool_3d=swa_pool_3d,
            attn_sink=self.attn_sink,
            swa_topk_3d=swa_topk_3d,
            cache_seqlens=attn_metadata.cache_seqlens_i32[:bsz],
            swa_block_table=swa_pool_bt[:bsz],
            sched_meta=sched_meta,
            fp8_op=self._get_fp8_decode_op(),
        )

    def _forward_decode_csa(
        self,
        x: torch.Tensor,
        qkv: "DecodeQKV",  # type: ignore[name-defined]
        bsz: int,
        q_len: int,
        start_pos: torch.Tensor,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """CSA layer (compress_ratio == 4). Indexer + main compressor
        both scatter into their pools; the indexer's topk buffer holds
        raw (pre +win) compressed local indices which the shared
        dual-pool epilogue consumes."""
        from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV

        assert self.indexer is not None, "CSA layer must have an indexer"
        # Indexer fills ``topk_buffer_compressed[:bsz]`` + self-scatters
        # nested compressor state into INDEXER_KV / INDEXER_STATE.
        self.indexer.forward_decode_vectorized(
            x,
            qkv.qr,
            start_pos,
            attn_metadata.topk_buffer_compressed[:bsz],
        )
        # Main CSA compressor emits boundary compressed-K into
        # CSA_KV / CSA_STATE (required by the dual-pool paged read).
        self.compressor.forward_decode_vectorized(x, start_pos)
        # CSA cmp_local_raw = indexer's raw indices (the +win offset is
        # added later inside the epilogue's translate path).
        cmp_local_raw = attn_metadata.topk_buffer_compressed[:bsz].clone()
        return self._forward_decode_compressed(
            qkv.q,
            cmp_local_raw,
            bsz,
            q_len,
            attn_metadata,
            cmp_attn_type=CSA_KV,
        )

    def _forward_decode_hca(
        self,
        x: torch.Tensor,
        qkv: "DecodeQKV",  # type: ignore[name-defined]
        bsz: int,
        q_len: int,
        start_pos: torch.Tensor,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """HCA layer (compress_ratio == 128). Compressor writes
        HCA_KV / HCA_STATE (no indexer). ``cmp_local_raw`` is the dense
        idx precomputed once per step by
        ``update_decode_metadata_in_place._build_dense_compressed_idxs``
        (reused across all HCA layers via ``topk_total_by_ratio[128]``)."""
        from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV

        assert self.indexer is None, "HCA layer must not have an indexer"
        self.compressor.forward_decode_vectorized(x, start_pos)
        win = self.window_size
        tt_h = attn_metadata.topk_total_by_ratio.get(int(self.compress_ratio))
        assert tt_h is not None, (
            f"[DSV4 decode HCA] topk_total_by_ratio[{int(self.compress_ratio)}] "
            f"missing (layer={self.layer_id})"
        )
        cmp_local_raw = tt_h[:bsz, :, win:]
        return self._forward_decode_compressed(
            qkv.q,
            cmp_local_raw,
            bsz,
            q_len,
            attn_metadata,
            cmp_attn_type=HCA_KV,
        )

    def _forward_decode_compressed(
        self,
        q: torch.Tensor,  # [B, 1, H, D]
        cmp_local_raw: torch.Tensor,  # [B, 1, K_cmp] int32 pool-local idx
        bsz: int,
        q_len: int,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
        cmp_attn_type: int,
    ) -> torch.Tensor:
        """Shared CSA/HCA epilogue: translate pool-local → global slots
        for both SWA and compressed pools, then one dual-pool FlashMLA
        call (``extra_k_cache`` + ``extra_indices_in_kvcache`` merges
        softmax in-kernel; mirrors vLLM ``deepseek_v4_attention.py:849-865``)."""
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8.decode.attention_kernels import (
            attn_fp8_dual_paged,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
            get_or_build_sched_meta,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
            build_req_id_per_token,
            translate_local_to_global_slots,
        )

        win = self.window_size
        T = bsz * q_len
        K_cmp = cmp_local_raw.shape[-1]

        swa_pool_3d = self._pool_view_3d_fp8(SWA_KV)
        swa_pool_bt = attn_metadata.pool_block_tables.get(SWA_KV)
        cmp_pool_3d = self._pool_view_3d_fp8(cmp_attn_type)
        cmp_pool_bt = attn_metadata.pool_block_tables.get(cmp_attn_type)
        assert (
            swa_pool_3d is not None
            and cmp_pool_3d is not None
            and swa_pool_bt is not None
            and swa_pool_bt.numel() > 0
            and cmp_pool_bt is not None
            and cmp_pool_bt.numel() > 0
            and attn_metadata.swa_abs_idx is not None
        ), (
            f"[DSV4 decode compressed] FP8 SWA + cmp pools + block tables "
            f"required (layer={self.layer_id}, cmp_attn_type={cmp_attn_type}); "
            f"kv_cache_bound={self._kv_cache is not None}"
        )

        # Translate SWA local → global slots (iter3.3: metadata caches
        # the result once per step, shared across all 43 layers).
        if attn_metadata.swa_global_slots is not None:
            req_id = attn_metadata.req_id_per_token[:T]
            swa_global = attn_metadata.swa_global_slots[:T]
        else:
            req_id = build_req_id_per_token(bsz, q_len, swa_pool_bt.device)
            swa_local = attn_metadata.swa_abs_idx[:bsz].reshape(T, win)
            swa_global = translate_local_to_global_slots(
                req_id,
                swa_pool_bt[:bsz],
                swa_local,
                self._pool_entries_per_block(SWA_KV),
            )

        # Translate compressed local → global slots. HCA layers share a
        # precomputed ``hca_cmp_global_slots`` (dense idx input is
        # identical across HCA layers); CSA layers must translate per-
        # layer because ``topk_buffer_compressed`` is populated by the
        # per-layer indexer.
        if self.indexer is None and attn_metadata.hca_cmp_global_slots is not None:
            cmp_global = attn_metadata.hca_cmp_global_slots[:T]
        else:
            cmp_global = translate_local_to_global_slots(
                req_id,
                cmp_pool_bt[:bsz],
                cmp_local_raw.reshape(T, K_cmp),
                self._pool_entries_per_block(cmp_attn_type),
            )

        # Iter3.2: swa_global / cmp_global are int32 from the translator,
        # so only ``.view + .contiguous`` is needed for FlashMLA's
        # [B, q_len, K] contract.
        swa_topk_3d = swa_global.view(bsz, q_len, win).contiguous()
        cmp_topk_3d = cmp_global.view(bsz, q_len, K_cmp).contiguous()

        sched_meta = get_or_build_sched_meta(
            attn_metadata,
            batch_size=bsz,
            q_len=q_len,
            num_heads=self.n_heads,
            topk=win,
            extra_attn_type=cmp_attn_type,
        )
        return attn_fp8_dual_paged(
            q=q,
            swa_pool_3d=swa_pool_3d,
            cmp_pool_3d=cmp_pool_3d,
            attn_sink=self.attn_sink,
            swa_topk_3d=swa_topk_3d,
            cmp_topk_3d=cmp_topk_3d,
            cache_seqlens=attn_metadata.cache_seqlens_i32[:bsz],
            swa_block_table=swa_pool_bt[:bsz],
            sched_meta=sched_meta,
            fp8_op=self._get_fp8_decode_op(),
        )

    # ==================================================================
    # Prefill — flat ``[T, dim]`` input, B==1 invariant.
    # ==================================================================
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[Any] = None,
        block_tables_by_type: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Prefill entry point.

        ``x``: flat ``[T, dim]`` (single-request, B==1 — enforced by
        the FIFO scheduler's ``max_context_batch_size=1`` setting and
        ``DeepSeekV4Model.forward``). ``positions``: ``[T]`` int64 of
        absolute token positions; ``positions[0]`` is the prefill
        start position. We don't read it eagerly — under broadcast
        meta the sp_int is already on ``self._prefill_meta_shared``
        (synced once in ``forward.py`` for all layers); standalone
        path syncs once inside ``_build_shared_prefill_meta``.
        ``kv_cache`` and ``block_tables_by_type`` are stashed on
        ``self`` for the duration of the call so the many
        ``_prefill_*`` / pool helpers can resolve via
        ``self._kv_cache`` without threading the handles through
        every signature.
        """
        assert (
            x.dim() == 2
        ), f"DSv4 Attention prefill expects flat [T, dim]; got shape {tuple(x.shape)}"
        # Prefill is FP8-only on this branch — every downstream helper
        # (``_prefill_write_swa_fp8_paged``, ``_attn_fp8_swa_via_kv_full``,
        # ``_attn_via_workspace``) hard-assumes FP8 KV-cache pools. Hoist
        # downstream can be removed entirely.
        prev_kv = self._kv_cache
        prev_bt = self._block_tables_by_type
        if kv_cache is not None:
            self._kv_cache = kv_cache
        if block_tables_by_type is not None:
            self._block_tables_by_type = block_tables_by_type
        try:
            with record_function_range("dsv4.fp8.attn.set_pool_context"):
                self._set_compressor_pool_context()
            try:
                with record_function_range(
                    f"dsv4.fp8.attn.L{self.layer_id:02d}.prefill"
                ):
                    return self._forward_prefill(x, positions)
            finally:
                with record_function_range("dsv4.fp8.attn.clear_pool_context"):
                    self._clear_compressor_pool_context()
        finally:
            self._kv_cache = prev_kv
            self._block_tables_by_type = prev_bt

    def _forward_prefill(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Prefill body. ``x`` is flat ``[T, dim]``; output flat ``[T, dim]``.

        Three mutually-exclusive paths gated by ``compress_ratio``:
          * 0   → :meth:`_forward_prefill_swa_only`
          * 4   → :meth:`_forward_prefill_csa`   (indexer + compressor)
          * 128 → :meth:`_forward_prefill_hca`   (compressor, dense idx)

        All three share the same prologue: ``_prefill_common_setup`` →
        ``_prefill_compute_qkv`` → ``_prefill_write_swa_fp8_paged``.
        ``csa`` / ``hca`` additionally share the ``[sliding | compressed]``
        kv_cat + sparse_attn epilogue via :meth:`_forward_prefill_compressed`.

        FP8 KV-cache is asserted at the public ``forward()`` entry; this
        body assumes FP8 unconditionally.
        """
        with record_function_range("dsv4.fp8.attn.prefill.common_setup"):
            common = self._prefill_common_setup(x, positions)
        with record_function_range("dsv4.fp8.attn.prefill.compute_qkv"):
            qkv = self._prefill_compute_qkv(x, common)
        # SWA pool write — every FP8 layer populates the SWA pool for
        # downstream decode. Safe to do before attention because new K
        # (abs pos [sp, sp+S)) and any cont-prefill prefix tail
        # (abs pos [sp-P, sp)) target disjoint slots.
        with record_function_range("dsv4.fp8.attn.prefill.swa_write"):
            self._prefill_write_swa_fp8_paged(common, qkv.kv_full)

        if self.compress_ratio == 0:
            with record_function_range("dsv4.fp8.attn.prefill.path_swa"):
                out = self._forward_prefill_swa_only(qkv, common)
        elif self.compress_ratio == 4:
            with record_function_range("dsv4.fp8.attn.prefill.path_csa"):
                out = self._forward_prefill_csa(x, qkv, common)
        elif self.compress_ratio == 128:
            with record_function_range("dsv4.fp8.attn.prefill.path_hca"):
                out = self._forward_prefill_hca(x, qkv, common)
        else:
            raise AssertionError(f"unknown compress_ratio={self.compress_ratio}")
        return out

    # ------------------------------------------------------------------
    # Per-path prefill bodies
    # ------------------------------------------------------------------
    def _forward_prefill_swa_only(
        self, qkv: PrefillQKV, common: PrefillMeta
    ) -> torch.Tensor:
        """SWA-only path (compress_ratio == 0). Skips kv_cat + sparse_attn.
        Cold/warmup attends over BF16 ``kv_full`` directly; continuation
        builds ``[prefix_tail | new_K_bf16]`` in a workspace and runs
        ``flash_mla_sparse_fwd`` over it. ``any_cont`` is varlen-aware
        (set from ``prefix_lengths.any()`` under varlen, ``sp_int > 0``
        otherwise) so a B>1 batch with any continuation request takes the
        workspace path."""
        if not common.any_cont or self._kv_cache is None:
            with record_function_range("dsv4.fp8.attn.swa.via_kv_full"):
                o = self._attn_fp8_swa_via_kv_full(qkv, common)
        else:
            with record_function_range("dsv4.fp8.attn.swa.via_concat"):
                o = self._attn_fp8_swa_via_concat(qkv, common)
        with record_function_range("dsv4.fp8.attn.prefill.output_proj"):
            out_3d = self._prefill_output_proj(o, common)
        return out_3d.squeeze(0)

    def _forward_prefill_csa(
        self,
        x: torch.Tensor,
        qkv: PrefillQKV,
        common: PrefillMeta,
    ) -> torch.Tensor:
        """CSA path (compress_ratio == 4). Sparse compress topk via the
        IndexerFP8 lightning indexer; main compressor writes the CSA
        pool with hoisted meta. Attention runs through the vLLM-style
        workspace path (dual FP8 dequant + BF16 overlay + flash_mla_sparse_fwd).
        Falls back to BF16 ``kv_full`` attention on warmup (workspace_meta None)."""
        from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8

        assert isinstance(
            self.indexer, IndexerFP8
        ), "CSA layer requires IndexerFP8 (BF16 indexer not supported on this branch)"
        assert common.csa_meta is not None, (
            "CSA prefill requires common.csa_meta — built by " "_build_csa_prefill_meta"
        )

        # Phase-3a part 3: IndexerFP8.forward + nested CompressorFP8.forward
        # are now flat-input-native (accept ``[T_total, dim]`` /
        # ``[T_total, q_lora]``). Drop the legacy ``unsqueeze(0)`` so the
        # batched flat caller hits the same code path without rewrapping.
        with record_function_range("dsv4.fp8.attn.csa.indexer"):
            raw = self.indexer(x, qkv.qr, common.csa_meta.indexer_meta)
        return self._forward_prefill_compressed(
            x,
            qkv,
            common,
            cmp_topk_runtime=raw,
            compressor_meta=common.csa_meta.compressor_meta,
            workspace_meta=common.csa_meta.workspace_meta,
        )

    def _forward_prefill_hca(
        self,
        x: torch.Tensor,
        qkv: PrefillQKV,
        common: PrefillMeta,
    ) -> torch.Tensor:
        """HCA path (compress_ratio == 128). Dense compressed indices live
        in ``workspace_meta.dense_cmp_topk``; runtime cmp_topk is None.
        Main compressor writes the HCA pool with hoisted meta."""
        assert self.indexer is None, "HCA layer must not have an indexer"
        assert common.hca_meta is not None, (
            "HCA prefill requires common.hca_meta — built by " "_build_hca_prefill_meta"
        )

        return self._forward_prefill_compressed(
            x,
            qkv,
            common,
            cmp_topk_runtime=None,
            compressor_meta=common.hca_meta.compressor_meta,
            workspace_meta=common.hca_meta.workspace_meta,
        )

    def _forward_prefill_compressed(
        self,
        x: torch.Tensor,
        qkv: PrefillQKV,
        common: PrefillMeta,
        cmp_topk_runtime: Optional[torch.Tensor],
        compressor_meta,
        workspace_meta: Optional[WorkspaceMeta],
    ) -> torch.Tensor:
        """Shared CSA/HCA epilogue: write compressed-K via main compressor
        (with hoisted ``compressor_meta``), then run the workspace-path
        attention. Falls back to ``_attn_fp8_swa_via_kv_full`` on warmup
        when ``workspace_meta`` is None (pool context unbound)."""
        # Compressor write (return value is unused — compressor handles its
        # own pool dual-write; the workspace path re-reads the just-written
        # tail via dequantize_and_gather_k_cache, with BF16 overlay on top).
        # Phase-3a: ``CompressorFP8.forward`` is now flat-input-native
        # (accepts ``[T_total, dim]``). Drop the ``unsqueeze(0)`` legacy
        # rewrap so the batched (B>1) caller can reuse this code path
        # without re-shaping. B==1 reaches the same kernel — same flat
        # ``[T_total, dim]`` reshape inside ``_launch``.
        with record_function_range("dsv4.fp8.attn.compressed.compressor"):
            self.compressor(x, common.sp_int, meta=compressor_meta)

        if workspace_meta is None:
            # Warmup forward: pool not bound. Fall back to BF16 ``kv_full``
            # SWA-only attention so framework shape inference still runs.
            with record_function_range("dsv4.fp8.attn.compressed.warmup_attn"):
                o = self._attn_fp8_swa_via_kv_full(qkv, common)
        else:
            with record_function_range("dsv4.fp8.attn.compressed.workspace_attn"):
                o = self._attn_via_workspace(
                    qkv, common, workspace_meta, cmp_topk_runtime
                )
        with record_function_range("dsv4.fp8.attn.prefill.output_proj"):
            out_3d = self._prefill_output_proj(o, common)
        return out_3d.squeeze(0)

    def _attn_via_workspace(
        self,
        qkv: PrefillQKV,
        common: PrefillMeta,
        workspace_meta: "WorkspaceMeta",
        cmp_topk_runtime: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """vLLM-style workspace path for CSA/HCA prefill (varlen B>=1, non-CP).

        Pipeline (kernel-only — every elementwise op pre-baked in
        :meth:`_build_workspace_meta`):
          1. Allocate ``workspace [B, M, head_dim]`` BF16 zeros.
          2. ``dequantize_and_gather_k_cache(cmp_pool, offset=0)`` →
             ``workspace[b, 0:N_b, :]`` per request (per-req ``cmp_seq_lens``
             handle the per-row variable length; ``[N_b, N_max)`` stays zero).
          3. ``dequantize_and_gather_k_cache(swa_pool, offset=N_max)`` →
             ``workspace[b, N_max:N_max+P_b, :]`` per request. SWA only
             stores the tail blocks; fresh ``S_b`` rows are supplied by step 4.
          4. BF16 overlay freshly computed new K via single ``index_copy_``
             over ``workspace.view(B*M, D)`` using ``wm.new_k_slot_in_flat``
             (already encodes ``M*req_id + N + P_b + local_pos``).
          5. ``combine_topk_swa_indices`` packs the per-query
             ``[compressed_valid | swa_valid]`` index list. The kernel
             internally computes per-token ``min((pos+1)//ratio, TOP_K)``
             so HCA's full ``arange(N_max)`` row gets masked even for tokens
             whose request has ``N_b < N_max``.
          6. ``flash_mla_sparse_fwd`` over the ``[B*M, 1, D]`` workspace view.

        Mirrors vLLM ``DeepseekV4MultiHeadLatentAttentionWrapper._forward_prefill``.

        ``cmp_topk_runtime`` is the indexer output (CSA path); for HCA it's
        ignored and ``workspace_meta.dense_cmp_topk`` (precomputed
        ``arange(N_max)`` per token) is used instead.
        """
        from flash_mla import flash_mla_sparse_fwd  # type: ignore[import-not-found]

        from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, HCA_KV, SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_dequant_triton as _swa_dq
        from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
            combine_topk_swa_indices,
            combine_topk_swa_indices_cp,
        )

        ratio = self.compress_ratio
        ratio_tag = "csa" if ratio == 4 else "hca"
        cmp_at = CSA_KV if ratio == 4 else HCA_KV
        swa_pool_3d = self._pool_view_3d_fp8(SWA_KV)
        cmp_pool_3d = self._pool_view_3d_fp8(cmp_at)
        assert (
            swa_pool_3d is not None and cmp_pool_3d is not None
        ), "FP8 SWA + CSA/HCA pools required for workspace path"

        assert workspace_meta is not None, (
            "_attn_via_workspace requires a non-None WorkspaceMeta; the caller "
            "must short-circuit to _attn_fp8_swa_via_kv_full when the pool is "
            "unbound (warmup)."
        )
        wm = workspace_meta
        # ``B`` from the per-request meta — bit-equal to legacy 1 when meta
        # was built from the scalar branch (swa_seq_lens shape == [1]).
        B = int(wm.swa_seq_lens.shape[0])
        assert B == common.batch_size, (
            f"workspace meta B ({B}) != common.batch_size ({common.batch_size}); "
            "meta builder fed an inconsistent shape — likely a varlen / legacy "
            "dispatch mismatch."
        )
        D = self.head_dim
        with record_function_range("dsv4.fp8.attn.workspace.alloc"):
            workspace = torch.zeros(
                (B, wm.M, D), dtype=torch.bfloat16, device=qkv.q.device
            )

        if wm.N > 0:
            with record_function_range("dsv4.fp8.attn.workspace.gather_cmp"):
                _swa_dq.dequantize_and_gather_k_cache(
                    out=workspace,
                    k_cache=cmp_pool_3d,
                    seq_lens=wm.cmp_seq_lens,
                    gather_lens=None,
                    block_table=wm.cmp_bt_int32,
                    block_size=wm.cmp_eb,
                    offset=0,
                )
        # SWA dequant only reads the cached prefix tail. The fresh new-K rows
        # are already available as BF16 and are overlaid below; cold prefill
        # does not need to read the SWA pool at all.
        if common.any_cont:
            with record_function_range("dsv4.fp8.attn.workspace.gather_swa_prefix"):
                _swa_dq.dequantize_and_gather_k_cache(
                    out=workspace,
                    k_cache=swa_pool_3d,
                    seq_lens=wm.swa_cache_seq_lens,
                    gather_lens=wm.swa_cache_gather_lens,
                    block_table=wm.swa_bt_int32,
                    block_size=wm.swa_eb,
                    offset=wm.N,
                )

        # BF16 overlay of freshly computed new K — single ``index_copy_``
        # using the meta-precomputed ``new_k_slot_in_flat``. Avoids the FP8
        # round-trip loss on tokens we just wrote, while keeping the hot
        # path free of casts / gathers / per-request slicing.
        with record_function_range("dsv4.fp8.attn.workspace.overlay_new_k"):
            kv_bf16 = qkv.kv_full.to(torch.bfloat16).reshape(-1, D)
            workspace.view(B * wm.M, D).index_copy_(0, wm.new_k_slot_in_flat, kv_bf16)

        # combine_topk: HCA uses precomputed dense arange(N_max); CSA uses
        # the runtime indexer output (raw compressed-pool offsets in [0, N_b)).
        # Indexer output contract: ``[T_total, K] int32 contiguous`` (set in
        # ``IndexerFP8.forward`` line 754 + ``view(out_shape)``); assert here
        # so a contract drift surfaces as a loud crash instead of a silent
        # per-layer ``squeeze``/``to``/``contiguous`` retag in the hot path.
        if wm.dense_cmp_topk is not None:
            cmp_topk = wm.dense_cmp_topk
        else:
            assert (
                cmp_topk_runtime is not None
            ), "CSA workspace path requires cmp_topk_runtime (indexer output)"
            assert (
                cmp_topk_runtime.dim() == 2
                and cmp_topk_runtime.dtype == torch.int32
                and cmp_topk_runtime.is_contiguous()
            ), (
                "cmp_topk_runtime contract violated: expected 2D int32 "
                f"contiguous, got dim={cmp_topk_runtime.dim()} "
                f"dtype={cmp_topk_runtime.dtype} contig={cmp_topk_runtime.is_contiguous()}"
            )
            cmp_topk = cmp_topk_runtime

        if common.cp_on:
            # Phase F2/Phase-2: kernel ``combine_topk_swa_indices`` derives
            # per-Q-row ``pos = start_pos + token_idx_in_query`` assuming Q
            # is a contiguous slice. Under zigzag CP each rank's Q rows
            # have non-contiguous global positions, so use the CP fused
            # kernel that consumes explicit positions directly.
            assert common.cp_ctx is not None
            cp_ctx_local = common.cp_ctx
            legacy_prefix_length = int(cp_ctx_local.prefix_length)
            use_cp_varlen = _use_varlen_prefill()
            if not use_cp_varlen:
                # Legacy DSV4_VARLEN_PREFILL=0 path keeps its B==1 invariant.
                assert (
                    int(wm.swa_seq_lens.shape[0]) == 1
                ), "legacy CP workspace path supports B=1; set DSV4_VARLEN_PREFILL=1 for B>1"
            combine_kwargs = dict(
                topk_indices=cmp_topk,
                global_positions=_flat_1d(cp_ctx_local.global_positions),
                sp_int=legacy_prefix_length,
                window_size=self.window_size,
                compress_ratio=ratio,
                topk=int(cmp_topk.shape[-1]),
                M=wm.M,
                N=wm.N,
            )
            if use_cp_varlen:
                assert common.req_id_per_token is not None
                assert common.prefix_lengths is not None
                combine_kwargs.update(
                    req_id_per_token=_flat_1d(common.req_id_per_token),
                    prefix_lengths=_flat_1d(common.prefix_lengths),
                )
            with record_function_range("dsv4.fp8.attn.workspace.combine_topk_cp"):
                combined_indices, combined_lens = combine_topk_swa_indices_cp(
                    **combine_kwargs
                )
        else:
            with record_function_range("dsv4.fp8.attn.workspace.combine_topk"):
                combined_indices, combined_lens = combine_topk_swa_indices(
                    topk_indices=cmp_topk,
                    query_start_loc=wm.qsl,
                    seq_lens=wm.swa_seq_lens,
                    gather_lens=wm.swa_gather_lens,
                    window_size=self.window_size,
                    compress_ratio=ratio,
                    topk=int(cmp_topk.shape[-1]),
                    M=wm.M,
                    N=wm.N,
                )

        # flash_mla_sparse_fwd is called once when ``s_q`` fits the safe
        # threshold; otherwise it is chunked along Q so each launch stays
        # below ``DSV4_FLASH_MLA_SPARSE_Q_CHUNK`` rows. Sparse attention
        # has no cross-Q dependency so chunking is bit-equal. The Q-chunk
        # is defensive: with the int64 row-indexing fix in
        # ``_combine_topk_swa_indices_cp_kernel`` long-context HCA L3
        # prefill works in a single launch, but the chunk caps still
        # protect against any latent int32 indexing inside flash_mla
        # itself. Set chunk <= 0 to disable.
        kv_view = workspace.view(B * wm.M, 1, D)
        indices_3d = combined_indices.unsqueeze(1)

        q_chunk_env = os.environ.get("DSV4_FLASH_MLA_SPARSE_Q_CHUNK", "65536")
        try:
            q_chunk = int(q_chunk_env)
        except ValueError:
            q_chunk = 65536
        s_q = qkv.q.shape[0]
        if q_chunk <= 0 or s_q <= q_chunk:
            with record_function_range("dsv4.fp8.attn.workspace.flash_mla_sparse_fwd"):
                o3, _, _ = flash_mla_sparse_fwd(
                    q=qkv.q,
                    kv=kv_view,
                    indices=indices_3d,
                    sm_scale=self.softmax_scale,
                    attn_sink=self.attn_sink,
                    topk_length=combined_lens,
                )
            return o3.unsqueeze(0)

        chunks: list[torch.Tensor] = []
        with record_function_range("dsv4.fp8.attn.workspace.flash_mla_sparse_fwd"):
            for start in range(0, s_q, q_chunk):
                end = min(start + q_chunk, s_q)
                o_part, _, _ = flash_mla_sparse_fwd(
                    q=qkv.q[start:end],
                    kv=kv_view,
                    indices=indices_3d[start:end],
                    sm_scale=self.softmax_scale,
                    attn_sink=self.attn_sink,
                    topk_length=combined_lens[start:end],
                )
                chunks.append(o_part)
        o3 = torch.cat(chunks, dim=0)
        return o3.unsqueeze(0)

    # ------------------------------------------------------------------
    # Per-call meta + Q/KV
    # ------------------------------------------------------------------
    def _set_prefill_meta_shared(self, meta: Optional["PrefillMeta"]) -> None:
        """Inject the (compress_ratio bucket) shared prefill meta built by
        the upper layer (V4Transformer.forward_layers). The shared meta is
        layer-invariant within a ratio, so the upper layer builds it once
        per ratio and broadcasts to every layer attention sharing that
        ratio. ``None`` clears the binding (used at end of forward).
        """
        self._prefill_meta_shared = meta

    def _ensure_freqs_cis_bound(self) -> None:
        """Bind ``self.freqs_cis`` onto this layer's compressor / indexer
        chain so their forward()s can read ``self.freqs_cis`` without an
        extra parameter. Idempotent — safe to call many times. Required
        on every layer (not just the meta-build rep) because each layer
        owns its own compressor / indexer instance.
        """
        if not self.compress_ratio:
            return
        if self.compressor.freqs_cis is None:
            self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            if self.indexer.freqs_cis is None:
                self.indexer.freqs_cis = self.freqs_cis
            if self.indexer.compressor.freqs_cis is None:
                self.indexer.compressor.freqs_cis = self.freqs_cis

    def _build_shared_prefill_meta(
        self,
        x: torch.Tensor,
        positions: Union[int, torch.Tensor],
        sp_per_req: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
    ) -> "PrefillMeta":
        """Build the layer-invariant (within compress_ratio bucket) part
        of per-call prefill metadata. All host-side prep work that
        doesn't depend on ``self.layer_id`` lives here so the upper layer
        can run it once per ratio and broadcast to every same-ratio
        attention via :meth:`_set_prefill_meta_shared`. Standalone path
        falls back to running this per-layer.

        ``positions`` accepts either a ``[T]`` int64 tensor (from the
        normal call path) or a pre-synced int (used by upper-layer
        broadcast meta builders that already paid the sync once for
        the whole batch). When a tensor is passed we sync once here.
        """
        seqlen = int(x.shape[0])
        rd = self.rope_head_dim
        device = x.device

        # CP plumbing: this builder threads ``cp_ctx`` onto ``PrefillMeta`` so
        # downstream SWA write / sparse attention / compressor paths can switch
        # between rank-local Q metadata and the all-gathered KV/write view.
        # CP=1 collapses to the legacy single-rank values.
        cp_ctx = getattr(self, "_cp_ctx", None)
        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1

        # Sync ``positions[0]`` -> int once. Tensor input pays the sync here
        # exactly once per (forward, ratio bucket); int input has already
        # been synced by the upper-layer broadcast builder. Used by the
        # ``DSV4_VARLEN_PREFILL=0`` legacy fallback only.
        if isinstance(positions, torch.Tensor):
            sp_int = int(positions.reshape(-1)[0].item())
        else:
            sp_int = int(positions)

        # Single contract guard for the entire prefill stack. Every downstream
        # builder (``_build_workspace_meta``, ``_build_compressor_meta``,
        # ``_build_csa_prefill_meta``, ``IndexerFP8.prepare``) trusts the
        # same invariant — they don't re-check varlen tensors / re-assert
        # ``batch_size``:
        #   * ``use_varlen=True``  ⇒ caller MUST populate every per-request
        #     tensor (varlen path consumes them directly).
        #   * ``use_varlen=False`` ⇒ legacy ``DSV4_VARLEN_PREFILL=0`` bisect
        #     channel; B==1 only (the legacy scalar plumbing collapses every
        #     formula on ``sp_int`` / ``seqlen`` ⇒ wrong meta under B>1).
        # Branch fused with the freqs/topk/any_cont compute below so pyright
        # narrowing on the per-request tensors flows through to the
        # _varlen sub-builder calls (sequential ``is not None`` asserts so
        # narrowing kicks in at each one).
        use_varlen = _use_varlen_prefill()
        win = self.window_size
        # Under CP each rank holds only ``chunk_length`` tokens locally; the
        # full prefill sequence has length ``cp_ctx.seq_len_full``. Code that
        # needs the global length (SWA pool write after all-gather, KV-side
        # cu_seqlens) reads ``seqlen_full``; rank-local code keeps using
        # ``seqlen``.
        seqlen_full = cp_ctx.seq_len_full if cp_on else seqlen

        if use_varlen:
            _msg = (
                "varlen prefill requires the upper layer to populate "
                "cu_seqlens / input_lengths / prefix_lengths / position_ids / "
                "req_id_per_token / sp_per_req / batch_size / max_seqlen_q. "
                "Set DSV4_VARLEN_PREFILL=0 to take the legacy scalar path."
            )
            assert cu_seqlens is not None, _msg
            assert input_lengths is not None, _msg
            assert prefix_lengths is not None, _msg
            assert position_ids is not None, _msg
            assert req_id_per_token is not None, _msg
            assert sp_per_req is not None, _msg
            assert batch_size > 0 and max_seqlen_q > 0, _msg

            cu_seqlens = _flat_1d(cu_seqlens)
            input_lengths = _flat_1d(input_lengths)
            prefix_lengths = _flat_1d(prefix_lengths)
            position_ids = _flat_1d(position_ids)
            req_id_per_token = _flat_1d(req_id_per_token)
            sp_per_req = _flat_1d(sp_per_req)
            assert (
                position_ids.numel() == seqlen
            ), f"position_ids must be flat [T_total={seqlen}], got {position_ids.shape}"
            assert (
                req_id_per_token.numel() == seqlen
            ), f"req_id_per_token must be flat [T_total={seqlen}], got {req_id_per_token.shape}"
            assert (
                cu_seqlens.numel() == batch_size + 1
            ), f"cu_seqlens must be [B+1={batch_size + 1}], got {cu_seqlens.shape}"
            assert (
                input_lengths.numel() == batch_size
            ), f"input_lengths must be [B={batch_size}], got {input_lengths.shape}"
            assert (
                prefix_lengths.numel() == batch_size
            ), f"prefix_lengths must be [B={batch_size}], got {prefix_lengths.shape}"
            assert (
                sp_per_req.numel() == batch_size
            ), f"sp_per_req must be [B={batch_size}], got {sp_per_req.shape}"

            position_ids_eff = position_ids
            cu_seqlens_for_k = cu_seqlens
            if cp_on:
                assert cp_ctx is not None
                position_ids_eff = _flat_1d(
                    cp_ctx.global_positions.to(device=device, dtype=torch.long)
                )
                if cp_ctx.cu_seqlens_global is not None:
                    cu_seqlens_for_k = _flat_1d(
                        cp_ctx.cu_seqlens_global.to(device=device, dtype=torch.int32)
                    )
            # Per-token absolute-position RoPE gather. For B==1 contiguous
            # this is bit-equal to the legacy slice; for B>1 it's the only
            # correct option since requests interleave on the flat token axis.
            with record_function_range("dsv4.fp8.meta.varlen.freqs_topk"):
                freqs_cis = self.freqs_cis.index_select(
                    0,
                    position_ids_eff.to(device=self.freqs_cis.device, dtype=torch.long),
                )
                topk_idxs = _get_window_topk_idxs_varlen(
                    win,
                    cu_seqlens_for_k,
                    position_ids_eff,
                    prefix_lengths,
                    req_id_per_token,
                )  # [T_total, win]
                # One .item() sync per forward — the same order of magnitude as
                # ``int(positions[0].item())`` we already pay above.
                any_cont = bool((prefix_lengths > 0).any().item())

            # SWA dispatch inlined (was the trivial ``_build_swa_prefill_meta``
            # trampoline). Per-request tensors flow in directly with the
            # narrowed Tensor (non-Optional) types from the asserts above.
            with record_function_range("dsv4.fp8.meta.swa_varlen"):
                swa_meta = self._build_swa_prefill_meta_varlen(
                    seqlen=seqlen,
                    device=device,
                    any_cont=any_cont,
                    batch_size=batch_size,
                    cu_seqlens=cu_seqlens,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                )
        else:
            assert batch_size == 1, (
                "DSV4_VARLEN_PREFILL=0 (legacy scalar path) only supports "
                f"B==1; got batch_size={batch_size}. Set DSV4_VARLEN_PREFILL=1 "
                "for batched prefill."
            )
            # ``DSV4_VARLEN_PREFILL=0`` regression channel — legacy B==1 path.
            with record_function_range("dsv4.fp8.meta.legacy.freqs_topk"):
                freqs_cis = self.freqs_cis[sp_int : sp_int + seqlen]
                topk_idxs = _get_window_topk_idxs(win, 1, seqlen, sp_int, device)
                any_cont = sp_int > 0
            with record_function_range("dsv4.fp8.meta.swa_legacy"):
                swa_meta = self._build_swa_prefill_meta_legacy(seqlen, sp_int, device)

        # Bind freqs_cis to this layer's compressor / indexer chain
        # (idempotent — safe to call from both standalone and meta-broadcast paths).
        self._ensure_freqs_cis_bound()

        # row_seqlens_full: [1] long tensor. Reused by SWA pool read/write
        # helpers (BF16 path) — they refuse a None for the per-row seqlens.
        row_seqlens_full = torch.tensor([seqlen_full], device=device, dtype=torch.long)

        # ``use_varlen`` threaded explicitly to the compressor sub-builders;
        # they consume it as the single dispatch switch (env is read exactly
        # once per forward / ratio bucket above, never re-read downstream).
        csa_meta: Optional[CsaPrefillMeta] = None
        hca_meta: Optional[HcaPrefillMeta] = None
        if self.compress_ratio == 4:
            with record_function_range("dsv4.fp8.meta.csa"):
                csa_meta = self._build_csa_prefill_meta(
                    seqlen,
                    sp_int,
                    device,
                    use_varlen=use_varlen,
                    batch_size=batch_size,
                    cu_seqlens=cu_seqlens,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    sp_per_req=sp_per_req,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    max_seqlen_q=max_seqlen_q,
                )
        elif self.compress_ratio == 128:
            with record_function_range("dsv4.fp8.meta.hca"):
                hca_meta = self._build_hca_prefill_meta(
                    seqlen,
                    sp_int,
                    device,
                    use_varlen=use_varlen,
                    batch_size=batch_size,
                    cu_seqlens=cu_seqlens,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    sp_per_req=sp_per_req,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    max_seqlen_q=max_seqlen_q,
                )

        return PrefillMeta(
            seqlen=seqlen,
            seqlen_full=seqlen_full,
            rd=rd,
            device=device,
            cp_ctx=cp_ctx,
            cp_on=cp_on,
            freqs_cis=freqs_cis,
            topk_idxs=topk_idxs,
            sp_int=sp_int,
            any_cont=any_cont,
            row_seqlens_full=row_seqlens_full,
            use_varlen=use_varlen,
            sp_per_req=sp_per_req,
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            position_ids=position_ids,
            req_id_per_token=req_id_per_token,
            max_seqlen_q=max_seqlen_q,
            swa_meta=swa_meta,
            csa_meta=csa_meta,
            hca_meta=hca_meta,
        )

    # ------------------------------------------------------------------
    # Per-ratio compressor metadata builders (CSA / HCA)
    # ------------------------------------------------------------------
    def _build_csa_prefill_meta(
        self,
        seqlen: int,
        sp_int: int,
        device: torch.device,
        *,
        use_varlen: bool,
        batch_size: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        sp_per_req: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
    ) -> CsaPrefillMeta:
        """Build CSA-layer per-call metadata: indexer prepare + main CSA
        compressor prepare_metadata.

        Pool context binding: this method runs from the broadcast-meta
        path (``forward_layers`` → ``_build_and_propagate_prefill_meta``)
        BEFORE the per-layer ``_set_compressor_pool_context`` would
        otherwise fire inside ``_forward_prefill_internal_wrapper``, so we
        bind here ourselves. Without this bind the indexer's
        ``self._kv_block_table`` / ``_state_block_table`` are still None
        and the hoist inside ``IndexerFP8.prepare`` silently no-ops —
        ``compressor_meta`` comes back as None and the per-call slot
        mapping (~20 small kernels per CSA layer) ends up rebuilt on the
        hot path between the FP32 SGEMM and ``_save_partial_states_kernel``.

        One bind covers both halves: ``_set_compressor_pool_context``
        wires the indexer AND the host compressor in the same call, and
        ``_build_compressor_meta`` is inlined here so we don't redundantly
        re-bind for the second meta. No try/finally — if something below
        raises we want the process to die instead of leaking a stale
        binding into the next call.
        """
        from rtp_llm.models_py.modules.dsv4.attn_type import INDEXER_KV
        from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
            build_prepare_metadata_args,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8

        assert isinstance(self.indexer, IndexerFP8), "CSA layer requires IndexerFP8"

        idx_bt = (
            self._block_tables_by_type.get(INDEXER_KV)
            if self._block_tables_by_type is not None
            else None
        )
        idx_eb = self._pool_entries_per_block(INDEXER_KV)

        with record_function_range("dsv4.fp8.meta.csa.bind_pool"):
            self._set_compressor_pool_context()
        with record_function_range("dsv4.fp8.meta.csa.indexer_prepare"):
            indexer_meta = self.indexer.prepare(
                bsz=1,
                seqlen=seqlen,
                sp_int=sp_int,
                device=device,
                kv_block_table=idx_bt,
                kv_eb=idx_eb,
                use_varlen=use_varlen,
                batch_size=batch_size,
                cu_seqlens=cu_seqlens,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                position_ids=position_ids,
                req_id_per_token=req_id_per_token,
                max_seqlen_q=max_seqlen_q,
            )
        cp_ctx_local = getattr(self, "_cp_ctx", None)
        cp_active = cp_ctx_local is not None and cp_ctx_local.cp_size > 1
        if cp_active:
            # Under CP the main compressor all-gathers KV/score to the full
            # global sequence. Build the matching full-sequence metadata once
            # per ratio bucket here (the prefill_meta broadcast path) instead
            # of rebuilding it inside every layer's compressor.forward.
            assert cp_ctx_local is not None
            with record_function_range("dsv4.fp8.meta.csa.cp_compressor_prepare"):
                (
                    cp_positions,
                    cp_b_idx,
                    cp_seq_start_per_req,
                    cp_cu_seq_per_req,
                ) = build_cp_full_prefill_positions(cp_ctx_local, device)
                assert cp_ctx_local.input_lengths_global is not None
                compressor_meta = self.compressor.prepare_metadata(
                    cp_positions,
                    cp_b_idx,
                    is_batched=bool(cp_ctx_local.input_lengths_global.numel() > 1),
                    seq_start_per_req=cp_seq_start_per_req,
                    cu_seq_per_req=cp_cu_seq_per_req,
                )
        else:
            # Inline (vs. calling ``_build_compressor_meta``) is load-bearing:
            # it shares the surrounding pool bind with ``indexer.prepare`` so
            # the indexer's nested compressor-meta hoist fires under one bind.
            with record_function_range("dsv4.fp8.meta.csa.compressor_prepare"):
                cmp_args = build_prepare_metadata_args(
                    use_varlen=use_varlen,
                    device=device,
                    sp_int=sp_int,
                    seqlen=seqlen,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    seq_start_per_req=sp_per_req,
                    cu_seqlens=cu_seqlens,
                )
                compressor_meta = self.compressor.prepare_metadata(**cmp_args)
        with record_function_range("dsv4.fp8.meta.csa.clear_pool"):
            self._clear_compressor_pool_context()

        with record_function_range("dsv4.fp8.meta.csa.workspace"):
            workspace_meta = self._build_workspace_meta(
                seqlen,
                sp_int,
                device,
                with_dense_cmp_topk=False,
                use_varlen=use_varlen,
                batch_size=batch_size,
                cu_seqlens=cu_seqlens,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sp_per_req=sp_per_req,
                position_ids=position_ids,
                req_id_per_token=req_id_per_token,
                max_seqlen_q=max_seqlen_q,
            )
        return CsaPrefillMeta(
            indexer_meta=indexer_meta,
            compressor_meta=compressor_meta,
            workspace_meta=workspace_meta,
        )

    def _build_hca_prefill_meta(
        self,
        seqlen: int,
        sp_int: int,
        device: torch.device,
        *,
        use_varlen: bool,
        batch_size: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        sp_per_req: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
    ) -> HcaPrefillMeta:
        """Build HCA-layer per-call metadata: main HCA compressor
        prepare_metadata."""
        cp_ctx_local = getattr(self, "_cp_ctx", None)
        if cp_ctx_local is not None and cp_ctx_local.cp_size > 1:
            # CP compressor metadata is for the all-gathered global sequence.
            # It is still layer-invariant within the HCA ratio bucket, so build
            # it in the prefill_meta broadcast path and pass it to the hot path.
            with record_function_range("dsv4.fp8.meta.hca.cp_compressor_prepare"):
                self._set_compressor_pool_context()
                try:
                    (
                        cp_positions,
                        cp_b_idx,
                        cp_seq_start_per_req,
                        cp_cu_seq_per_req,
                    ) = build_cp_full_prefill_positions(cp_ctx_local, device)
                    assert cp_ctx_local.input_lengths_global is not None
                    compressor_meta = self.compressor.prepare_metadata(
                        cp_positions,
                        cp_b_idx,
                        is_batched=bool(cp_ctx_local.input_lengths_global.numel() > 1),
                        seq_start_per_req=cp_seq_start_per_req,
                        cu_seq_per_req=cp_cu_seq_per_req,
                    )
                finally:
                    self._clear_compressor_pool_context()
        else:
            with record_function_range("dsv4.fp8.meta.hca.compressor_prepare"):
                compressor_meta = self._build_compressor_meta(
                    seqlen,
                    sp_int,
                    device,
                    use_varlen=use_varlen,
                    batch_size=batch_size,
                    cu_seqlens=cu_seqlens,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    sp_per_req=sp_per_req,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    max_seqlen_q=max_seqlen_q,
                )
        with record_function_range("dsv4.fp8.meta.hca.workspace"):
            workspace_meta = self._build_workspace_meta(
                seqlen,
                sp_int,
                device,
                with_dense_cmp_topk=True,
                use_varlen=use_varlen,
                batch_size=batch_size,
                cu_seqlens=cu_seqlens,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sp_per_req=sp_per_req,
                position_ids=position_ids,
                req_id_per_token=req_id_per_token,
                max_seqlen_q=max_seqlen_q,
            )
        return HcaPrefillMeta(
            compressor_meta=compressor_meta,
            workspace_meta=workspace_meta,
        )

    def _build_workspace_meta(
        self,
        seqlen: int,
        sp_int: int,
        device: torch.device,
        with_dense_cmp_topk: bool,
        *,
        use_varlen: bool,
        batch_size: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        sp_per_req: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
    ) -> Optional[WorkspaceMeta]:
        """Static index/dim metadata for the vLLM-style workspace + dual-
        gather + ``combine_topk_swa_indices`` flow. Returns ``None`` when
        pool context isn't bound (warmup) so callers fall through to BF16
        ``kv_full`` fast paths.

        Two branches share field semantics (see :class:`WorkspaceMeta`):

          * **varlen** (``_use_varlen_prefill() and batch_size > 1``):
            per-request ``N_b`` / ``gather_b`` / ``M_b`` derived from
            ``prefix_lengths`` + ``input_lengths`` + ``cu_seqlens`` +
            ``position_ids`` + ``req_id_per_token``. One ``.item()`` sync
            per forward via a single ``torch.stack([N_max, gather_max])``
            ⇒ ``.tolist()``. ``new_k_slot_in_flat`` per-token target into
            ``workspace.view(B*M, D)`` for the BF16 overlay.
          * **legacy B==1 / DSV4_VARLEN_PREFILL=0**: ``N`` / ``gather_len``
            collapse to scalar arithmetic from ``sp_int`` + ``seqlen``.
            ``new_k_slot_in_flat = arange(seqlen) + (N + P)`` is bit-equal
            to the legacy slice ``copy_`` over ``[N+P, N+P+S)``.

        ``with_dense_cmp_topk=True`` precomputes the dense ``arange(N_max)``
        topk grid HCA needs (``[T_total, N_max]`` int32 contiguous; the
        ``_combine_topk_swa_indices_kernel`` masks per-token validity via
        ``COMPRESS_RATIO`` so no per-token mask is required up here).
        CSA passes ``False`` and feeds runtime indexer output to combine_topk.
        """
        from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, HCA_KV, SWA_KV

        if self._kv_cache is None or self._block_tables_by_type is None:
            return None
        ratio = self.compress_ratio
        if ratio not in (4, 128):
            return None
        cmp_at = CSA_KV if ratio == 4 else HCA_KV

        swa_bt = self._block_tables_by_type.get(SWA_KV)
        cmp_bt = self._block_tables_by_type.get(cmp_at)
        if (
            swa_bt is None
            or swa_bt.numel() == 0
            or cmp_bt is None
            or cmp_bt.numel() == 0
        ):
            return None
        swa_eb = self._pool_entries_per_block(SWA_KV)
        cmp_eb = self._pool_entries_per_block(cmp_at)
        if swa_eb <= 0 or cmp_eb <= 0:
            return None

        win = self.window_size
        # ``use_varlen`` is required — set by ``_build_shared_prefill_meta``
        # (the single env-read point + contract guard for the whole prefill
        # stack). UT helpers must pass it explicitly.

        if use_varlen:
            assert cu_seqlens is not None
            assert input_lengths is not None
            assert prefix_lengths is not None
            assert position_ids is not None
            assert req_id_per_token is not None
            cu_seqlens = _flat_1d(cu_seqlens)
            input_lengths = _flat_1d(input_lengths)
            prefix_lengths = _flat_1d(prefix_lengths)
            position_ids = _flat_1d(position_ids)
            req_id_per_token = _flat_1d(req_id_per_token)
            assert (
                position_ids.numel() == seqlen
            ), f"position_ids must be flat [T_total={seqlen}], got {position_ids.shape}"
            assert (
                req_id_per_token.numel() == seqlen
            ), f"req_id_per_token must be flat [T_total={seqlen}], got {req_id_per_token.shape}"
            assert (
                cu_seqlens.numel() == batch_size + 1
            ), f"cu_seqlens must be [B+1={batch_size + 1}], got {cu_seqlens.shape}"
            assert (
                input_lengths.numel() == batch_size
            ), f"input_lengths must be [B={batch_size}], got {input_lengths.shape}"
            assert (
                prefix_lengths.numel() == batch_size
            ), f"prefix_lengths must be [B={batch_size}], got {prefix_lengths.shape}"
            B = batch_size
            sp_i32 = prefix_lengths.to(device=device, dtype=torch.int32)
            S_i32 = input_lengths.to(device=device, dtype=torch.int32)

            # CP awareness:
            # Under CP both pools (compressor + SWA) hold the FULL gathered
            # sequence, and ``qkv.kv_full`` consumed by the BF16 overlay
            # has been all-gathered into ``[seq_len_full, D]``. So for
            # workspace sizing + per-token slot mapping we must use the
            # GLOBAL per-request lengths (``cp_ctx.input_lengths_global``)
            # plus a synthesised ``[seq_len_full]`` global position /
            # req_id stream. ``cu_seqlens`` (qsl) stays rank-local because
            # the kernel form of ``combine_topk_swa_indices`` is replaced
            # by the CP combine path under attention, which consumes explicit
            # ``cp_ctx.global_positions`` directly.
            cp_ctx_local = getattr(self, "_cp_ctx", None)
            cp_active = cp_ctx_local is not None and cp_ctx_local.cp_size > 1
            if cp_active:
                # B>=1 multi-request supported. Pools (compressor + SWA) hold
                # the FULL gathered sequence and ``qkv.kv_full`` is the
                # all-gathered ``[seq_len_full, D]`` tensor.
                # ``cp_ctx.input_lengths_global`` is the per-request global
                # length array (B entries). We
                # synthesise a ``[seq_len_full]`` global per-token stream
                # of (position_ids, req_id_per_token) by bucketising
                # against per-request cumulative starts so the existing
                # ``new_k_slot_in_flat`` formula
                # ``req*M + N + P_req + (pos - sp_req)`` lands each global
                # token in workspace[req_id]'s SWA tail correctly. For
                # B==1 this collapses to the previous arange + sp_global.
                assert (
                    cp_ctx_local.input_lengths_global is not None
                ), "CP workspace meta requires cp_ctx.input_lengths_global"
                S_i32 = cp_ctx_local.input_lengths_global.to(
                    device=device, dtype=torch.int32
                )
                B = int(S_i32.shape[0])
                seq_len_full = int(cp_ctx_local.seq_len_full)
                cum_after = torch.cumsum(S_i32, 0).to(torch.int32)  # [B]
                cum_starts = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32, device=device),
                        cum_after[:-1],
                    ]
                )  # [B] each req's start in [0, seq_len_full)
                g_arange32 = torch.arange(
                    seq_len_full, device=device, dtype=torch.int32
                )
                # req_id[g] = searchsorted(cum_after, g, right=True).
                # torch.bucketize(input, boundaries, right=True) returns the
                # count of cumulative ends <= g for ascending boundaries.
                req_id_per_token_eff = torch.bucketize(
                    g_arange32, cum_after, right=True
                ).to(torch.int64)
                # Clamp in case of float rounding edge: every g < seq_len_full
                # must map to a valid req index in [0, B).
                req_id_per_token_eff.clamp_(max=B - 1)
                cum_starts_l64 = cum_starts.to(torch.int64)
                sp_l64_b = prefix_lengths.to(device=device, dtype=torch.long)
                position_ids_eff = (
                    g_arange32.to(torch.int64)
                    - cum_starts_l64.gather(0, req_id_per_token_eff)
                ) + sp_l64_b.gather(0, req_id_per_token_eff)
            else:
                position_ids_eff = _flat_1d(
                    position_ids.to(device=device, dtype=torch.int64)
                )
                req_id_per_token_eff = _flat_1d(
                    req_id_per_token.to(device=device, dtype=torch.int64)
                )

            seq_total_per_req = sp_i32 + S_i32  # [B]
            N_per_req = seq_total_per_req // ratio  # [B]
            P_per_req = torch.clamp_max(sp_i32, win - 1)  # [B]
            gather_len_per_req = S_i32 + P_per_req  # [B]

            # Single .item() sync — stack two scalars then one D2H tolist().
            maxes = torch.stack([N_per_req.max(), gather_len_per_req.max()])
            N_max, gather_len_max = (int(v) for v in maxes.tolist())
            N = N_max
            M = N_max + gather_len_max

            swa_seq_lens = seq_total_per_req.contiguous()
            cmp_seq_lens = N_per_req.contiguous()
            swa_gather_lens = gather_len_per_req.contiguous()
            swa_cache_seq_lens = sp_i32.contiguous()
            swa_cache_gather_lens = P_per_req.contiguous()
            qsl = cu_seqlens.to(device=device, dtype=torch.int32).contiguous()
            swa_bt_int32 = swa_bt[:B].to(device=device, dtype=torch.int32).contiguous()
            cmp_bt_int32 = cmp_bt[:B].to(device=device, dtype=torch.int32).contiguous()

            # Per-token scatter target — pre-baked elementwise on the
            # builder side so ``_attn_via_workspace`` is kernel-only.
            # Under CP these come from the synthesised global streams above
            # so the resulting ``new_k_slot_in_flat`` matches
            # ``qkv.kv_full.size(0) == seq_len_full``.
            sp_l64 = prefix_lengths.to(device=device, dtype=torch.long)
            req_l64 = req_id_per_token_eff
            pos_l64 = position_ids_eff
            P_per_req_l64 = P_per_req.to(torch.long)  # [B]
            new_k_slot_in_flat = (
                req_l64 * M
                + N
                + P_per_req_l64.gather(0, req_l64)
                + (pos_l64 - sp_l64.gather(0, req_l64))
            ).contiguous()

            T_total = seqlen
        else:
            # Legacy B==1 / DSV4_VARLEN_PREFILL=0 — bit-equal to pre-Phase-3.
            #
            # Legacy CP awareness (B==1 only, gated by ``self._cp_ctx``):
            # under CP both pools (compressor + SWA) hold the FULL gathered
            # sequence (Phase C / Phase F write the all-gather'd KV per
            # rank), so workspace sizing (N / gather_len / M) and the new
            # K BF16 overlay slot mapping must use ``seq_len_full`` rather
            # than the rank-local ``seqlen``. ``T_total`` stays rank-local
            # because Q rows are still local to this rank — ``dense_cmp_topk``
            # and combine_topk consume it per local Q row. ``qsl`` is also
            # rank-local because the kernel form is replaced by a Python
            # vectorized builder under CP (see ``_attn_via_workspace``).
            cp_ctx_local = getattr(self, "_cp_ctx", None)
            cp_active = cp_ctx_local is not None and cp_ctx_local.cp_size > 1
            if cp_active:
                assert (
                    batch_size == 1
                ), "legacy CP workspace path supports B=1; set DSV4_VARLEN_PREFILL=1 for B>1"
                seqlen_eff = int(cp_ctx_local.seq_len_full)
            else:
                seqlen_eff = seqlen

            B = 1
            seq_total = sp_int + seqlen_eff
            N = seq_total // ratio
            P = min(sp_int, win - 1)
            gather_len = seqlen_eff + P
            M = N + gather_len

            swa_bt_int32 = swa_bt[:B].to(device=device, dtype=torch.int32).contiguous()
            cmp_bt_int32 = cmp_bt[:B].to(device=device, dtype=torch.int32).contiguous()
            swa_seq_lens = torch.tensor([seq_total], device=device, dtype=torch.int32)
            cmp_seq_lens = torch.tensor([N], device=device, dtype=torch.int32)
            swa_gather_lens = torch.tensor(
                [gather_len], device=device, dtype=torch.int32
            )
            swa_cache_seq_lens = torch.tensor(
                [sp_int], device=device, dtype=torch.int32
            )
            swa_cache_gather_lens = torch.tensor([P], device=device, dtype=torch.int32)
            qsl = torch.tensor([0, seqlen], device=device, dtype=torch.int32)

            # ``new_k_slot_in_flat`` writes the BF16 overlay of freshly
            # computed K. Non-CP: K is rank-local (== input), size
            # ``seqlen``; CP: K has been all-gathered into ``qkv.kv_full``
            # by ``_prefill_compute_qkv`` and now spans the full sequence
            # ``seq_len_full``, so the scatter target is a contiguous
            # ``arange(seq_len_full)`` into the SWA region [N+P, N+P+seq_len_full).
            new_k_slot_in_flat = (
                torch.arange(seqlen_eff, device=device, dtype=torch.long) + (N + P)
            ).contiguous()

            T_total = seqlen

        dense_cmp_topk: Optional[torch.Tensor] = None
        if with_dense_cmp_topk:
            if N > 0:
                dense_cmp_topk = (
                    torch.arange(N, device=device, dtype=torch.int32)
                    .view(1, N)
                    .expand(T_total, N)
                    .contiguous()
                )
            else:
                dense_cmp_topk = torch.empty(
                    (T_total, 0), device=device, dtype=torch.int32
                )

        return WorkspaceMeta(
            M=M,
            N=N,
            swa_eb=swa_eb,
            cmp_eb=cmp_eb,
            swa_bt_int32=swa_bt_int32,
            cmp_bt_int32=cmp_bt_int32,
            swa_seq_lens=swa_seq_lens,
            cmp_seq_lens=cmp_seq_lens,
            swa_gather_lens=swa_gather_lens,
            swa_cache_seq_lens=swa_cache_seq_lens,
            swa_cache_gather_lens=swa_cache_gather_lens,
            qsl=qsl,
            dense_cmp_topk=dense_cmp_topk,
            new_k_slot_in_flat=new_k_slot_in_flat,
        )

    def _build_compressor_meta(
        self,
        seqlen: int,
        sp_int: int,
        device: torch.device,
        *,
        use_varlen: bool,
        batch_size: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        sp_per_req: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
    ):
        """Run the main compressor's ``prepare_metadata`` with its pool
        context temporarily bound. Returns ``CompressorMeta``. The pool
        context binding (``_set_compressor_pool_context``) reads CSA_KV/
        CSA_STATE for ratio=4 and HCA_KV/HCA_STATE for ratio=128 based
        on ``self.compress_ratio``.

        ``use_varlen`` is required — set by ``_build_shared_prefill_meta``
        (the single env-read point + contract guard for the whole prefill
        stack). UT helpers must pass it explicitly so a missing kwarg
        surfaces as ``TypeError`` instead of silently picking up the
        ambient env.
        """
        from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
            build_prepare_metadata_args,
        )

        cmp_args = build_prepare_metadata_args(
            use_varlen=use_varlen,
            device=device,
            sp_int=sp_int,
            seqlen=seqlen,
            position_ids=position_ids,
            req_id_per_token=req_id_per_token,
            seq_start_per_req=sp_per_req,
            cu_seqlens=cu_seqlens,
        )
        self._set_compressor_pool_context()
        try:
            return self.compressor.prepare_metadata(**cmp_args)
        finally:
            self._clear_compressor_pool_context()

    def _prefill_common_setup(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> PrefillMeta:
        """Return the (possibly upper-layer-injected) shared prefill meta.
        Standalone callers (no upper-layer broadcast) build it per-layer
        here, syncing ``positions[0]`` to int once inside the builder.
        """
        meta = self._prefill_meta_shared
        if meta is None:
            meta = self._build_shared_prefill_meta(x, positions)
        return meta

    def _build_swa_prefill_meta_varlen(
        self,
        *,
        seqlen: int,
        device: torch.device,
        any_cont: bool,
        batch_size: int,
        cu_seqlens: torch.Tensor,
        input_lengths: torch.Tensor,
        prefix_lengths: torch.Tensor,
        position_ids: torch.Tensor,
        req_id_per_token: torch.Tensor,
    ) -> SwaPrefillMeta:
        """Varlen path: B>=1, per-request tensor plumbing.

        Three return points, each constructing a complete ``SwaPrefillMeta``
        with explicit field values (no ``_replace`` chaining):

          1. **Warmup** (pool unbound): only ``topk_length_kv_full`` set
             (SWA-only); all pool / Group-1 / Group-2 fields are ``None``/0.
          2. **CSA/HCA layer** (pool bound, ``compress_ratio != 0``):
             Group-1 write meta only; Group-2 lives on ``WorkspaceMeta``.
          3. **SWA-only layer** (pool bound, ``compress_ratio == 0``):
             Full meta. ``cache_*`` / ``combined_*`` populated on
             continuation; ``None``/0 on all-cold.
        """
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_ops_triton as _swa_ops

        cu_seqlens = _flat_1d(cu_seqlens)
        input_lengths = _flat_1d(input_lengths)
        prefix_lengths = _flat_1d(prefix_lengths)
        position_ids = _flat_1d(position_ids)
        req_id_per_token = _flat_1d(req_id_per_token)
        assert (
            position_ids.numel() == seqlen
        ), f"position_ids must be flat [T_total={seqlen}], got {position_ids.shape}"
        assert (
            req_id_per_token.numel() == seqlen
        ), f"req_id_per_token must be flat [T_total={seqlen}], got {req_id_per_token.shape}"
        assert (
            cu_seqlens.numel() == batch_size + 1
        ), f"cu_seqlens must be [B+1={batch_size + 1}], got {cu_seqlens.shape}"
        assert (
            input_lengths.numel() == batch_size
        ), f"input_lengths must be [B={batch_size}], got {input_lengths.shape}"
        assert (
            prefix_lengths.numel() == batch_size
        ), f"prefix_lengths must be [B={batch_size}], got {prefix_lengths.shape}"

        win = self.window_size
        is_swa_only = self.compress_ratio == 0
        num_tokens = seqlen  # T_total — flat token axis

        # Cache-independent — needed on warmup for ALL layer types because
        # CSA/HCA layers fall back to ``_attn_fp8_swa_via_kv_full`` when
        # workspace_meta is None (pool unbound).
        topk_length_kv_full: Optional[torch.Tensor] = None
        if is_swa_only or self._kv_cache is None:
            sp_per_token = prefix_lengths.to(torch.int32).gather(
                0, req_id_per_token.to(torch.int64)
            )  # [T_total]
            local_pos = position_ids.to(torch.int32) - sp_per_token  # [T_total]
            topk_length_kv_full = torch.clamp(local_pos + 1, max=win)

        # Warmup short-circuit — pool not bound, no Group-1 / Group-2 to build.
        bt = (
            self._block_tables_by_type.get(SWA_KV)
            if self._block_tables_by_type is not None
            else None
        )
        eb = self._pool_entries_per_block(SWA_KV)
        if self._kv_cache is None or bt is None or bt.numel() == 0 or eb <= 0:
            return SwaPrefillMeta(
                slot_mapping=None,
                query_start_loc=None,
                combined_seq_lens=None,
                topk_length_kv_full=topk_length_kv_full,
                combined_gather_lens=None,
                combined_gather_len_max=0,
                M=0,
                cache_seq_lens=None,
                cache_gather_lens=None,
                prefix_len_max=0,
                combined_indices=None,
                combined_lens=None,
                slot_in_flat=None,
            )

        # Group-1 (every pool-bound FP8 layer): SWA pool write meta.
        #
        # CP-aware: under cp_size > 1 the framework's ``cu_seqlens`` /
        # ``input_lengths`` are rank-local (already split by ZigZag), but
        # ``_prefill_compute_qkv`` all-gathers KV to ``[seq_len_full]`` in
        # GLOBAL request order before this write meta is consumed. So the
        # slot_mapping we build here is sized for the global view — one
        # slot per gather'd token — using the global cu_seqlens /
        # input_lengths derived in ``build_cp_context``. Each rank still
        # writes the full segment to its own paged pool (decode does not
        # gather); pools end up bit-identical across ranks. CP=1 falls
        # back to the rank-local tensors and is bit-equal to the legacy
        # path.
        #
        # Group-2 (attention) meta further down keeps using the rank-local
        # ``cu_seqlens`` / ``input_lengths`` because Q stays rank-local
        # under CP — Phase D wires up the cu_seqlens "double-track" for
        # ``flash_mla_sparse_fwd``. So we build the write view as a
        # standalone trio (``write_*``) and keep the rank-local
        # ``query_start_loc`` / ``combined_seq_lens`` for downstream.
        B = batch_size
        cp_ctx = getattr(self, "_cp_ctx", None)
        cp_on_write = (
            cp_ctx is not None
            and cp_ctx.cp_size > 1
            and cp_ctx.cu_seqlens_global is not None
            and cp_ctx.input_lengths_global is not None
        )
        query_start_loc = cu_seqlens.to(device=device, dtype=torch.int32).contiguous()
        combined_seq_lens = (
            prefix_lengths.to(torch.int32) + input_lengths.to(torch.int32)
        ).contiguous()
        if cp_on_write:
            write_B = int(cp_ctx.input_lengths_global.numel())
            write_query_start_loc = _flat_1d(
                cp_ctx.cu_seqlens_global.to(device=device, dtype=torch.int32)
            ).contiguous()
            write_combined_seq_lens = (
                prefix_lengths.to(torch.int32)[:write_B]
                + _flat_1d(cp_ctx.input_lengths_global.to(torch.int32))
            ).contiguous()
            write_num_tokens = cp_ctx.seq_len_full
        else:
            write_B = B
            write_query_start_loc = query_start_loc
            write_combined_seq_lens = combined_seq_lens
            write_num_tokens = num_tokens
        bt_swa = bt[:write_B].to(device=device, dtype=torch.int32).contiguous()
        slot_mapping = _swa_ops.compute_swa_slot_mapping(
            block_table=bt_swa,
            query_start_loc=write_query_start_loc,
            seq_lens=write_combined_seq_lens,
            block_size=eb,
            num_tokens=write_num_tokens,
        )

        # CSA/HCA: Group-1 only. Their attention meta lives on workspace_meta.
        if not is_swa_only:
            return SwaPrefillMeta(
                slot_mapping=slot_mapping,
                query_start_loc=query_start_loc,
                combined_seq_lens=combined_seq_lens,
                topk_length_kv_full=None,
                combined_gather_lens=None,
                combined_gather_len_max=0,
                M=0,
                cache_seq_lens=None,
                cache_gather_lens=None,
                prefix_len_max=0,
                combined_indices=None,
                combined_lens=None,
                slot_in_flat=None,
            )

        # Group-2 (SWA-only attention meta).
        #
        # CP-aware sizing: under cp_on_write, the workspace and attention
        # are shaped for the GLOBAL gather'd new K (``seq_len_full`` rows
        # of fresh K plus prefix tail), since ``_prefill_compute_qkv``
        # all-gathers ``kv_full`` to ``[seq_len_full]`` and that's what
        # gets scattered into the workspace. ``combined_gather_lens`` /
        # ``M`` therefore use the global write trio above.
        # ``query_start_loc`` for the kernel call must match the seq_lens
        # view (the kernel derives ``query_len = qsl[b+1]-qsl[b]`` and
        # ``prefix_len = seq_len - query_len``), so we feed the write trio
        # consistently. Q stays rank-local at attention time — we rebuild
        # ``combined_indices`` / ``combined_lens`` for rank-local Q tokens
        # with explicit GLOBAL positions further below.
        combined_gather_lens = _swa_ops.compute_prefill_gather_lens(
            seq_lens=write_combined_seq_lens,
            query_start_loc=write_query_start_loc,
            num_prefills=write_B,
            num_decodes=0,
            window_size=win,
        )
        # Single .item() sync per forward — ``combined_gather_lens`` already
        # encodes ``input_lengths[b] + min(prefix_lengths[b], win-1)`` per
        # request; its max is exactly ``combined_gather_len_max``.
        combined_gather_len_max = int(combined_gather_lens.max().item())
        M = max(combined_gather_len_max, 1)

        # cache_* + combined_* only populated on continuation (via_concat).
        # ``prefix_len_max`` under varlen is only consumed as a boolean
        # dequant guard in ``_attn_fp8_swa_via_concat`` — the per-request
        # P_b math runs through ``cache_gather_lens``. Sentinel ``1`` on
        # continuation avoids an extra ``.item()`` sync.
        if any_cont:
            # cache_seq_lens / cache_gather_lens read the SWA prefix tail
            # from each rank's own paged pool (Phase C wrote the full
            # gather'd KV on every rank). prefix_lengths is rank-invariant
            # under CP, so the same per-req tensors work for CP=1 and CP>1.
            # Slice to ``write_B`` so CP B=1 (write_B=1) doesn't pull in
            # the full rank-local request count from a longer prefix tensor.
            cache_seq_lens = prefix_lengths.to(device=device, dtype=torch.int32)[
                :write_B
            ].contiguous()
            cache_gather_lens = (
                torch.clamp_max(prefix_lengths, win - 1)
                .to(device=device, dtype=torch.int32)[:write_B]
                .contiguous()
            )
            if cp_on_write:
                # CP path: build per-Q-token attention meta with explicit
                # rank-local CP positions. B>1 needs request offsets in the
                # flattened workspace; the generic non-CP Triton kernel
                # assumes contiguous Q and cannot derive those under zigzag CP.
                if not _use_varlen_prefill():
                    assert (
                        write_B == 1
                    ), "legacy CP via_concat path supports B=1; set DSV4_VARLEN_PREFILL=1 for B>1"

                topk_indices_empty = torch.empty(
                    (seqlen, 0), dtype=torch.int32, device=device
                )
                combined_indices, combined_lens = _swa_ops.combine_topk_swa_indices_cp(
                    topk_indices=topk_indices_empty,
                    global_positions=_flat_1d(cp_ctx.global_positions),
                    sp_int=int(prefix_lengths[0].item()),
                    window_size=win,
                    compress_ratio=1,
                    topk=0,
                    M=M,
                    N=0,
                    req_id_per_token=req_id_per_token,
                    prefix_lengths=prefix_lengths,
                )

                full_req_ids = torch.repeat_interleave(
                    torch.arange(write_B, device=device, dtype=torch.long),
                    _flat_1d(
                        cp_ctx.input_lengths_global.to(device=device, dtype=torch.long)
                    ),
                )
                full_prefix = prefix_lengths.to(device=device, dtype=torch.long)[
                    :write_B
                ]
                full_starts = cp_ctx.cu_seqlens_global.to(
                    device=device, dtype=torch.long
                )[:-1]
                g_arange = torch.arange(
                    cp_ctx.seq_len_full, device=device, dtype=torch.long
                )
                local_pos = g_arange - full_starts.gather(0, full_req_ids)
                P_b_full = torch.clamp_max(full_prefix, win - 1)
                slot_in_flat = (
                    full_req_ids * M + P_b_full.gather(0, full_req_ids) + local_pos
                ).contiguous()
            else:
                topk_indices_empty = torch.empty(
                    (num_tokens, 0), dtype=torch.int32, device=device
                )
                combined_indices, combined_lens = _swa_ops.combine_topk_swa_indices(
                    topk_indices=topk_indices_empty,
                    query_start_loc=query_start_loc,
                    seq_lens=combined_seq_lens,
                    gather_lens=combined_gather_lens,
                    window_size=win,
                    compress_ratio=1,
                    topk=0,
                    M=M,
                    N=0,
                )
                # Pre-bake the per-token scatter index for ``via_concat``
                # step-2. All inputs are layer-invariant (per-batch tensors
                # + window_size + M); building once here keeps the attn
                # helper free of casts / gathers / arith on every cont
                # layer.
                prefix_l64 = prefix_lengths.to(device=device, dtype=torch.long)
                req_id_l64 = req_id_per_token.to(device=device, dtype=torch.long)
                pos_l64 = position_ids.to(device=device, dtype=torch.long)
                P_b = torch.clamp_max(prefix_l64, win - 1)
                slot_in_flat = (
                    req_id_l64 * M
                    + P_b.gather(0, req_id_l64)
                    + (pos_l64 - prefix_l64.gather(0, req_id_l64))
                ).contiguous()
            prefix_len_max = 1
        else:
            cache_seq_lens = None
            cache_gather_lens = None
            combined_indices = None
            combined_lens = None
            slot_in_flat = None
            prefix_len_max = 0

        return SwaPrefillMeta(
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            combined_seq_lens=combined_seq_lens,
            topk_length_kv_full=topk_length_kv_full,
            combined_gather_lens=combined_gather_lens,
            combined_gather_len_max=combined_gather_len_max,
            M=M,
            cache_seq_lens=cache_seq_lens,
            cache_gather_lens=cache_gather_lens,
            prefix_len_max=prefix_len_max,
            combined_indices=combined_indices,
            combined_lens=combined_lens,
            slot_in_flat=slot_in_flat,
        )

    def _build_swa_prefill_meta_legacy(
        self,
        seqlen: int,
        sp_int: int,
        device: torch.device,
    ) -> SwaPrefillMeta:
        """Legacy B==1 scalar path — ``DSV4_VARLEN_PREFILL=0`` / CP fallback.

        Bit-equal to the pre-Phase-2 implementation. Same three return
        points as the varlen variant (warmup → CSA/HCA → SWA-only), each
        constructing ``SwaPrefillMeta`` directly with explicit fields so
        the two functions diff side-by-side. B==1 is enforced by the
        contract guard in ``_build_shared_prefill_meta``.
        """
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_ops_triton as _swa_ops

        win = self.window_size
        is_swa_only = self.compress_ratio == 0
        bsz = 1
        num_tokens = bsz * seqlen

        topk_length_kv_full: Optional[torch.Tensor] = None
        if is_swa_only or self._kv_cache is None:
            positions = torch.arange(num_tokens, device=device, dtype=torch.int32)
            topk_length_kv_full = torch.clamp(positions + 1, max=win)

        bt = (
            self._block_tables_by_type.get(SWA_KV)
            if self._block_tables_by_type is not None
            else None
        )
        eb = self._pool_entries_per_block(SWA_KV)
        if self._kv_cache is None or bt is None or bt.numel() == 0 or eb <= 0:
            return SwaPrefillMeta(
                slot_mapping=None,
                query_start_loc=None,
                combined_seq_lens=None,
                topk_length_kv_full=topk_length_kv_full,
                combined_gather_lens=None,
                combined_gather_len_max=0,
                M=0,
                cache_seq_lens=None,
                cache_gather_lens=None,
                prefix_len_max=0,
                combined_indices=None,
                combined_lens=None,
                slot_in_flat=None,
            )

        seq_total = sp_int + seqlen
        query_start_loc = torch.tensor(
            [0, num_tokens], device=device, dtype=torch.int32
        )
        combined_seq_lens = torch.tensor([seq_total], device=device, dtype=torch.int32)
        bt_swa = bt[:bsz].to(device=device, dtype=torch.int32).contiguous()
        slot_mapping = _swa_ops.compute_swa_slot_mapping(
            block_table=bt_swa,
            query_start_loc=query_start_loc,
            seq_lens=combined_seq_lens,
            block_size=eb,
            num_tokens=num_tokens,
        )

        if not is_swa_only:
            return SwaPrefillMeta(
                slot_mapping=slot_mapping,
                query_start_loc=query_start_loc,
                combined_seq_lens=combined_seq_lens,
                topk_length_kv_full=None,
                combined_gather_lens=None,
                combined_gather_len_max=0,
                M=0,
                cache_seq_lens=None,
                cache_gather_lens=None,
                prefix_len_max=0,
                combined_indices=None,
                combined_lens=None,
                slot_in_flat=None,
            )

        combined_gather_lens = _swa_ops.compute_prefill_gather_lens(
            seq_lens=combined_seq_lens,
            query_start_loc=query_start_loc,
            num_prefills=bsz,
            num_decodes=0,
            window_size=win,
        )
        combined_gather_len_max = seqlen + min(sp_int, win - 1)
        M = max(combined_gather_len_max, 1)

        if sp_int > 0:
            prefix_len = min(sp_int, win - 1)
            cache_seq_lens = torch.tensor([sp_int], device=device, dtype=torch.int32)
            cache_gather_lens = torch.tensor(
                [prefix_len], device=device, dtype=torch.int32
            )
            topk_indices_empty = torch.empty(
                (num_tokens, 0), dtype=torch.int32, device=device
            )
            combined_indices, combined_lens = _swa_ops.combine_topk_swa_indices(
                topk_indices=topk_indices_empty,
                query_start_loc=query_start_loc,
                seq_lens=combined_seq_lens,
                gather_lens=combined_gather_lens,
                window_size=win,
                compress_ratio=1,
                topk=0,
                M=M,
                N=0,
            )
            prefix_len_max = prefix_len
        else:
            cache_seq_lens = None
            cache_gather_lens = None
            combined_indices = None
            combined_lens = None
            prefix_len_max = 0

        return SwaPrefillMeta(
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            combined_seq_lens=combined_seq_lens,
            topk_length_kv_full=topk_length_kv_full,
            combined_gather_lens=combined_gather_lens,
            combined_gather_len_max=combined_gather_len_max,
            M=M,
            cache_seq_lens=cache_seq_lens,
            cache_gather_lens=cache_gather_lens,
            prefix_len_max=prefix_len_max,
            combined_indices=combined_indices,
            combined_lens=combined_lens,
            slot_in_flat=None,
        )

    def _prefill_compute_qkv(self, x: torch.Tensor, common: PrefillMeta) -> PrefillQKV:
        """Q/KV path — RMSNorm + LoRA Q + KV linears + fused RMSNorm-RoPE.

        Internally uses ``[1, T, ...]`` so ``fused_rmsnorm_rope`` sees the
        ``(B, S, …)`` layout it expects. Returned tensors keep the 3D
        shape because downstream pool/compressor helpers rely on it.
        """
        x_3d = x.unsqueeze(0)
        rd = common.rd
        # Q path
        with record_function_range("dsv4.fp8.attn.qkv.q_lora_a_norm"):
            qr = self._rmsnorm_weighted(
                self._lin(self.wq_a, x_3d), self.q_norm
            )  # [1, T, q_lora_rank]
        with record_function_range("dsv4.fp8.attn.qkv.q_lora_b_rope"):
            q = self._lin(self.wq_b, qr).unflatten(-1, (self.n_heads, self.head_dim))
            q = fused_rmsnorm_rope(q, None, common.freqs_cis, rd, eps=self.eps)

        # KV path (single MQA head) — rank-local under CP.
        with record_function_range("dsv4.fp8.attn.qkv.kv_proj_rope"):
            kv_in = self._lin(self.wkv, x_3d)
            kv = fused_rmsnorm_rope(
                kv_in, self.kv_norm, common.freqs_cis, rd, eps=self.eps
            )

        if common.cp_on:
            # Dispatch on _use_varlen_prefill: varlen (default) supports
            # B>=1 via the flat helper; legacy keeps the B==1 [B, T, *F]
            # path. Both produce [1, seq_len_full, head_dim] downstream.
            if _use_varlen_prefill():
                from rtp_llm.models_py.modules.dsv4.cp import cp_all_gather_full_varlen

                with record_function_range("dsv4.fp8.attn.qkv.cp_gather_varlen"):
                    kv_flat = kv.reshape(kv.size(0) * kv.size(1), *kv.shape[2:])
                    kv_full_flat = cp_all_gather_full_varlen(kv_flat, common.cp_ctx)
                    kv_full = kv_full_flat.unsqueeze(0)
            else:
                with record_function_range("dsv4.fp8.attn.qkv.cp_gather"):
                    kv_full = cp_all_gather_full(
                        kv.squeeze(0), common.cp_ctx
                    ).unsqueeze(0)
        else:
            kv_full = kv

        return PrefillQKV(
            qr=qr.squeeze(0),
            q=q.squeeze(0),
            kv_full=kv_full.squeeze(0),
        )

    # ------------------------------------------------------------------
    # FP8 SWA-only fast path: paged-tail write + concat-from-cache
    # ------------------------------------------------------------------
    def _prefill_write_swa_fp8_paged(
        self, common: PrefillMeta, kv_full: torch.Tensor
    ) -> None:
        """Single-launch quantize + insert into the FP8 SWA pool, using
        the paged-tail ``slot_mapping`` pre-built in ``common.swa_meta``.

        Independent of the legacy ring-write in ``_prefill_write_swa_to_pool``;
        the paged-tail formula matches ``DSV4_CACHE_LAYOUT.md §6`` and the
        decode-side dequant kernel. Segments outside the SWA pool's last
        ``2*eb`` capacity get ``slot=-1`` and are skipped. No-op on
        warmup (``swa_meta`` write fields are ``None``).
        """
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_kv_insert_triton as _ins

        meta = common.swa_meta
        if meta is None or meta.slot_mapping is None:
            return
        packed_3d = self._pool_view_3d_fp8(SWA_KV)
        if packed_3d is None:
            return

        k_bf16 = kv_full.reshape(-1, self.head_dim)
        if k_bf16.dtype != torch.bfloat16:
            k_bf16 = k_bf16.to(torch.bfloat16)
        if self.layer_id == 0 and os.environ.get("DSV4_SWA_WRITE_DBG") == "1":
            try:
                sm = meta.slot_mapping
                bt = (
                    self._block_tables_by_type.get(SWA_KV)
                    if self._block_tables_by_type is not None
                    else None
                )
                bt_numel = int(bt.numel()) if bt is not None else -1
                bt_shape = list(bt.shape) if bt is not None else None
                pool_total_slots = int(packed_3d.shape[0])
                sm_total = int(sm.numel())
                sm_valid = int((sm >= 0).sum().item())
                sm_max = int(sm.max().item()) if sm_total > 0 else -1
                sm_min = int(sm.min().item()) if sm_total > 0 else -1
                k_rows = int(k_bf16.shape[0])
                _cp_on = bool(getattr(common, "cp_on", False))
                _seq_full = (
                    int(common.cp_ctx.seq_len_full)
                    if _cp_on and common.cp_ctx is not None
                    else -1
                )
                print(
                    f"[DSV4_SWA_WRITE_DBG] cp_on={_cp_on} k_rows={k_rows} seq_len_full={_seq_full} "
                    f"slot_mapping_total={sm_total} valid={sm_valid} min={sm_min} max={sm_max} "
                    f"swa_bt_shape={bt_shape} swa_bt_numel={bt_numel} pool_total_slots={pool_total_slots}",
                    flush=True,
                )
            except Exception as _e:
                print(f"[DSV4_SWA_WRITE_DBG] failed: {_e}", flush=True)
        with record_function_range("dsv4.fp8.attn.swa.quant_insert"):
            _ins.quantize_and_insert_k_cache(k_bf16, packed_3d, meta.slot_mapping)

    def _attn_fp8_swa_via_kv_full(
        self,
        qkv: PrefillQKV,
        common: PrefillMeta,
    ) -> torch.Tensor:
        """sparse_fwd over BF16 ``kv_full`` directly — no FP8 round-trip.

        Used by:
          * cold prefill (``sp == 0``) — pool capacity (``2 * eb``)
            can't hold the full prefill anyway; attend over BF16 K
            we just computed.
          * warmup forward (``self._kv_cache is None``) — pool not yet
            allocated; needed for the framework's dry-run shape inference.

        Caller is responsible for pre-writing the new K to the FP8 SWA
        pool (via ``_prefill_write_swa_fp8_paged``) for future decode
        reads — write order doesn't matter here since this path doesn't
        read from the pool.
        """
        from flash_mla import flash_mla_sparse_fwd  # type: ignore[import-not-found]

        meta = common.swa_meta
        assert meta is not None and meta.topk_length_kv_full is not None, (
            "FP8 SWA prefill requires common.swa_meta.topk_length_kv_full; "
            "built in _prefill_common_setup for FP8 SWA-only layers."
        )

        # ``topk_idxs`` shape:
        #   * varlen path → ``[T_total, win]`` flat-KV indices (per-request
        #     window, ``cu_seqlens[b] + local_pos`` baked in).
        #   * legacy / CP path → ``[1, T, win]``.
        # Either lands at flash_mla_sparse_fwd's ``[T, 1, win]`` indices
        # contract after the same ``squeeze + unsqueeze + cast`` chain.
        #
        # CP fresh prefill alignment: under cp_size > 1 ``qkv.q`` is
        # rank-local ``[T_local, H, D]`` and ``qkv.kv_full`` is the
        # all-gathered ``[seq_len_full, D]`` in GLOBAL request order. The
        # varlen topk builder receives CP global positions plus global
        # per-request cu_seqlens, so indices address the gathered KV for B>=1.
        if common.cp_on:
            # B>=1 is allowed under varlen (default DSV4_VARLEN_PREFILL=1).
            # Legacy DSV4_VARLEN_PREFILL=0 keeps the B==1 invariant.
            if not _use_varlen_prefill():
                assert (
                    common.cp_ctx is None or common.batch_size == 1
                ), "legacy CP attention path supports B=1; set DSV4_VARLEN_PREFILL=1 for B>1"
            assert (
                qkv.kv_full.size(0) == common.seqlen_full
            ), "CP gather should produce kv_full sized to seq_len_full"
        ti = common.topk_idxs
        if ti.dim() == 3:
            ti = ti.squeeze(0)
        indices = ti.unsqueeze(1).to(torch.int32)

        with record_function_range("dsv4.fp8.attn.swa.flash_mla_kv_full"):
            o3, _, _ = flash_mla_sparse_fwd(
                q=qkv.q,
                kv=qkv.kv_full.unsqueeze(1),
                indices=indices,
                sm_scale=self.softmax_scale,
                attn_sink=self.attn_sink,
                topk_length=meta.topk_length_kv_full,
            )
        return o3.unsqueeze(0)

    def _attn_fp8_swa_via_concat(
        self,
        qkv: PrefillQKV,
        common: PrefillMeta,
    ) -> torch.Tensor:
        """Continuation prefill (any req with prefix > 0): prefix-from-cache
        + new-K-bf16 concat. Varlen-aware ``[B, M, D]`` workspace.

        Pipeline (per request b, ``S_b = input_lengths[b]``,
        ``P_b = min(prefix_lengths[b], win-1)``):
          1. ``dequantize_and_gather_k_cache`` reads each request's trailing
             ``P_b`` cached tokens (SWA prefix tail, abs pos ``[sp_b - P_b, sp_b)``)
             into ``workspace[b, :P_b, :]``. Already batched via
             ``cache_seq_lens`` / ``cache_gather_lens`` ``[B]`` tensors.
          2. Vectorized scatter places the freshly computed new K into
             ``workspace[b, P_b:P_b+S_b, :]`` — no per-request Python loop.
          3. ``flash_mla_sparse_fwd`` over ``workspace.view(B*M, 1, D)``;
             ``combined_indices`` already carries ``M*batch_idx + slot``
             from ``combine_topk_swa_indices``.

        Legacy path (``DSV4_VARLEN_PREFILL=0`` / B==1 only): falls back to
        ``[1, M, D]`` workspace + scalar ``S`` / ``P`` slice. B==1 under
        varlen collapses to bit-equal output (``B*M = M``, scatter writes
        the same flat slot range, indices are unchanged).
        """
        from flash_mla import flash_mla_sparse_fwd  # type: ignore[import-not-found]

        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_dequant_triton as _swa_dq

        meta = common.swa_meta
        assert (
            meta is not None
            and meta.combined_indices is not None
            and meta.cache_seq_lens is not None
            and meta.cache_gather_lens is not None
        ), (
            "via_concat path requires swa_meta with cache_* and combined_* "
            "fields (only built when any request has prefix > 0)"
        )

        packed_3d = self._pool_view_3d_fp8(SWA_KV)
        assert packed_3d is not None, "FP8 SWA pool unavailable"
        eb = int(packed_3d.shape[1])
        D = self.head_dim

        # ``common.use_varlen`` is set by ``_build_shared_prefill_meta``'s
        # single varlen-tensor presence assert. No re-checking needed here.
        use_varlen = common.use_varlen
        B = common.batch_size if use_varlen else 1
        bt = self._block_tables_by_type[SWA_KV][:B]
        bt_int32 = bt.to(device=qkv.q.device, dtype=torch.int32).contiguous()

        # zero is necessary to avoid potential NaN values broadcast from uninitialized memory
        workspace = torch.zeros(
            (B, meta.M, D),
            dtype=torch.bfloat16,
            device=qkv.q.device,
        )

        # 1. Prefix tail dequant. ``cache_*`` are per-request ``[B]`` tensors
        # under varlen, single-element under legacy — same kernel call.
        if meta.prefix_len_max > 0:
            with record_function_range("dsv4.fp8.attn.swa_concat.gather_prefix"):
                _swa_dq.dequantize_and_gather_k_cache(
                    out=workspace,
                    k_cache=packed_3d,
                    seq_lens=meta.cache_seq_lens,
                    gather_lens=meta.cache_gather_lens,
                    block_table=bt_int32,
                    block_size=eb,
                    offset=0,
                )

        # 2. New K BF16 overlay.
        kv_bf16 = qkv.kv_full.to(torch.bfloat16)
        if use_varlen:
            # ``meta.slot_in_flat`` is pre-baked in ``_build_swa_prefill_meta_varlen``
            # (= ``req_id * M + min(prefix, win-1) + local_pos``). Single
            # ``index_copy_`` here — no per-layer casts / gathers / arith.
            assert (
                meta.slot_in_flat is not None
            ), "via_concat varlen path expects pre-baked slot_in_flat in swa_meta"
            with record_function_range("dsv4.fp8.attn.swa_concat.overlay_new_k"):
                workspace.view(B * meta.M, D).index_copy_(0, meta.slot_in_flat, kv_bf16)
        else:
            # Legacy B==1 single slice — bit-equal to the original code.
            S = common.seqlen
            P = meta.prefix_len_max
            with record_function_range("dsv4.fp8.attn.swa_concat.overlay_new_k"):
                workspace[:, P : P + S, :].copy_(kv_bf16.unsqueeze(0))

        # 3. flash_mla_sparse_fwd over the [B*M] flat KV view.
        with record_function_range("dsv4.fp8.attn.swa_concat.flash_mla"):
            o3, _, _ = flash_mla_sparse_fwd(
                q=qkv.q,
                kv=workspace.view(B * meta.M, 1, D),
                indices=meta.combined_indices.unsqueeze(1),
                sm_scale=self.softmax_scale,
                attn_sink=self.attn_sink,
                topk_length=meta.combined_lens,
            )
        return o3.unsqueeze(0)

    # ------------------------------------------------------------------
    # Output projection: inv-RoPE + wo_a + wo_b + tp all-reduce
    # ------------------------------------------------------------------
    def _prefill_output_proj(
        self, o: torch.Tensor, common: PrefillMeta
    ) -> torch.Tensor:
        """Inverse-RoPE + grouped wo_a + wo_b + (TP) all-reduce.

        Fast path (CUDA, non-empty): single fused Triton kernel
        (``fused_inv_rope_fp8_quant``) emits the exact ``(fp8 [M,G,K],
        scale [M,G,K/512])`` layout ``deep_gemm.fp8_einsum`` consumes,
        collapsing what used to be apply_rotary_emb + per-group
        per_token_group_quant_fp8 + einsum into one launch.

        Eager fallback (CPU / empty): explicit inv-rotary + bf16-dequant +
        ``einsum("bsgd,grd->bsgr")``, kept for warmup / unit tests.

        Output shape ``[1, T, dim]`` (caller squeezes to ``[T, dim]``).
        """
        rd = common.rd
        seqlen = common.seqlen
        freqs_cis = common.freqs_cis

        if o.is_cuda and o.numel() > 0:
            chunk_tokens = int(os.environ.get("DSV4_ATTN_OUT_CHUNK_TOKENS", "65536"))
            if chunk_tokens > 0 and seqlen > chunk_tokens:
                if freqs_cis.dim() == 2:
                    freqs_all = freqs_cis
                else:
                    freqs_all = freqs_cis.reshape(seqlen, -1)
                out = torch.empty(
                    1, seqlen, self.dim, dtype=torch.bfloat16, device=o.device
                )
                out_2d = out.view(seqlen, self.dim)
                o_tokens = o.reshape(seqlen, self.n_heads * self.head_dim)
                for token_start in range(0, seqlen, chunk_tokens):
                    token_end = min(token_start + chunk_tokens, seqlen)
                    chunk_len = token_end - token_start
                    o_3d = (
                        o_tokens[token_start:token_end]
                        .reshape(chunk_len, self.n_heads, self.head_dim)
                        .contiguous()
                    )
                    freqs_per_token = freqs_all[token_start:token_end].contiguous()
                    with record_function_range(
                        "dsv4.fp8.attn.out.fused_inv_rope_quant"
                    ):
                        o_fp8, o_scale = fused_inv_rope_fp8_quant(
                            o_3d,
                            freqs_per_token,
                            n_groups=self.n_groups,
                            heads_per_group=self.n_heads // self.n_groups,
                            nope_dim=self.head_dim - self.rope_head_dim,
                            rope_head_dim=self.rope_head_dim,
                        )
                    with record_function_range("dsv4.fp8.attn.out.wo_a_einsum"):
                        o_chunk = self._wo_a_einsum_from_fp8(
                            o_fp8, o_scale, 1, chunk_len
                        )
                    with record_function_range("dsv4.fp8.attn.out.wo_b"):
                        out_2d[token_start:token_end].copy_(
                            self.wo_b(o_chunk.flatten(2).reshape(chunk_len, -1))
                        )
                if self.tp_size > 1:
                    from rtp_llm.models_py.distributed.collective_torch import (
                        Group,
                        all_reduce,
                    )

                    with record_function_range("dsv4.fp8.attn.out.tp_all_reduce"):
                        all_reduce(out, Group.TP)
                return out

            o_3d = o.reshape(seqlen, self.n_heads, self.head_dim)
            if freqs_cis.dim() == 2:
                freqs_per_token = freqs_cis.contiguous()
            else:
                freqs_per_token = freqs_cis.reshape(seqlen, -1).contiguous()
            with record_function_range("dsv4.fp8.attn.out.fused_inv_rope_quant"):
                o_fp8, o_scale = fused_inv_rope_fp8_quant(
                    o_3d,
                    freqs_per_token,
                    n_groups=self.n_groups,
                    heads_per_group=self.n_heads // self.n_groups,
                    nope_dim=self.head_dim - self.rope_head_dim,
                    rope_head_dim=self.rope_head_dim,
                )
            with record_function_range("dsv4.fp8.attn.out.wo_a_einsum"):
                o = self._wo_a_einsum_from_fp8(o_fp8, o_scale, 1, seqlen)
        else:
            with record_function_range("dsv4.fp8.attn.out.eager_inv_rope_wo_a"):
                apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
                o = o.reshape(1, seqlen, self.n_groups, -1)
                wo_a_bf16 = _fp8_dequant_to_fp32(self.wo_a_w, self.wo_a_s).to(o.dtype)
                wo_a = wo_a_bf16.view(self.n_groups, self.o_lora_rank, -1)
                o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        with record_function_range("dsv4.fp8.attn.out.wo_b"):
            out = self._lin(self.wo_b, o.flatten(2))
        if self.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            with record_function_range("dsv4.fp8.attn.out.tp_all_reduce"):
                all_reduce(out, Group.TP)
        return out
