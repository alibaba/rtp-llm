"""DeepSeek-V4 Attention — vLLM-flow variant (BF16 KV pool throughout).

Standalone parallel of :class:`Attention` in ``attention.py``. Same
public surface and same constructor args, but constructs
:class:`CompressorVLLM` / :class:`IndexerVLLM` for the nested modules
and adds the ``_forward_prefill_vllm`` family + dispatch hook in
``_forward_body``. Selected at construction time by ``block.py`` when
``DSV4_BF16_VLLM=1`` (see ``attention.DSV4_BF16_VLLM``); legacy
``Attention`` is selected when off.

Per-token state-pool / fused-boundary writer + BF16 KV pool throughout —
no FP8 quant, no UE8M0 scales. Falls through to the legacy body for SWA-
only layers, batched prefills, decode, and any other path where the
vllm-flow prefill helpers don't apply.

Kept in lockstep with ``attention.py``: bug-fixes to shared helpers
(``_lin``, ``_pool_view``, ``_prefill_write_swa_to_pool``,
``_gather_kv_cache_dense_from_pool``, ``_wo_a_einsum_from_fp8``, …)
need to be applied to **both** files.
"""

import math
import os

# ---------------------------------------------------------------------------
# vLLM-flow prefill metadata bundles. Mirror of source's ``PrefillMeta`` /
# ``PrefillQKV`` (FP8 attention.py) but stripped to the BF16 essentials
# (no FP8 workspace, no fused varlen slot mapping). Consumed by
# ``Attention._forward_prefill_vllm`` and friends.
# ---------------------------------------------------------------------------
from typing import Any, Dict
from typing import NamedTuple as _NamedTuple
from typing import Optional, Union

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
from rtp_llm.models_py.modules.dsv4._metadata_triton import (
    build_cp_compress_topk_idxs,
    build_cp_window_topk_idxs,
    build_swa_pool_slot_mapping,
)
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.compressor_vllm import CompressorVLLM


class _VLLMPrefillCommon(_NamedTuple):
    bsz: int
    seqlen: int  # rank-local Q length (== chunk_length under CP)
    sp_int: int  # absolute prefix length
    end_pos: int  # = sp_int + seqlen (non-CP) / cp_ctx.seq_len_total (CP)
    is_fresh_prefill: bool  # = (sp_int == 0)
    device: torch.device
    freqs_cis: torch.Tensor  # rank-local per-Q freqs (cp_freqs_cis_local under CP)
    swa_dense_len: int  # SWA prefix length in the dense kv_cat layout
    # hoisted meta — built once per layer-call, reused across compressor +
    # nested indexer compressor:
    compressor_meta: Any  # CompressorMeta from compressor_vllm
    indexer_meta: Optional[Any]  # _IndexerVLLMPrefillMeta or None (HCA)
    # CP context — None / cp_size <= 1 means single-rank fast path.  When
    # active, callers must:
    #   * all-gather rank-local KV before SWA pool write / sparse_attn read
    #   * use ``cp_ctx.global_positions`` for window/compress topk indices
    #   * size the dense kv_cat as ``[seq_len_total + seq_len_total/ratio]``
    cp_ctx: Optional["CPContext"]
    cp_on: bool


class _VLLMPrefillQKV(_NamedTuple):
    q: torch.Tensor  # [B, S, n_heads, head_dim] bf16 — rank-local under CP
    qr: torch.Tensor  # [B, S, q_lora_rank] bf16 — rank-local under CP
    kv_full: torch.Tensor  # [B, T, head_dim] bf16 — all-gathered global under CP


from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    cp_all_gather_full_async,
    cp_freqs_cis_local,
    cp_wait_gather_full,
)
from rtp_llm.models_py.modules.dsv4.indexer_vllm import IndexerVLLM
from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear, _fp8_dequant_to_fp32
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
    precompute_freqs_cis,
)
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


_V4_FP8_BLOCK_CFG = Fp8BlockWiseQuantConfig()


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


# V4 author's TileLang sparse_attn kernel — vendored from
# /mnt/nas1/hf/DeepSeek-V4-Flash/inference/kernel.py:sparse_attn_kernel.
# V4 is MQA + Q-LoRA (NOT MLA); this is the author-authored kernel for
# exactly this math. Falls back to the PyTorch reference `_sparse_attn`
# when tilelang is unavailable (e.g. environments where libstdc++
# symbols don't match tilelang's pre-built libtvm.so).
from rtp_llm.models_py.modules.dsv4 import tilelang_kernels as _tl_kernels


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
    if global_positions.is_cuda:
        with record_function_range("dsv4.attn.cp_window_topk_meta"):
            return build_cp_window_topk_idxs(
                global_positions,
                bsz=bsz,
                seq_len_total=seq_len_total,
                window_size=window_size,
            )

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


def _get_compress_topk_idxs(
    ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int, device
) -> torch.Tensor:
    end_pos = start_pos + seqlen
    T_comp = end_pos // ratio
    if T_comp == 0:
        return torch.full((bsz, seqlen, 0), -1, dtype=torch.long, device=device)
    cols = torch.arange(T_comp, device=device)
    global_pos = torch.arange(start_pos, end_pos, device=device).unsqueeze(1)
    allowed = (global_pos + 1) // ratio
    matrix = torch.where(cols.unsqueeze(0) < allowed, cols.unsqueeze(0) + offset, -1)
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _get_window_topk_idxs_batched(
    window_size: int,
    max_seqlen: int,
    sp_tensor: torch.Tensor,  # [B] int64
    row_seqlens: torch.Tensor,  # [B] int64 — valid new-token count per row
    device,
) -> torch.Tensor:
    """Batched variant of :func:`_get_window_topk_idxs` for heterogeneous
    (sp, seqlen) prefill.  Returns ``[B, max_seqlen, window_size]`` int64
    with linear absolute KV indices.

    For each row ``b`` and local query position ``i < row_seqlens[b]``:
      global pos ``g = sp_tensor[b] + i``
      returns ``[max(0, g-window_size+1), g]``; remaining columns filled
      with ``-1``.

    Rows with ``i >= row_seqlens[b]`` (padding) get an entire row of
    ``-1``, so the sparse_attn kernel contributes nothing for them.
    """
    bsz = int(sp_tensor.shape[0])
    if bsz == 0 or max_seqlen == 0 or window_size == 0:
        return torch.full(
            (bsz, max_seqlen, window_size), -1, dtype=torch.long, device=device
        )
    sp_t = sp_tensor.to(device=device, dtype=torch.long)
    seq_t = row_seqlens.to(device=device, dtype=torch.long)
    i = torch.arange(max_seqlen, device=device, dtype=torch.long)  # [S]
    global_pos = sp_t.unsqueeze(1) + i.unsqueeze(0)  # [B, S]
    offsets = torch.arange(window_size, device=device, dtype=torch.long)  # [W]
    window_start = (global_pos - window_size + 1).clamp_min(0)
    idxs = window_start.unsqueeze(-1) + offsets.view(1, 1, -1)
    matrix = torch.where(idxs <= global_pos.unsqueeze(-1), idxs, -1)
    # Padding rows (i >= row_seqlens[b]) -> whole row -1.
    row_mask = i.unsqueeze(0) < seq_t.unsqueeze(1)  # [B, S]
    matrix = torch.where(row_mask.unsqueeze(-1), matrix, torch.full_like(matrix, -1))
    return matrix.contiguous()


def _get_compress_topk_idxs_batched(
    ratio: int,
    max_seqlen: int,
    offset: int,
    sp_tensor: torch.Tensor,  # [B] int64
    row_seqlens: torch.Tensor,  # [B] int64
    device,
) -> torch.Tensor:
    """Batched variant of :func:`_get_compress_topk_idxs`.

    Returns ``[B, max_seqlen, K]`` where ``K`` is bounded by
    ``max(sp + seqlen) // ratio`` across the batch.  For each row ``b``
    with ``g = sp_tensor[b] + i`` where ``i < row_seqlens[b]``:
      * ``sp_tensor[b] > 0``: query at global position ``sp+i`` sees
        compressed blocks ``[0, (sp+i+1)//ratio)``.
      * ``sp_tensor[b] == 0``: row sees compressed blocks ``[0, (i+1)//ratio)``
        at query ``i`` (causal over compressed KV).

    Rows with ``i >= row_seqlens[b]`` (padding) emit all ``-1``.
    """
    bsz = int(sp_tensor.shape[0])
    if bsz == 0 or ratio == 0:
        return torch.full((bsz, max_seqlen, 0), -1, dtype=torch.long, device=device)
    sp_t = sp_tensor.to(device=device, dtype=torch.long)
    seq_t = row_seqlens.to(device=device, dtype=torch.long)
    end_per_row = (sp_t + seq_t) // ratio  # [B]
    K = int(end_per_row.max().item()) if bsz > 0 else 0
    if K == 0:
        return torch.full((bsz, max_seqlen, 0), -1, dtype=torch.long, device=device)
    k = torch.arange(K, device=device, dtype=torch.long)  # [K]
    i = torch.arange(max_seqlen, device=device, dtype=torch.long)  # [S]

    global_pos = sp_t.unsqueeze(1) + i.unsqueeze(0)  # [B, S]
    max_allowed_bi = (global_pos + 1) // ratio  # [B, S]
    k_b_i = k.view(1, 1, K).expand(bsz, max_seqlen, K)
    valid_k = k_b_i < max_allowed_bi.unsqueeze(-1)  # [B, S, K]
    matrix = torch.where(valid_k, k_b_i + offset, torch.full_like(k_b_i, -1))
    # Padding rows -> all -1.
    row_mask = i.unsqueeze(0) < seq_t.unsqueeze(1)  # [B, S]
    matrix = torch.where(row_mask.unsqueeze(-1), matrix, torch.full_like(matrix, -1))
    return matrix.contiguous()


def _get_compress_topk_idxs_cp(
    ratio: int,
    bsz: int,
    seq_len_total: int,
    offset: int,
    global_positions: torch.Tensor,
) -> torch.Tensor:
    """CP-prefill variant of the dense HCA compressed-KV index list
    (the branch used when no Indexer is present — compress_ratio == 128).
    Q at GLOBAL position g reads compressed KV blocks [0, (g+1)//ratio),
    which live at offsets [offset, offset + (g+1)//ratio) inside the
    attention-side concatenated [sliding | compressed] tensor.  Return
    shape [bsz, S_local, seq_len_total // ratio]."""
    device = global_positions.device
    S_local = int(global_positions.shape[0])
    T_comp = max(seq_len_total // ratio, 0)
    if T_comp == 0:
        return torch.full((bsz, S_local, 0), -1, dtype=torch.long, device=device)
    if global_positions.is_cuda:
        with record_function_range("dsv4.attn.cp_compress_topk_meta"):
            return build_cp_compress_topk_idxs(
                global_positions,
                bsz=bsz,
                seq_len_total=seq_len_total,
                ratio=ratio,
                offset=offset,
            )
    cols = torch.arange(T_comp, device=device)  # [T_comp]
    max_allowed = (global_positions + 1) // ratio  # [S_local]
    mask = cols.unsqueeze(0) >= max_allowed.unsqueeze(1)  # [S_local, T_comp]
    matrix = torch.where(
        mask,
        torch.full_like(cols, -1).expand(S_local, -1),
        cols.expand(S_local, -1) + offset,
    )
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _sparse_attn(
    q: torch.Tensor,  # [B, S, H, D]
    kv: torch.Tensor,  # [B, T_kv, D]   (single KV head, shared across H)
    sink: torch.Tensor,  # [H]   FP32 logit added to softmax denom (per-head sink)
    topk_idxs: torch.Tensor,  # [B, S, K] long; -1 entries are masked out
    softmax_scale: float,
) -> torch.Tensor:
    """Reference PyTorch sparse attention with attention sink.

    Output: [B, S, H, D]
    """
    bsz, seqlen, n_heads, head_dim = q.size()
    K = topk_idxs.size(-1)
    valid = topk_idxs >= 0  # [B, S, K]
    safe_idxs = topk_idxs.clamp_min(0)  # [B, S, K]

    # gather selected KV: [B, S, K, D]
    idx_expanded = safe_idxs.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    kv_exp = kv.unsqueeze(1).expand(-1, seqlen, -1, -1)  # [B, S, T_kv, D]
    selected = torch.gather(kv_exp, 2, idx_expanded)  # [B, S, K, D]

    # logits: [B, S, H, K] = einsum(qhd, kd)
    q_f = q.float()
    selected_f = selected.float()
    logits = torch.einsum("bshd,bskd->bshk", q_f, selected_f) * softmax_scale
    # mask invalid slots
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))

    # Softmax with attn_sink — matches official `sparse_attn_kernel`:
    #   scores_max = max over logits only (NOT including sink)
    #   exp_logits = exp(logits - scores_max)
    #   acc_o = Σ exp_logits · v
    #   sum_exp = Σ exp_logits + exp(sink - scores_max)
    #   out = acc_o / sum_exp
    # Note: we do NOT include sink in `scores_max`, and the numerator has no sink term.
    scores_max = logits.amax(dim=-1, keepdim=True).clamp_min(-1e30)  # [B, S, H, 1]
    exp_logits = torch.exp(logits - scores_max)  # [B, S, H, K]
    sink_logit = sink.view(1, 1, n_heads, 1).expand_as(scores_max)
    exp_sink = torch.exp(sink_logit - scores_max)  # [B, S, H, 1]
    sum_exp = exp_logits.sum(dim=-1, keepdim=True) + exp_sink  # [B, S, H, 1]

    # acc_o = Σ_k exp_logits[k] · selected[k]
    acc_o = torch.einsum("bshk,bskd->bshd", exp_logits, selected_f)  # [B, S, H, D]
    out = acc_o / sum_exp  # divide each head by its denom
    return out.to(q.dtype)


class AttentionVLLM(nn.Module):
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
            self.compressor = CompressorVLLM(
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
                self.indexer = IndexerVLLM(
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

        # CP context bound per-forward by V4Transformer.  None = no CP.
        self._cp_ctx: Optional[CPContext] = None

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
        self._pool_spec: Dict[int, tuple] = {
            SWA_KV: (torch.bfloat16, head_dim),
            CSA_KV: (torch.bfloat16, head_dim),
            HCA_KV: (torch.bfloat16, head_dim),
            INDEXER_KV: (torch.bfloat16, idx_hd),
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
        return raw_u8[:, :useful_bytes].view(vec_dtype).view(-1, vec_dim)

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
        from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
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
        bt_long = bt[:bsz].to(device=device, dtype=torch.long)
        block_id = bt_long.gather(
            1, block_in_seq.unsqueeze(0).expand(bsz, -1)
        )  # [B, T]
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
        from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
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
        block_id = bt_long.gather(1, safe_block.unsqueeze(0).expand(bsz, -1))
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
        from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        if self._kv_cache is None or self._block_tables_by_type is None:
            return
        bt = self._block_tables_by_type.get(SWA_KV)
        if bt is None or bt.numel() == 0:
            return
        pool_view = self._pool_view(SWA_KV)
        eb = self._pool_entries_per_block(SWA_KV)
        if pool_view is None or eb <= 0:
            return

        max_seqlen = int(kv_full.shape[1])
        if max_seqlen == 0:
            return
        device = kv_full.device
        max_blocks = bt.shape[1]

        # Write every token from this prefill that maps to a currently
        # allocated SWA block-table entry.  Attention itself only consumes a
        # sliding window, but prefix-cache reuse may stop at an earlier block
        # boundary than this request's final token.  Those boundary states need
        # the full request tail physical blocks populated, not just the final
        # ``window_size`` rows.
        max_n_write = max_seqlen
        if max_n_write == 0:
            return

        with record_function_range("dsv4.attn.swa_pool_slot_meta"):
            slot_mapping = build_swa_pool_slot_mapping(
                bt[:bsz],
                bsz=bsz,
                T=max_n_write,
                eb=eb,
                sp=sp,
                row_seqlens=row_seqlens,
            )

        if slot_mapping is None:
            # CPU/reference fallback.
            if isinstance(sp, torch.Tensor):
                sp_t = sp.to(device=device, dtype=torch.long)
                if sp_t.dim() == 0:
                    sp_t = sp_t.unsqueeze(0)
                if sp_t.numel() == 1 and bsz > 1:
                    sp_t = sp_t.expand(bsz)
            else:
                sp_t = torch.full((bsz,), int(sp), device=device, dtype=torch.long)

            if row_seqlens is None:
                seq_t = torch.full((bsz,), max_seqlen, device=device, dtype=torch.long)
            elif isinstance(row_seqlens, torch.Tensor):
                seq_t = row_seqlens.to(device=device, dtype=torch.long)
                if seq_t.dim() == 0:
                    seq_t = seq_t.unsqueeze(0)
                if seq_t.numel() == 1 and bsz > 1:
                    seq_t = seq_t.expand(bsz)
            else:
                seq_t = torch.full(
                    (bsz,), int(row_seqlens), device=device, dtype=torch.long
                )

            j = torch.arange(max_n_write, device=device, dtype=torch.long)
            row_valid = j.unsqueeze(0) < seq_t.unsqueeze(1)
            global_pos = sp_t.unsqueeze(1) + j.unsqueeze(0)
            block_in_seq = global_pos // eb
            in_block = global_pos % eb
            bt_long = bt[:bsz].to(device=device, dtype=torch.long)
            in_capacity = block_in_seq < max_blocks
            safe_in_seq = torch.where(
                in_capacity, block_in_seq, torch.zeros_like(block_in_seq)
            )
            block_id = bt_long.gather(1, safe_in_seq)
            valid = (block_id > 0) & in_capacity & row_valid
            slot = torch.where(
                valid, block_id * eb + in_block, torch.full_like(in_block, -1)
            )
            slot_mapping = slot.reshape(-1)

        # Source rows are already dense request-major: src[b, j] = kv_full[b, j].
        source = kv_full[:bsz, :max_n_write]
        source_flat = source.reshape(bsz * max_n_write, self.head_dim)
        write_kv_to_pool(source_flat, slot_mapping, pool_view, mask_negative=True)

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
        bt_long = bt[:bsz].to(device=device, dtype=torch.long)
        block_id = bt_long.gather(1, safe_block)
        valid = valid_pos & in_capacity & (block_id > 0)
        safe_slot = torch.where(
            valid, block_id * eb + in_block, torch.zeros_like(block_id)
        )

        from rtp_llm.models_py.modules.dsv4._pool_triton import masked_gather_from_pool

        return masked_gather_from_pool(
            pool_view,
            safe_slot,
            valid,
            out_shape=(bsz, win, self.head_dim),
            dtype=dtype,
        )

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
        block_id = bt_long.gather(1, safe_block.unsqueeze(0).expand(bsz, -1))
        valid = in_capacity_row.unsqueeze(0) & (block_id > 0)
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        if _rt.should_record_layer(self.layer_id):
            _rt.record_if_level(2, f"L{self.layer_id:02d}_swa_bt", bt_long)
            _rt.record_if_level(
                2,
                f"L{self.layer_id:02d}_swa_abs_block_id",
                block_id,
            )
            _rt.record_if_level(
                2,
                f"L{self.layer_id:02d}_swa_eb",
                torch.tensor(
                    [eb, self.window_size, dense_len], device=device, dtype=torch.int32
                ),
            )
        safe_slot = torch.where(
            valid, block_id * eb + in_block.unsqueeze(0), torch.zeros_like(block_id)
        )

        from rtp_llm.models_py.modules.dsv4._pool_triton import masked_gather_from_pool

        out = masked_gather_from_pool(
            pool_view,
            safe_slot,
            valid,
            out_shape=(bsz, dense_len, self.head_dim),
            dtype=dtype,
        )
        if current_kv_full is not None and current_kv_full.shape[1] > 0:
            local_pos = pos.unsqueeze(0) - sp_t.unsqueeze(1)  # [B, dense_len]
            max_current = int(current_kv_full.shape[1])
            overlay = (
                (local_pos >= 0)
                & (local_pos < seq_t.unsqueeze(1))
                & (local_pos < max_current)
            )
            safe_local = local_pos.clamp(min=0, max=max_current - 1)
            src = current_kv_full
            if src.dtype != dtype:
                src = src.to(dtype)
            src = torch.gather(
                src,
                1,
                safe_local.unsqueeze(-1).expand(-1, -1, self.head_dim),
            )
            out = torch.where(overlay.unsqueeze(-1), src, out)
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
            kv_view = self._pool_view(INDEXER_KV)
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
        pool_view = self._pool_view(attn_type)
        eb = self._pool_entries_per_block(attn_type)
        if pool_view is None or eb <= 0:
            return None
        max_blocks = bt.shape[1]
        pool_capacity = max_blocks * eb
        pos = torch.arange(T, device=device, dtype=torch.long)  # [T]
        in_capacity_row = pos < pool_capacity  # [T]
        safe_pos = torch.where(in_capacity_row, pos, torch.zeros_like(pos))
        block_in_seq = safe_pos // eb
        in_block = safe_pos % eb
        bt_long = bt[:bsz].to(device=device, dtype=torch.long)
        block_id = bt_long.gather(
            1, block_in_seq.unsqueeze(0).expand(bsz, -1)
        )  # [B, T]
        in_capacity = in_capacity_row.unsqueeze(0).expand(bsz, -1)  # [B, T]
        valid = (block_id > 0) & in_capacity
        safe_slot = torch.where(
            valid,
            block_id * eb + in_block.unsqueeze(0),
            torch.zeros_like(block_id),
        )  # [B, T]
        from rtp_llm.models_py.modules.dsv4._pool_triton import masked_gather_from_pool

        return masked_gather_from_pool(
            pool_view,
            safe_slot,
            valid,
            out_shape=(bsz, T, vec_dim),
            dtype=dtype,
        )

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
        attn_metadata: "DSv4DecodeAttnMetadata",  # type: ignore[name-defined]
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
        attn_metadata: "DSv4DecodeAttnMetadata",  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """Decode-only attention forward.

        Per-request batched: every request has its own ``start_pos`` from
        ``attn_metadata.start_pos[B]``. KV writes use the metadata's
        ``slot_mapping_swa`` / ``slot_mapping_compressed[ratio]`` indices
        into the (still register_buffer-backed) per-layer KV cache.

        Decode-only — does NOT touch the prefill ``forward`` arm. Phase
        4 will swap the TileLang sparse_attn for FlashMLA + FP8 KV here.
        """
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
        from rtp_llm.models_py.modules.dsv4.decode.sparse_attn_decode_op import (
            SparseAttnV4DecodeOp,
        )

        bsz, q_len, _ = x.size()
        assert q_len == 1, "Phase 2: q_len==1 only (MTP/spec-decode is later)"
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        device = x.device
        _dbg_decode = _rt.should_record_layer(self.layer_id)
        # Slice metadata to actual bsz — the CUDA-graph impl allocates
        # buffers at max_bs, but the captured graph at BS=k must read only
        # the [:k] prefix. Padding entries [k:max_bs] are stale across
        # replays; mixing them with k-sized index arrays in fancy
        # indexing (e.g. self.kv_state[b_idx[k], slot[max_bs]] = ...)
        # broadcasts the same row to multiple slots and corrupts state.
        start_pos = attn_metadata.start_pos[:bsz]  # [bsz] int32

        # #50: Compressor / Indexer are fully pool-backed — no persistent
        # Python-owned kv_state / score_state / kv_cache buffers.  Bind
        # freqs_cis on first call (idempotent); pool context is set per-
        # forward below in the try/finally.
        if self.compress_ratio:
            if self.compressor.freqs_cis is None:
                self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                if self.indexer.freqs_cis is None:
                    self.indexer.freqs_cis = self.freqs_cis
                if self.indexer.compressor.freqs_cis is None:
                    self.indexer.compressor.freqs_cis = self.freqs_cis

        # Q path
        with record_function_range("dsv4.attn.q_proj_norm_rope"):
            qr = self._rmsnorm_weighted(
                self._lin(self.wq_a, x), self.q_norm
            )  # [B, 1, q_lora]
            q = self._lin(self.wq_b, qr).unflatten(
                -1, (self.n_heads, self.head_dim)
            )  # [B, 1, H, D]
            # Per-request RoPE on q_pe — each req has its own start_pos. Vectorized
            # via apply_rotary_emb_batched (mirrors vLLM's batched cos/sin lookup).
            freqs_cis_per_req = self.freqs_cis[start_pos.long()]  # [B, freqs_dim]
            q = fused_rmsnorm_rope(q, None, freqs_cis_per_req, rd, eps=self.eps)
        if _dbg_decode:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_q", q)

        # KV path (single MQA head)
        with record_function_range("dsv4.attn.kv_proj_norm_rope"):
            kv = fused_rmsnorm_rope(
                self._lin(self.wkv, x),
                self.kv_norm,
                freqs_cis_per_req,
                rd,
                eps=self.eps,
            )
        if _dbg_decode:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_kv", kv)

        # Phase E5b: direct SWA pool write (register_buffer retired).
        kv_flat = kv.reshape(bsz * q_len, self.head_dim)  # [T, head_dim]
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        swa_pool_slots = attn_metadata.pool_write_slot_mappings.get(SWA_KV)
        swa_view = self._pool_view(SWA_KV)
        if _dbg_decode and swa_pool_slots is not None:
            _rt.record_if_level(
                2,
                f"L{self.layer_id:02d}_decode_swa_write_slot",
                swa_pool_slots[: bsz * q_len],
            )
        with record_function_range("dsv4.attn.swa_pool_write"):
            if (
                swa_view is not None
                and swa_pool_slots is not None
                and swa_pool_slots.numel() > 0
            ):
                write_kv_to_pool(
                    kv_flat,
                    swa_pool_slots[: bsz * q_len],
                    swa_view,
                    mask_negative=False,
                )

        # CSA / HCA: build / fill compressed topk; write compressed-K.
        # Stage 3B: when the metadata is from CUDA-graph capture, dispatch
        # to the vectorized (Python-branch-free) variants so the captured
        # forward holds no data-dependent control flow. Eager Phase 2 path
        # keeps the loop variants for byte-equal regression safety.
        # Always use the vectorized compressor/indexer decode variants.
        # Originally gated on cuda-graph capture for byte-equal regression
        # safety; the loop variants do per-request .item() D2H syncs which
        # serialize the GPU stream. vectorized is math-equivalent and
        # graph-capturable.
        use_vec = True
        topk_idxs: torch.Tensor
        if self.compress_ratio:
            # Slice topk_buffer_compressed to actual bsz so indexer writes
            # only the [:bsz] prefix (graph impl allocates [max_bs, ...]).
            topk_buf_cmp = attn_metadata.topk_buffer_compressed[:bsz]
            if self.indexer is not None:
                # CSA layer (ratio=4): indexer fills topk_buf_cmp with
                # per-request compressed-block indices and self-scatters
                # its nested compressor's kv_cache + state back into the
                # INDEXER_KV / INDEXER_STATE pool via the set_pool_context
                # lifecycle (#50).
                if use_vec:
                    with record_function_range("dsv4.attn.indexer"):
                        self.indexer.forward_decode_vectorized(
                            x,
                            qr,
                            start_pos,
                            topk_buf_cmp,
                        )
                else:
                    with record_function_range("dsv4.attn.indexer"):
                        self.indexer.forward_decode(
                            x,
                            qr,
                            start_pos,
                            topk_buf_cmp,
                        )
                # The Indexer owns INDEXER_KV/INDEXER_STATE only.  CSA
                # attention reads CSA_KV, so the layer's main compressor
                # must also emit the boundary compressed-K before the paged
                # dual-pool gather below.
                if use_vec:
                    with record_function_range("dsv4.attn.compressor"):
                        self.compressor.forward_decode_vectorized(x, start_pos)
                else:
                    with record_function_range("dsv4.attn.compressor"):
                        self.compressor.forward_decode(x, start_pos)
                # Stitch indexer output into topk_total compressed half (with +win offset).
                topk_total = attn_metadata.topk_total_by_ratio[4][
                    :bsz
                ]  # [bsz, 1, win+K]
                idx_with_off = torch.where(
                    topk_buf_cmp >= 0,
                    topk_buf_cmp + win,
                    topk_buf_cmp,
                )
                topk_total[:, :, win:] = idx_with_off
                topk_idxs = topk_total
            else:
                # HCA layer (ratio=128): use the dense-filled topk from builder.
                topk_total = attn_metadata.topk_total_by_ratio[ratio][:bsz].clone()
                cmp_part = topk_total[:, :, win:]
                cmp_part = torch.where(cmp_part >= 0, cmp_part + win, cmp_part)
                topk_total[:, :, win:] = cmp_part
                topk_idxs = topk_total
                # Compressor self-binds/writes/scatters compressed-K into
                # the HCA_KV + HCA_STATE pool via set_pool_context lifecycle.
                if _dbg_decode:
                    self.compressor._dbg_prefix = f"L{self.layer_id:02d}_decode_cmp"
                if use_vec:
                    try:
                        with record_function_range("dsv4.attn.compressor"):
                            self.compressor.forward_decode_vectorized(x, start_pos)
                    finally:
                        if _dbg_decode:
                            self.compressor._dbg_prefix = None
                else:
                    try:
                        with record_function_range("dsv4.attn.compressor"):
                            self.compressor.forward_decode(x, start_pos)
                    finally:
                        if _dbg_decode:
                            self.compressor._dbg_prefix = None
        else:
            # SWA-only layer: just window topk. Already request-local ring slots.
            topk_idxs = attn_metadata.topk_window_idxs[:bsz]

        # Phase 2B-2b: capture raw (no +win offset) compressed local idx for
        # the optional paged dual-pool read below. For CSA the indexer's
        # output buffer holds raw values (the +win is added in-place later);
        # for HCA we synthesize the dense [0..compressed_lens) range.
        cmp_local_raw: Optional[torch.Tensor] = None
        if self.compress_ratio:
            ratio_l = int(self.compress_ratio)
            if self.indexer is not None:
                # CSA: indexer just wrote raw indices into topk_buf_cmp.
                cmp_local_raw = attn_metadata.topk_buffer_compressed[:bsz].clone()
            else:
                # HCA: dense read of [0..compressed_lens) for each request.
                cmp_lens_h = attn_metadata.compressed_lens.get(ratio_l)
                tt_h = attn_metadata.topk_total_by_ratio.get(ratio_l)
                if cmp_lens_h is not None and tt_h is not None:
                    K_h = tt_h.shape[-1] - win
                    dense = (
                        torch.arange(K_h, device=cmp_lens_h.device, dtype=torch.int32)
                        .view(1, 1, K_h)
                        .expand(bsz, q_len, K_h)
                    )
                    cmp_local_raw = torch.where(
                        dense < cmp_lens_h[:bsz].view(bsz, 1, 1),
                        dense,
                        torch.full_like(dense, -1),
                    )

        # Sparse attn over per-request KV view.
        # NOTE: kv_cache layout is [max_B, win + max_seq_len/ratio, head_dim].
        # For SWA-only layers, only the [:, :win, :] slice carries valid data
        # but the buffer is allocated as [max_B, win, head_dim] (no compressed
        # tail). For CSA/HCA, the full buffer is used.
        sparse_op = SparseAttnV4DecodeOp(
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            softmax_scale=self.softmax_scale,
        )

        # Phase 2B-2a paged read (SWA-only layers, zero-copy):
        #   q[B, 1, H, D] → reshape view [1, B, H, D]
        #   kv = swa_pool_view [num_global_slots, D] → unsqueeze [1, num_slots, D]
        #   topk = swa_global_slots [B, win] → unsqueeze [1, B, win]
        # No gather, no packed buffer — TileLang kernel does indirect read
        # through ``kv[by, idxs[i], j]`` (mirrors vLLM/flash_mla pattern).
        # Gated on env flag for safe rollout; CSA/HCA paths fall through to
        # legacy register_buffer below until Phase 2B-2b.
        from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, HCA_KV, SWA_KV

        swa_pool_bt = attn_metadata.pool_block_tables.get(SWA_KV)
        swa_view_cache = self._pool_view(SWA_KV)

        # Decide which paged read variant to use, if any.
        use_paged_swa_read = (
            not self.compress_ratio
            and swa_view_cache is not None
            and attn_metadata.swa_abs_idx is not None
            and swa_pool_bt is not None
            and swa_pool_bt.numel() > 0
        )
        # CSA layer cmp pool = CSA_KV; HCA layer cmp pool = HCA_KV.
        cmp_attn_type = (
            CSA_KV
            if (self.compress_ratio == 4)
            else HCA_KV if (self.compress_ratio == 128) else None
        )
        cmp_pool_bt = (
            attn_metadata.pool_block_tables.get(cmp_attn_type)
            if cmp_attn_type is not None
            else None
        )
        cmp_view_cache = (
            self._pool_view(cmp_attn_type) if cmp_attn_type is not None else None
        )
        use_paged_dual_read = (
            self.compress_ratio in (4, 128)
            and swa_view_cache is not None
            and cmp_view_cache is not None
            and attn_metadata.swa_abs_idx is not None
            and swa_pool_bt is not None
            and swa_pool_bt.numel() > 0
            and cmp_pool_bt is not None
            and cmp_pool_bt.numel() > 0
            and cmp_local_raw is not None
        )

        if use_paged_swa_read or use_paged_dual_read:
            from rtp_llm.models_py.modules.dsv4.decode.paged_topk_translator import (
                build_req_id_per_token,
                gather_dual_pool_kv_packed,
                translate_local_to_global_slots,
            )

            T = bsz * q_len
            req_id = build_req_id_per_token(bsz, q_len, swa_pool_bt.device)
            swa_eb = self._pool_entries_per_block(SWA_KV)
            swa_local = attn_metadata.swa_abs_idx[:bsz].reshape(T, win)
            with record_function_range("dsv4.attn.kv_gather_dense_or_paged"):
                swa_global = translate_local_to_global_slots(
                    req_id,
                    swa_pool_bt[:bsz],
                    swa_local,
                    swa_eb,
                )
            if _dbg_decode:
                _rt.record_if_level(
                    2, f"L{self.layer_id:02d}_decode_topk_idxs", topk_idxs
                )
                _rt.record_if_level(
                    2, f"L{self.layer_id:02d}_decode_swa_local", swa_local
                )
                _rt.record_if_level(
                    2, f"L{self.layer_id:02d}_decode_swa_global", swa_global
                )

            if use_paged_swa_read:
                # Zero-copy: pool view fed straight to TileLang kernel.
                q_packed = (
                    q.transpose(0, 1).contiguous() if q_len > 1 else q.transpose(0, 1)
                )
                with record_function_range("dsv4.attn.sparse_attn"):
                    o_packed = sparse_op.forward(
                        q_packed,
                        swa_view_cache.unsqueeze(0),
                        self.attn_sink,
                        swa_global.view(1, T, win).contiguous(),
                    )
                o = o_packed.transpose(0, 1)
            else:
                # Dual-pool: TileLang kernel can't take 2 kv tensors so we
                # gather both into a packed scratch and call sparse_attn
                # with identity topk = arange(win+K). Memory cost noted in
                # paged_topk_translator.gather_dual_pool_kv_packed docstring.
                cmp_eb = self._pool_entries_per_block(cmp_attn_type)
                K_cmp = cmp_local_raw.shape[-1]
                cmp_local = cmp_local_raw.reshape(T, K_cmp)
                with record_function_range("dsv4.attn.kv_gather_dense_or_paged"):
                    cmp_global = translate_local_to_global_slots(
                        req_id,
                        cmp_pool_bt[:bsz],
                        cmp_local,
                        cmp_eb,
                    )
                if _dbg_decode:
                    _rt.record_if_level(
                        2, f"L{self.layer_id:02d}_decode_cmp_local", cmp_local
                    )
                    _rt.record_if_level(
                        2, f"L{self.layer_id:02d}_decode_cmp_global", cmp_global
                    )
                assert q_len == 1, (
                    "Phase 2B-2b dual-pool paged read currently supports "
                    f"q_len=1 only (got {q_len})"
                )
                with record_function_range("dsv4.attn.kv_gather_dense_or_paged"):
                    kv_packed_4d = gather_dual_pool_kv_packed(
                        swa_view_cache,
                        cmp_view_cache,
                        swa_global,
                        cmp_global,
                        self.head_dim,
                        bsz,
                        q_len,
                    )  # [B, 1, win+K, D]
                kv_packed = kv_packed_4d.view(bsz, win + K_cmp, self.head_dim)
                if _dbg_decode:
                    _rt.record_if_level(
                        2, f"L{self.layer_id:02d}_decode_kv_packed", kv_packed
                    )
                swa_valid = (swa_global >= 0).view(bsz, q_len, win)
                cmp_valid = (cmp_global >= 0).view(bsz, q_len, K_cmp)
                swa_topk = (
                    torch.arange(win, device=kv_packed.device, dtype=torch.int32)
                    .view(1, 1, win)
                    .expand(bsz, q_len, win)
                )
                cmp_topk = (
                    torch.arange(K_cmp, device=kv_packed.device, dtype=torch.int32)
                    .add_(win)
                    .view(1, 1, K_cmp)
                    .expand(bsz, q_len, K_cmp)
                )
                packed_topk = torch.cat(
                    [
                        torch.where(swa_valid, swa_topk, torch.full_like(swa_topk, -1)),
                        torch.where(cmp_valid, cmp_topk, torch.full_like(cmp_topk, -1)),
                    ],
                    dim=-1,
                )
                with record_function_range("dsv4.attn.sparse_attn"):
                    o = sparse_op.forward(
                        q, kv_packed, self.attn_sink, packed_topk.contiguous()
                    )
        else:
            # Phase E5b: register_buffer retired.  Production decode must
            # populate paged metadata; any path here is a caller bug.
            # Include gating state so a regression surfaces precisely
            # rather than as a bare "no path taken" error.
            pool_bt_keys = (
                list(attn_metadata.pool_block_tables.keys())
                if attn_metadata.pool_block_tables is not None
                else None
            )
            raise RuntimeError(
                "[DSV4] forward_decode requires paged metadata "
                f"(layer={self.layer_id}, ratio={self.compress_ratio}); "
                "Phase E5b removed the register_buffer fallback. "
                f"swa_view_cache={'set' if swa_view_cache is not None else 'None'}, "
                f"swa_pool_bt_numel={swa_pool_bt.numel() if swa_pool_bt is not None else 'None'}, "
                f"swa_abs_idx={'set' if attn_metadata.swa_abs_idx is not None else 'None'}, "
                f"cmp_attn_type={cmp_attn_type}, "
                f"cmp_view_cache={'set' if cmp_view_cache is not None else 'None'}, "
                f"cmp_pool_bt_numel={cmp_pool_bt.numel() if cmp_pool_bt is not None else 'None'}, "
                f"cmp_local_raw={'set' if cmp_local_raw is not None else 'None'}, "
                f"pool_bt_keys={pool_bt_keys}, "
                f"kv_cache_bound={self._kv_cache is not None}"
            )

        # Grouped output projection: inverse-RoPE + FP8 quant + wo_a einsum.
        # Fused path collapses torch ``apply_rotary_emb_batched`` (5 launches)
        # + per-group ``per_token_group_quant_fp8`` (G launches) into ONE
        # Triton kernel emitting ``(fp8 [M,G,K], scale [M,G,K/512])`` in the
        # einsum-expected UE8M0 layout.  Matches vLLM ``deepseek_v4_attention.py``.
        with record_function_range("dsv4.attn.out_proj"):
            if o.is_cuda and o.numel() > 0:
                o_fp8, o_scale = fused_inv_rope_fp8_quant(
                    o,
                    freqs_cis_per_req,
                    n_groups=self.n_groups,
                    heads_per_group=self.n_heads // self.n_groups,
                    nope_dim=self.head_dim - self.rope_head_dim,
                    rope_head_dim=self.rope_head_dim,
                )
                del o
                o = self._wo_a_einsum_from_fp8(o_fp8, o_scale, bsz, q_len)
            else:
                apply_rotary_emb_batched(o[..., -rd:], freqs_cis_per_req, inverse=True)
                o = o.reshape(bsz, q_len, self.n_groups, -1)
                wo_a_bf16 = _fp8_dequant_to_fp32(self.wo_a_w, self.wo_a_s).to(o.dtype)
                wo_a = wo_a_bf16.view(self.n_groups, self.o_lora_rank, -1)
                o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
            out = self._lin(self.wo_b, o.flatten(2))
        if self.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            with record_function_range("dsv4.attn.tp_all_reduce"):
                all_reduce(out, Group.TP)
        return out

    def forward(
        self,
        x: torch.Tensor,
        start_pos,
        sequence_lengths=None,
        kv_cache: Optional[Any] = None,
        block_tables_by_type: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """qwen3-style: ``kv_cache`` (framework KVCache handle) and
        ``block_tables_by_type`` (per-request, per-attn_type block tables)
        flow in as kwargs on every forward call. We stash them on ``self``
        for the duration of this call so the many internal helpers
        (``_pool_view``, ``_prefill_*_write_*``, ``_bind_compressor_state
        _for_prefill``, etc.) can use ``self._kv_cache`` /
        ``self._block_tables_by_type`` without threading the handle
        through every method signature. try/finally restores the prior
        state so recursive or mis-used callers see deterministic clears.

        Accepts either ``[B, S, dim]`` (legacy) or the flat ``[T, dim]``
        layout that vLLM uses.  Flat input is reboxed to ``[1, T, dim]``
        for the existing ``_forward_body`` and squeezed on the way out
        so the caller sees ``[T, dim]``.  Full internal flatten is a
        follow-up; for now this just closes the ``Block.forward``
        unsqueeze/squeeze shim.
        """
        prev_kv = self._kv_cache
        prev_bt = self._block_tables_by_type
        if kv_cache is not None:
            self._kv_cache = kv_cache
        if block_tables_by_type is not None:
            self._block_tables_by_type = block_tables_by_type
        was_flat = x.dim() == 2
        x_in = x.unsqueeze(0) if was_flat else x
        try:
            self._set_compressor_pool_context()
            try:
                out = self._forward_body(x_in, start_pos, sequence_lengths)
            finally:
                self._clear_compressor_pool_context()
        finally:
            self._kv_cache = prev_kv
            self._block_tables_by_type = prev_bt
        return out.squeeze(0) if was_flat else out

    def _forward_body(
        self, x: torch.Tensor, start_pos, sequence_lengths=None
    ) -> torch.Tensor:
        """Forward pass. start_pos can be int (B=1) or tensor [B] for batched decode."""
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        # Master switch: when MOEDBG=0 the AND short-circuits so neither the
        # layer_id compare nor any record_if_level call site below runs.
        _dbg = _rt.should_record_layer(self.layer_id)
        bsz, seqlen, _ = x.size()

        # vLLM-flow prefill dispatch (env switch + CSA/HCA layer +
        # bsz==1 prefill).  Falls through to the legacy body for SWA-only
        # layers, batched prefills, decode, etc.  CP prefill is supported
        # via the CP branch in ``_prefill_common_setup_vllm`` /
        # ``_prefill_compute_qkv_vllm`` / ``_forward_prefill_compressed_vllm``
        # (rank-local Q × all-gathered KV with global topk).
        is_batched_local = isinstance(start_pos, torch.Tensor) and start_pos.numel() > 1
        if seqlen > 1 and not is_batched_local and self.compress_ratio in (4, 128):
            return self._forward_prefill_vllm(x, start_pos, sequence_lengths)
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        device = x.device

        cp_ctx = self._cp_ctx
        is_batched = isinstance(start_pos, torch.Tensor) and start_pos.numel() > 1
        is_batched_decode = is_batched and seqlen == 1
        is_batched_prefill = is_batched and seqlen > 1
        is_prefill_attn = is_batched_prefill or (not is_batched and seqlen > 1)
        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1 and is_prefill_attn

        # ``any_cont`` decides whether prefill attention reads the temporary
        # linear [current KV | current compressed KV] buffer (fresh prefill) or
        # the framework paged pools (continuation / mixed prefill).  Compute it
        # before topk construction because those two layouts use different
        # index spaces.
        if cp_on:
            sp_int = int(cp_ctx.prefix_length)
            any_cont = sp_int > 0
        elif isinstance(start_pos, torch.Tensor):
            if start_pos.numel() == 1:
                sp_int = int(start_pos.item())
            else:
                sp_int = 0  # placeholder; batched paths use start_pos tensor directly
            any_cont = bool((start_pos > 0).any().item())
        else:
            sp_int = int(start_pos)
            any_cont = sp_int > 0

        # Per-token RoPE angles.  Non-CP uses the contiguous window
        # freqs_cis[start_pos:start_pos+seqlen]; CP selects at each
        # rank-local token's GLOBAL position; batched prefill picks
        # per-row positions via gather.
        if cp_on:
            freqs_cis = cp_freqs_cis_local(self.freqs_cis, cp_ctx)
        elif is_batched_decode:
            # Batched decode: each batch element at different position, seqlen=1
            # Gather freqs for each batch element's position
            positions = start_pos.to(torch.long)
            freqs_cis = self.freqs_cis[positions]  # [B, rope_dim//2]
            freqs_cis = freqs_cis.unsqueeze(1)  # [B, 1, rope_dim//2]
        elif is_batched_prefill:
            # Batched prefill: each row at its own sp, advanced by local
            # index i ∈ [0, seqlen).  Returns [B, seqlen, rope_dim//2].
            sp_t = start_pos.to(device=device, dtype=torch.long)  # [B]
            positions = sp_t.unsqueeze(1) + torch.arange(
                seqlen, device=device, dtype=torch.long
            ).unsqueeze(
                0
            )  # [B, S]
            pos_max = int(self.freqs_cis.shape[0])
            positions = positions.clamp(max=pos_max - 1)
            freqs_cis = self.freqs_cis[positions]  # [B, S, rope_dim//2]
        else:
            sp = int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
            freqs_cis = self.freqs_cis[sp : sp + seqlen]

        # #50: Compressor / Indexer are fully pool-backed — no persistent
        # Python buffers to allocate.  Bind freqs_cis once (idempotent).
        if self.compress_ratio:
            if self.compressor.freqs_cis is None:
                self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                if self.indexer.freqs_cis is None:
                    self.indexer.freqs_cis = self.freqs_cis
                if self.indexer.compressor.freqs_cis is None:
                    self.indexer.compressor.freqs_cis = self.freqs_cis

        # Q path
        with record_function_range("dsv4.attn.q_proj_norm_rope"):
            qr = self._rmsnorm_weighted(
                self._lin(self.wq_a, x), self.q_norm
            )  # [B, S, q_lora_rank]
            if _dbg:
                _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_qr_norm", qr)
            q = self._lin(self.wq_b, qr).unflatten(-1, (self.n_heads, self.head_dim))
            if _dbg:
                _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_q_pre_rmsnorm", q)
            # Per-head QK RMSNorm (no learnable scale, per official code) + partial
            # RoPE, single Triton launch.  ``fused_rmsnorm_rope`` handles both
            # prefill (``freqs_cis`` per-position ``[S, rd/2]``) and batched-
            # decode-in-prefill (``[B, 1, rd/2]``) via the unified freq_stride.
            q = fused_rmsnorm_rope(q, None, freqs_cis, rd, eps=self.eps)
        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_q_post_rope", q)
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_freqs_cis", freqs_cis)

        # KV path (single KV head) — rank-local under CP.
        with record_function_range("dsv4.attn.kv_proj_norm_rope"):
            kv_in = self._lin(self.wkv, x)
            kv = fused_rmsnorm_rope(kv_in, self.kv_norm, freqs_cis, rd, eps=self.eps)
        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_kv_post_rope_local", kv)

        # Under CP prefill, start the KV all-gather as soon as rank-local KV
        # is ready. Top-k/indexer work below is independent and can overlap
        # with the collective; restore/wait happens at the first KV consumer.
        kv_full_handle = None
        if cp_on:
            assert cp_ctx is not None
            if kv.dim() != 3 or kv.size(0) != 1:
                raise RuntimeError(
                    f"CP attention KV expects [1, T_local, D], got {tuple(kv.shape)}"
                )
            # CP gather API is flattened token-major 2D; attention owns the
            # module-local batch dimension.
            with record_function_range("dsv4.attn.cp_kv_gather_start"):
                kv_full_handle = cp_all_gather_full_async(kv.squeeze(0), cp_ctx)
            kv_full = None
            seqlen_full = cp_ctx.seq_len_full
        else:
            kv_full = kv
            seqlen_full = seqlen
        if cp_on:
            prefill_kv_len = cp_ctx.seq_len_total
        elif is_batched_prefill:
            if sequence_lengths is not None:
                row_seqlens_for_kv = sequence_lengths.to(
                    device=device, dtype=torch.long
                )
            else:
                row_seqlens_for_kv = torch.full(
                    (bsz,), seqlen, device=device, dtype=torch.long
                )
            prefill_kv_len = int(
                (start_pos.to(device=device, dtype=torch.long) + row_seqlens_for_kv)
                .max()
                .item()
            )
        else:
            prefill_kv_len = sp_int + seqlen
        prefill_swa_dense_len = prefill_kv_len
        # Build topk_idxs — rows = rank-local Q; columns reference either the
        # fresh-prefill [sliding | compressed] tensor or the continuation
        # paged-pool [SWA absolute stream | compressed] tensor.
        with record_function_range("dsv4.attn.topk.window"):
            if cp_on:
                topk_idxs = _get_window_topk_idxs_cp(
                    win,
                    bsz,
                    prefill_kv_len,
                    cp_ctx.global_positions,
                )
            elif is_batched_decode:
                # Batched decode: vectorized topk_idxs — each batch at different ring position
                sp = start_pos % win  # [B]
                offsets = torch.arange(win, device=device)  # [win]
                idxs = (sp.unsqueeze(1) + 1 + offsets.unsqueeze(0)) % win  # [B, win]
                valid_count = torch.clamp(start_pos + 1, max=win)  # [B]
                invalid = offsets.unsqueeze(0) < (win - valid_count.unsqueeze(1))
                idxs = torch.where(invalid, -1, idxs)
                topk_idxs = idxs.unsqueeze(1)  # [B, 1, win]
            elif is_batched_prefill:
                sp_t = start_pos.to(device=device, dtype=torch.long)
                if sequence_lengths is not None:
                    row_seqlens = sequence_lengths.to(device=device, dtype=torch.long)
                else:
                    row_seqlens = torch.full(
                        (bsz,), seqlen, device=device, dtype=torch.long
                    )
                topk_idxs = _get_window_topk_idxs_batched(
                    win, seqlen, sp_t, row_seqlens, device
                )
            else:
                sp = (
                    int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
                )
                topk_idxs = _get_window_topk_idxs(win, bsz, seqlen, sp, device)
        if self.compress_ratio:
            # Fresh prefill attends over [current sliding KV | current compressed KV].
            # Continuation prefill uses a dense absolute SWA view; decode keeps
            # the compact [SWA ring | compressed] layout.
            if is_prefill_attn:
                offset = prefill_swa_dense_len
            else:
                offset = win
            with record_function_range("dsv4.attn.topk.compress"):
                if self.indexer is not None:
                    if _dbg:
                        self.indexer._dbg_prefix = f"L{self.layer_id:02d}_attn_idx"
                    if is_batched_decode:
                        # Indexer now supports tensor start_pos directly
                        compress_idxs = self.indexer(x, qr, start_pos, offset)
                    elif is_batched_prefill:
                        # Indexer batched prefill passes through sp tensor;
                        # Indexer handles per-row seqlens internally.
                        compress_idxs = self.indexer(
                            x,
                            qr,
                            start_pos,
                            offset,
                            sequence_lengths=sequence_lengths,
                        )
                    else:
                        compress_idxs = self.indexer(x, qr, start_pos, offset)
                    if _dbg:
                        self.indexer._dbg_prefix = None
                        _rt.record_if_level(
                            2,
                            f"L{self.layer_id:02d}_attn_compress_idxs",
                            compress_idxs,
                        )
                elif cp_on:
                    compress_idxs = _get_compress_topk_idxs_cp(
                        ratio,
                        bsz,
                        prefill_kv_len,
                        offset,
                        cp_ctx.global_positions,
                    )
                elif is_batched_decode:
                    # Vectorized compress_idxs for HCA batched decode (no indexer)
                    n_entries = (start_pos + 1) // ratio  # [B]
                    max_entries = int(n_entries.max().item())
                    if max_entries > 0:
                        entry_range = torch.arange(max_entries, device=device)
                        valid = entry_range.unsqueeze(0) < n_entries.unsqueeze(
                            1
                        )  # [B, max_entries]
                        c_idxs = torch.where(
                            valid, entry_range.unsqueeze(0) + offset, -1
                        )
                        compress_idxs = c_idxs.unsqueeze(1)  # [B, 1, max_entries]
                    else:
                        compress_idxs = torch.full(
                            (bsz, 1, 0), -1, device=device, dtype=torch.long
                        )
                elif is_batched_prefill:
                    sp_t_cmp = start_pos.to(device=device, dtype=torch.long)
                    if sequence_lengths is not None:
                        row_seqlens_cmp = sequence_lengths.to(
                            device=device, dtype=torch.long
                        )
                    else:
                        row_seqlens_cmp = torch.full(
                            (bsz,), seqlen, device=device, dtype=torch.long
                        )
                    compress_idxs = _get_compress_topk_idxs_batched(
                        ratio, seqlen, offset, sp_t_cmp, row_seqlens_cmp, device
                    )
                else:
                    sp_int = (
                        int(start_pos)
                        if isinstance(start_pos, torch.Tensor)
                        else start_pos
                    )
                    compress_idxs = _get_compress_topk_idxs(
                        ratio, bsz, seqlen, sp_int, offset, device
                    )
            with record_function_range("dsv4.attn.topk.cat_cast"):
                topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
                topk_idxs = topk_idxs.long()
        else:
            with record_function_range("dsv4.attn.topk.cat_cast"):
                topk_idxs = topk_idxs.long()

        dbg_last_idx = None
        dbg_pos_idx = None
        dbg_pos_name = None
        if _dbg:
            if cp_on:
                last_pos = cp_ctx.seq_len_total - 1
                last_mask = (cp_ctx.global_positions == last_pos) & cp_ctx.local_is_real
                last_idx_t = torch.nonzero(last_mask, as_tuple=False).flatten()
                if last_idx_t.numel() > 0:
                    dbg_last_idx = int(last_idx_t[0].item())
                dbg_pos = getattr(_rt, "_DBG_GLOBAL_POS", -1)
                if dbg_pos >= 0:
                    pos_mask = (
                        cp_ctx.global_positions == dbg_pos
                    ) & cp_ctx.local_is_real
                    pos_idx_t = torch.nonzero(pos_mask, as_tuple=False).flatten()
                    if pos_idx_t.numel() > 0:
                        dbg_pos_idx = int(pos_idx_t[0].item())
                        dbg_pos_name = f"pos{dbg_pos}"
            elif bsz == 1:
                if sequence_lengths is not None and sequence_lengths.numel() > 0:
                    dbg_last_idx = int(sequence_lengths.reshape(-1)[0].item()) - 1
                else:
                    dbg_last_idx = seqlen - 1
                if dbg_last_idx < 0 or dbg_last_idx >= seqlen:
                    dbg_last_idx = None
                dbg_pos = getattr(_rt, "_DBG_GLOBAL_POS", -1)
                if dbg_pos >= 0:
                    dbg_pos_idx = dbg_pos - sp_int
                    if dbg_pos_idx < 0 or dbg_pos_idx >= seqlen:
                        dbg_pos_idx = None
                    else:
                        dbg_pos_name = f"pos{dbg_pos}"
            if dbg_last_idx is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_q_last",
                    q[:, dbg_last_idx : dbg_last_idx + 1],
                )
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_topk_last",
                    topk_idxs[:, dbg_last_idx : dbg_last_idx + 1],
                )
            if dbg_pos_idx is not None and dbg_pos_name is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_q_{dbg_pos_name}",
                    q[:, dbg_pos_idx : dbg_pos_idx + 1],
                )
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_topk_{dbg_pos_name}",
                    topk_idxs[:, dbg_pos_idx : dbg_pos_idx + 1],
                )

        if cp_on:
            assert kv_full_handle is not None
            with record_function_range("dsv4.attn.cp_kv_gather_wait"):
                kv_full = cp_wait_gather_full(kv_full_handle).unsqueeze(0)
        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_kv_full", kv_full)

        if is_prefill_attn:
            if cp_on:
                pool_read_start = cp_ctx.prefix_length
                row_seqlens_for_pool = (
                    torch.full((bsz,), seqlen_full, device=device, dtype=torch.long)
                    if any_cont
                    else seqlen_full
                )
            else:
                pool_read_start = start_pos
                row_seqlens_for_pool = (
                    sequence_lengths
                    if sequence_lengths is not None
                    else torch.full((bsz,), seqlen, device=device, dtype=torch.long)
                )
            prefill_swa_dense_for_attn = None
            if any_cont:
                with record_function_range("dsv4.attn.kv_gather_dense_or_paged"):
                    prefill_swa_dense_for_attn = (
                        self._prefill_read_swa_dense_abs_from_pool(
                            bsz,
                            pool_read_start,
                            row_seqlens_for_pool,
                            prefill_swa_dense_len,
                            current_kv_full=kv_full,
                        )
                    )
            # Phase E5b: direct SWA pool write from kv_full (no register_buffer
            # intermediary).  The framework SWA pool is absolute-positioned;
            # prefix KV for continuation prefill already lives in the pool from
            # prior calls.
            if cp_on:
                # CP prefill has rank-local Q but kv_full is already all-gathered
                # in logical order for the current input.  Continuation prefill
                # must preserve the reused prefix offset when writing SWA slots;
                # otherwise the suffix overwrites positions 0..N.
                swa_write_start = cp_ctx.prefix_length
                swa_write_lengths = torch.full(
                    (bsz,), seqlen_full, device=device, dtype=torch.long
                )
            else:
                swa_write_start = start_pos
                swa_write_lengths = sequence_lengths
            with record_function_range("dsv4.attn.swa_pool_write"):
                self._prefill_write_swa_to_pool(
                    bsz, kv_full, swa_write_start, swa_write_lengths
                )
            if self.compress_ratio:
                # #50: compressor self-binds kv_state / score_state /
                # kv_cache from the framework pool at entry, runs body,
                # scatters back at exit.  Pool context was set by the
                # outer ``forward`` wrapper in its try/finally.
                if _dbg:
                    self.compressor._dbg_prefix = f"L{self.layer_id:02d}_attn_cmp"
                with record_function_range("dsv4.attn.compressor"):
                    kv_compress = self.compressor(
                        x, start_pos, sequence_lengths=sequence_lengths
                    )
                if _dbg:
                    self.compressor._dbg_prefix = None
                    if kv_compress is not None:
                        _rt.record_if_level(
                            2,
                            f"L{self.layer_id:02d}_attn_kv_compress",
                            kv_compress,
                        )
                if kv_compress is not None:
                    cmp_at = None
                    if self.compress_ratio == 4:
                        from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV

                        cmp_at = CSA_KV
                    elif self.compress_ratio == 128:
                        from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV

                        cmp_at = HCA_KV
                    cmp_write_start = sp_int // ratio
                    if cmp_at is not None:
                        self._prefill_paged_write_kv_range(
                            cmp_at, kv_compress, bsz, cmp_write_start
                        )
                    if not any_cont:
                        kv_cat = torch.cat([kv_full, kv_compress], dim=1)
                    else:
                        # Phase E5b: pool-only read.  Register_buffer is
                        # gone; continuation prefill gathers SWA+compressed
                        # from the framework pool.  Must have paged ctx.
                        # Mixed batches (some sp==0, some sp>0) also go
                        # through pool read — sp==0 rows' KV was just
                        # written to the pool above, so round-trip is
                        # byte-equal.
                        with record_function_range(
                            "dsv4.attn.kv_gather_dense_or_paged"
                        ):
                            kv_cat = self._gather_kv_cache_dense_from_pool(
                                bsz,
                                pool_read_start,
                                row_seqlens_for_pool,
                                swa_dense_len=prefill_swa_dense_len,
                                swa_dense_override=prefill_swa_dense_for_attn,
                                cmp_T=prefill_kv_len // ratio,
                            )
                        assert kv_cat is not None, (
                            "Phase E5b: continuation prefill requires paged "
                            "ctx (pass kv_cache=... + block_tables_by_type=... "
                            "to Attention.forward via V4Transformer)."
                        )
                        cmp_base = prefill_swa_dense_len + cmp_write_start
                        cmp_end = min(cmp_base + kv_compress.shape[1], kv_cat.shape[1])
                        if cmp_end > cmp_base:
                            kv_cat[:bsz, cmp_base:cmp_end] = kv_compress[
                                :bsz, : cmp_end - cmp_base
                            ].to(kv_cat.dtype)
                else:
                    # CompressorVLLM returns None — it wrote the current
                    # chunk's compressed K to the framework pool internally
                    # via ``self._launch``.  We still need the [sliding |
                    # compressed] cat layout downstream because the indexer's
                    # topk indices reference compressed-pool offsets shifted
                    # by ``prefill_swa_dense_len``.  Gather both segments from
                    # the pool; pass ``kv_full`` as the SWA override on fresh
                    # prefill so the sliding read avoids a pool round-trip.
                    with record_function_range("dsv4.attn.kv_gather_dense_or_paged"):
                        kv_cat = self._gather_kv_cache_dense_from_pool(
                            bsz,
                            pool_read_start,
                            row_seqlens_for_pool,
                            swa_dense_len=prefill_swa_dense_len,
                            swa_dense_override=(
                                kv_full[:bsz]
                                if not any_cont
                                else prefill_swa_dense_for_attn
                            ),
                            cmp_T=prefill_kv_len // ratio,
                        )
                    assert (
                        kv_cat is not None
                    ), "CompressorVLLM kv_cat assembly requires paged ctx."
            else:
                if not any_cont:
                    kv_cat = kv_full
                else:
                    with record_function_range("dsv4.attn.kv_gather_dense_or_paged"):
                        kv_cat = self._gather_kv_cache_dense_from_pool(
                            bsz,
                            pool_read_start,
                            row_seqlens_for_pool,
                            swa_dense_len=prefill_swa_dense_len,
                            swa_dense_override=prefill_swa_dense_for_attn,
                        )
                    assert (
                        kv_cat is not None
                    ), "Phase E5b: continuation prefill requires paged ctx."
            if _dbg:
                _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_kv_cat", kv_cat)
                if dbg_last_idx is not None and bsz == 1:
                    idx = topk_idxs[0, dbg_last_idx]
                    valid = idx >= 0
                    if valid.any():
                        safe_idx = idx[valid].to(torch.long)
                        _rt.record_if_level(
                            2,
                            f"L{self.layer_id:02d}_attn_kv_selected_last",
                            kv_cat[:, safe_idx],
                        )
                if dbg_pos_idx is not None and dbg_pos_name is not None and bsz == 1:
                    idx = topk_idxs[0, dbg_pos_idx]
                    valid = idx >= 0
                    if valid.any():
                        safe_idx = idx[valid].to(torch.long)
                        _rt.record_if_level(
                            2,
                            f"L{self.layer_id:02d}_attn_kv_selected_{dbg_pos_name}",
                            kv_cat[:, safe_idx],
                        )
            with record_function_range("dsv4.attn.sparse_attn"):
                if _tl_kernels.tilelang_available():
                    o = _tl_kernels.sparse_attn(
                        q, kv_cat, self.attn_sink, topk_idxs, self.softmax_scale
                    )
                else:
                    o = _sparse_attn(
                        q, kv_cat, self.attn_sink, topk_idxs, self.softmax_scale
                    )
        else:
            # Phase E5b: eager decode via Attention.forward removed — the
            # register_buffer kv_cache that this arm relied on is retired.
            # Production decode must go through Attention.forward_decode with
            # paged metadata; any caller here is stale test or warmup code.
            raise NotImplementedError(
                "Phase E5b: Attention.forward eager-decode path retired "
                "(register_buffer kv_cache removed). Use forward_decode + "
                "DSv4DecodeAttnMetadata with paged pool descriptors."
            )

        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_sparse_out", o)
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_topk_idxs", topk_idxs)
            if dbg_last_idx is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_sparse_out_last",
                    o[:, dbg_last_idx : dbg_last_idx + 1],
                )
            if dbg_pos_idx is not None and dbg_pos_name is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_sparse_out_{dbg_pos_name}",
                    o[:, dbg_pos_idx : dbg_pos_idx + 1],
                )
        if not _dbg:
            del q, kv_cat, topk_idxs

        with record_function_range("dsv4.attn.out_proj"):
            if (
                _dbg
                and os.environ.get("DSV4_MOEDBG_PREFILL_EXPLICIT_OUT_PROJ", "1") != "0"
            ):
                # Keep the debuggable explicit path when tensor probes are enabled.
                apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
                _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_o_post_inv_rope", o)
                if dbg_last_idx is not None:
                    _rt.record_if_level(
                        2,
                        f"L{self.layer_id:02d}_attn_o_post_inv_rope_last",
                        o[:, dbg_last_idx : dbg_last_idx + 1],
                    )
                if dbg_pos_idx is not None and dbg_pos_name is not None:
                    _rt.record_if_level(
                        2,
                        f"L{self.layer_id:02d}_attn_o_post_inv_rope_{dbg_pos_name}",
                        o[:, dbg_pos_idx : dbg_pos_idx + 1],
                    )
                o = o.reshape(bsz, seqlen, self.n_groups, -1)
                wo_a_bf16 = _fp8_dequant_to_fp32(self.wo_a_w, self.wo_a_s).to(o.dtype)
                wo_a = wo_a_bf16.view(self.n_groups, self.o_lora_rank, -1)
                o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
            # Grouped output projection: inverse-RoPE + FP8 quant + wo_a einsum.
            # Same fused path as forward_decode (line ~1648) — collapses
            # apply_rotary_emb (1 launch) + per-group per_token_group_quant_fp8
            # (G launches) into ONE Triton kernel emitting (fp8 [M,G,K], scale
            # [M,G,K/512]) in the einsum-expected UE8M0 layout. Matches vLLM
            # ``deepseek_v4_attention.py``.
            elif o.is_cuda and o.numel() > 0:
                # The Triton kernel computes ``b_idx = pid_token // q_len_per_b``
                # to index freqs. Decode uses [B, rd/2] freqs with q_len_per_b=1
                # so b_idx == request index. In prefill we have per-position
                # freqs [S, rd/2]; pass o as 3D [B*S, H, D] with freqs_cis_per_b
                # = freqs.expand(B*S, ...) so q_len_per_b=1 and b_idx == token
                # index → each token reads its own row.
                o_3d = o.reshape(bsz * seqlen, self.n_heads, self.head_dim)
                if freqs_cis.dim() == 2:
                    # Per-position freqs [S, rd/2] — broadcast to [B*S, rd/2].
                    freqs_per_token = (
                        freqs_cis.unsqueeze(0)
                        .expand(bsz, -1, -1)
                        .reshape(bsz * seqlen, -1)
                        .contiguous()
                    )
                else:
                    freqs_per_token = freqs_cis.contiguous()
                o_fp8, o_scale = fused_inv_rope_fp8_quant(
                    o_3d,
                    freqs_per_token,
                    n_groups=self.n_groups,
                    heads_per_group=self.n_heads // self.n_groups,
                    nope_dim=self.head_dim - self.rope_head_dim,
                    rope_head_dim=self.rope_head_dim,
                )
                del o, o_3d, freqs_per_token
                o = self._wo_a_einsum_from_fp8(o_fp8, o_scale, bsz, seqlen)
            else:
                apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
                o = o.reshape(bsz, seqlen, self.n_groups, -1)
                wo_a_bf16 = _fp8_dequant_to_fp32(self.wo_a_w, self.wo_a_s).to(o.dtype)
                wo_a = wo_a_bf16.view(self.n_groups, self.o_lora_rank, -1)
                o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
            if _dbg:
                _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_wo_a_out", o)
            out = self._lin(self.wo_b, o.flatten(2))
        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_wo_b_out_pre_ar", out)
        if self.tp_size > 1:
            # wo_b is row-split along K — each rank produces a partial
            # sum; AR combines across the tp group.
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            with record_function_range("dsv4.attn.tp_all_reduce"):
                all_reduce(out, Group.TP)
            if _dbg:
                _rt.record_if_level(
                    2, f"L{self.layer_id:02d}_attn_wo_b_out_post_ar", out
                )
        return out

    # ==================================================================
    # vLLM-flow prefill (mirror of source ``fp8/attention.py``'s
    # ``_forward_prefill`` family, but BF16 KV-cache throughout and
    # using the local CompressorVLLM / IndexerVLLM with hoisted meta).
    #
    # Always-on for this class (block.py picks AttentionVLLM only when
    # ``DSV4_BF16_VLLM=1``); entry hook lives in :meth:`_forward_body`.
    # Constraints (matching
    # the source class):
    #   * single request only (bsz == 1)
    #   * no Context-Parallel (``set_cp_ctx`` accepts ``None`` only on
    #     IndexerVLLM); CP prefills fall through to the legacy path.
    #   * SWA-only layers (compress_ratio == 0) also fall through —
    #     IndexerVLLM/CompressorVLLM are no-ops there, so reusing the
    #     legacy path costs nothing and avoids re-implementing the
    #     window-only attention here.
    # ==================================================================

    def _forward_prefill_vllm(
        self,
        x: torch.Tensor,
        start_pos,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """vLLM-flow prefill entry. Mirrors source ``_forward_prefill``:
        common setup → QKV → SWA pool write → CSA / HCA dispatch.

        ``x`` is ``[1, S, dim]`` (caller already unsqueezed flat input);
        output is ``[1, S, dim]``.
        """
        del sequence_lengths  # bsz==1 invariant; sequence_lengths is redundant
        common = self._prefill_common_setup_vllm(x, start_pos)
        qkv = self._prefill_compute_qkv_vllm(x, common)
        self._prefill_write_swa_bf16_vllm(common, qkv.kv_full)
        if self.compress_ratio == 4:
            return self._forward_prefill_csa_vllm(x, qkv, common)
        if self.compress_ratio == 128:
            return self._forward_prefill_hca_vllm(x, qkv, common)
        raise AssertionError(
            f"_forward_prefill_vllm only handles compress_ratio in {{4, 128}}; "
            f"got {self.compress_ratio} (SWA-only layers should fall through "
            "to the legacy path)"
        )

    # ------------------------------------------------------------------
    # Common setup — single-pass build of every per-call meta the hot
    # path needs. Mirrors source ``_prefill_common_setup`` +
    # ``_build_csa_prefill_meta`` / ``_build_hca_prefill_meta`` (collapsed
    # into one pass since BF16 KV pool needs far fewer fields than the
    # FP8 workspace path).
    # ------------------------------------------------------------------
    def _prefill_common_setup_vllm(
        self, x: torch.Tensor, start_pos
    ) -> "_VLLMPrefillCommon":
        from rtp_llm.models_py.modules.dsv4.attn_type import INDEXER_KV
        from rtp_llm.models_py.modules.dsv4.compressor_vllm import (
            build_prefill_metadata as _build_compressor_prefill_metadata,
        )

        bsz, seqlen, _ = x.size()
        device = x.device
        cp_ctx = self._cp_ctx
        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1 and seqlen > 1

        if cp_on:
            # CP prefill — sp / end_pos / freqs_cis follow GLOBAL geometry;
            # rank-local Q dimension stays as ``seqlen`` (== chunk_length).
            sp_int = int(cp_ctx.prefix_length)
            end_pos = int(cp_ctx.seq_len_total)
            freqs_cis_slice = cp_freqs_cis_local(self.freqs_cis, cp_ctx)
        else:
            if isinstance(start_pos, torch.Tensor):
                sp_int = int(start_pos.item()) if start_pos.numel() == 1 else 0
            else:
                sp_int = int(start_pos)
            end_pos = sp_int + seqlen
            freqs_cis_slice = self.freqs_cis[sp_int : sp_int + seqlen]
        is_fresh = sp_int == 0

        # Bind freqs_cis to compressor / indexer (idempotent).
        if self.compressor.freqs_cis is None:
            self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            if self.indexer.freqs_cis is None:
                self.indexer.freqs_cis = self.freqs_cis
            if self.indexer.compressor.freqs_cis is None:
                self.indexer.compressor.freqs_cis = self.freqs_cis

        # Hoisted compressor meta — built once here, consumed by both
        # the host compressor and the nested indexer compressor (when
        # the latter shares positions / b_idx with the host, which it
        # does for bsz==1 prefill).
        # Under CP the compressor consumes the all-gathered ``[1, seq_len_full,
        # H]`` kv/score (see compressor_vllm.forward CP branch), so the meta
        # must cover ``seq_len_full`` positions — not the rank-local chunk.
        meta_seqlen = int(cp_ctx.seq_len_full) if cp_on else seqlen
        compressor_meta = _build_compressor_prefill_metadata(
            self.compressor, sp_int, bsz, meta_seqlen, device
        )

        # Hoisted indexer meta — only meaningful for CSA layers.  Under CP
        # the indexer's ``prepare`` reads ``self._cp_ctx`` and overrides
        # sp/end/positions internally; ``seqlen`` here is rank-local Q count.
        indexer_meta = None
        if self.indexer is not None:
            kv_block_table = (
                self._block_tables_by_type.get(INDEXER_KV)
                if self._block_tables_by_type is not None
                else None
            )
            kv_eb = self._pool_entries_per_block(INDEXER_KV)
            indexer_meta = self.indexer.prepare(
                bsz=bsz,
                seqlen=seqlen,
                sp_int=sp_int,
                device=device,
                kv_block_table=kv_block_table,
                kv_eb=kv_eb,
            )

        return _VLLMPrefillCommon(
            bsz=bsz,
            seqlen=seqlen,
            sp_int=sp_int,
            end_pos=end_pos,
            is_fresh_prefill=is_fresh,
            device=device,
            freqs_cis=freqs_cis_slice,
            swa_dense_len=end_pos,
            compressor_meta=compressor_meta,
            indexer_meta=indexer_meta,
            cp_ctx=cp_ctx if cp_on else None,
            cp_on=cp_on,
        )

    # ------------------------------------------------------------------
    # QKV proj + RMSNorm + RoPE, identical math to the legacy body
    # (lines ~2147-2191) — factored out so the hot path is readable.
    # ------------------------------------------------------------------
    def _prefill_compute_qkv_vllm(
        self, x: torch.Tensor, common: "_VLLMPrefillCommon"
    ) -> "_VLLMPrefillQKV":
        rd = self.rope_head_dim
        with record_function_range("dsv4.attn.q_proj_norm_rope"):
            qr = self._rmsnorm_weighted(self._lin(self.wq_a, x), self.q_norm)
            q = self._lin(self.wq_b, qr).unflatten(-1, (self.n_heads, self.head_dim))
            q = fused_rmsnorm_rope(q, None, common.freqs_cis, rd, eps=self.eps)
        with record_function_range("dsv4.attn.kv_proj_norm_rope"):
            kv_in = self._lin(self.wkv, x)
            kv_full = fused_rmsnorm_rope(
                kv_in, self.kv_norm, common.freqs_cis, rd, eps=self.eps
            )
        # CP prefill: rank-local KV needs to be all-gathered to the global
        # ``[1, seq_len_full, head_dim]`` layout that the SWA pool write +
        # downstream sparse_attn read both expect.  Mirror the legacy
        # ``_forward_body`` async-gather contract (see attention.py CP path).
        if common.cp_on:
            assert common.cp_ctx is not None
            assert kv_full.dim() == 3 and kv_full.size(0) == 1, (
                f"vLLM CP prefill KV expects [1, T_local, D], "
                f"got {tuple(kv_full.shape)}"
            )
            with record_function_range("dsv4.attn.cp_kv_gather_start"):
                handle = cp_all_gather_full_async(kv_full.squeeze(0), common.cp_ctx)
            with record_function_range("dsv4.attn.cp_kv_gather_wait"):
                kv_full = cp_wait_gather_full(handle).unsqueeze(0)
        return _VLLMPrefillQKV(q=q, qr=qr, kv_full=kv_full)

    # ------------------------------------------------------------------
    # SWA pool write — direct BF16 paged write (no FP8 quant). Reuses
    # the existing ``_prefill_write_swa_to_pool`` helper, which already
    # handles bsz>=1 / continuation prefill / sentinel block masking.
    # ------------------------------------------------------------------
    def _prefill_write_swa_bf16_vllm(
        self, common: "_VLLMPrefillCommon", kv_full: torch.Tensor
    ) -> None:
        bsz = common.bsz
        device = common.device
        # Under CP each rank writes the full all-gathered sequence to its
        # own SWA pool starting at the absolute prefix offset; outside CP
        # the rank-local seqlen is the absolute write length.
        write_len = int(common.cp_ctx.seq_len_full) if common.cp_on else common.seqlen
        swa_lengths = torch.full((bsz,), write_len, device=device, dtype=torch.long)
        with record_function_range("dsv4.attn.swa_pool_write"):
            self._prefill_write_swa_to_pool(bsz, kv_full, common.sp_int, swa_lengths)

    # ------------------------------------------------------------------
    # CSA / HCA dispatch (compress_ratio == 4 / 128).
    # ------------------------------------------------------------------
    def _forward_prefill_csa_vllm(
        self,
        x: torch.Tensor,
        qkv: "_VLLMPrefillQKV",
        common: "_VLLMPrefillCommon",
    ) -> torch.Tensor:
        """CSA path. IndexerVLLM produces sparse compressed-block topk
        with hoisted meta; main CompressorVLLM writes the CSA pool.
        Final attention runs through the shared
        :meth:`_forward_prefill_compressed_vllm` epilogue."""
        from rtp_llm.models_py.modules.dsv4.indexer_vllm import IndexerVLLM

        assert isinstance(self.indexer, IndexerVLLM), (
            "CSA vLLM prefill requires IndexerVLLM (mismatched indexer "
            "class — env switch likely changed mid-process)"
        )
        with record_function_range("dsv4.attn.indexer"):
            raw_int32 = self.indexer.forward_with_meta(x, qkv.qr, common.indexer_meta)
        # raw_int32 layout matches qkv.qr leading dims with K trailing.
        return self._forward_prefill_compressed_vllm(
            x, qkv, common, cmp_topk_runtime_int32=raw_int32
        )

    def _forward_prefill_hca_vllm(
        self,
        x: torch.Tensor,
        qkv: "_VLLMPrefillQKV",
        common: "_VLLMPrefillCommon",
    ) -> torch.Tensor:
        """HCA path. No indexer (dense compressed indices); main
        CompressorVLLM writes the HCA pool. Same epilogue as CSA."""
        assert self.indexer is None, "HCA layer must not have an indexer"
        return self._forward_prefill_compressed_vllm(
            x, qkv, common, cmp_topk_runtime_int32=None
        )

    # ------------------------------------------------------------------
    # Shared CSA / HCA epilogue: compressor write + dense KV cat +
    # sparse_attn + output proj.
    # ------------------------------------------------------------------
    def _forward_prefill_compressed_vllm(
        self,
        x: torch.Tensor,
        qkv: "_VLLMPrefillQKV",
        common: "_VLLMPrefillCommon",
        cmp_topk_runtime_int32: Optional[torch.Tensor],
    ) -> torch.Tensor:
        from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, HCA_KV

        bsz, seqlen, sp = common.bsz, common.seqlen, common.sp_int
        device = common.device
        ratio = self.compress_ratio
        win = self.window_size

        # Window topk (causal sliding view).  Under CP, rank-local Q at
        # local index i sits at GLOBAL position cp_ctx.global_positions[i];
        # the CP variant builds the per-row window in that global frame.
        if common.cp_on:
            assert common.cp_ctx is not None
            topk_window = _get_window_topk_idxs_cp(
                win, bsz, common.end_pos, common.cp_ctx.global_positions
            )
        else:
            topk_window = _get_window_topk_idxs(win, bsz, seqlen, sp, device)

        # Compressed topk for CSA (from IndexerVLLM raw int32 + offset)
        # or HCA (deterministic dense block range).
        offset = common.swa_dense_len  # SWA prefix length in kv_cat
        if cmp_topk_runtime_int32 is not None:
            cmp_topk = torch.where(
                cmp_topk_runtime_int32 >= 0,
                cmp_topk_runtime_int32 + offset,
                cmp_topk_runtime_int32,
            ).long()
        elif common.cp_on:
            assert common.cp_ctx is not None
            cmp_topk = _get_compress_topk_idxs_cp(
                ratio, bsz, common.end_pos, offset, common.cp_ctx.global_positions
            )
        else:
            cmp_topk = _get_compress_topk_idxs(ratio, bsz, seqlen, sp, offset, device)
        topk_idxs = torch.cat([topk_window, cmp_topk], dim=-1).long()

        # Main compressor — hoisted meta path; returns ``None`` since
        # CompressorVLLM scatters its output through the BF16 pool only.
        with record_function_range("dsv4.attn.compressor"):
            self.compressor(x, sp, meta=common.compressor_meta)
        # The vLLM compressor writes the compressed-K slot directly via
        # its own pool context — we still need to mirror it into the
        # global CSA/HCA pool view used by the dense gather below.
        # Fortunately ``_set_compressor_pool_context`` already bound the
        # same pool view, and the boundary kernel writes through that
        # view. So no additional mirror is needed here; the dense
        # gather will pick up the freshly written slots.

        # Build the dense [SWA | compressed] KV view sparse_attn reads.
        cmp_at = CSA_KV if ratio == 4 else HCA_KV
        cmp_T = common.end_pos // ratio
        # Pool-read geometry — CP path mirrors the legacy ``_forward_body``:
        # rank's pool already holds the full all-gathered sequence (compressor
        # writes it during ``_launch``, SWA write step above writes the
        # gathered ``qkv.kv_full``).  Reads use the global absolute frame.
        if common.cp_on:
            pool_read_lengths = torch.full(
                (bsz,),
                int(common.cp_ctx.seq_len_full),
                device=device,
                dtype=torch.long,
            )
        else:
            pool_read_lengths = torch.full(
                (bsz,), seqlen, device=device, dtype=torch.long
            )
        if common.is_fresh_prefill:
            # Fresh prefill: SWA part is in-memory (qkv.kv_full, already
            # all-gathered under CP); the compressed part lives in the pool
            # the compressor just wrote.  We still go through
            # ``_gather_kv_cache_dense_from_pool`` to pick up the compressed
            # slots; the SWA portion is overlaid from kv_full to avoid the
            # SWA pool round-trip.
            with record_function_range("dsv4.attn.kv_gather_dense_or_paged"):
                kv_cat = self._gather_kv_cache_dense_from_pool(
                    bsz,
                    sp,
                    pool_read_lengths,
                    swa_dense_len=common.swa_dense_len,
                    swa_dense_override=qkv.kv_full,
                    cmp_T=cmp_T,
                )
            assert kv_cat is not None, (
                "vLLM prefill requires paged context (kv_cache + "
                "block_tables_by_type bound by V4Transformer)"
            )
        else:
            with record_function_range("dsv4.attn.kv_gather_dense_or_paged"):
                kv_cat = self._gather_kv_cache_dense_from_pool(
                    bsz,
                    sp,
                    pool_read_lengths,
                    swa_dense_len=common.swa_dense_len,
                    cmp_T=cmp_T,
                )
            assert (
                kv_cat is not None
            ), "vLLM continuation prefill requires paged context"

        # Sparse attention via flash_mla_sparse_fwd (BF16 native, mirrors
        # source FP8 attention's ``_attn_via_workspace`` epilogue). Layout
        # massage:
        #   q       : [B=1, S, H, D]  → [S, H, D]
        #   kv      : [B=1, T_kv, D]  → [T_kv, 1, D]
        #   indices : [B=1, S, K]     → [S, 1, K_aligned] int32, padded
        #             to a multiple of 128 (matches source's
        #             ``_SPARSE_PREFILL_TOPK_ALIGNMENT``; SM100 head64
        #             kernel asserts ``params.topk % B_TOPK == 0`` with
        #             ``B_TOPK = 64`` so 128 is the safe upper bound).
        #             ``-1`` in the tail = kernel's invalid sentinel.
        #   attn_sink + sm_scale: unchanged
        # The kernel returns ``[S, H, D_v]`` BF16; we restore the leading
        # batch dim so :meth:`_prefill_output_proj_vllm` can ingest it.
        from flash_mla import flash_mla_sparse_fwd  # type: ignore[import-not-found]

        _SPARSE_PREFILL_TOPK_ALIGN = 128

        with record_function_range("dsv4.attn.sparse_attn"):
            q_flat = qkv.q.squeeze(0).contiguous()
            kv_flat = kv_cat.squeeze(0).unsqueeze(1).contiguous()

            indices_i32 = topk_idxs.squeeze(0).to(torch.int32)
            K_total = int(indices_i32.shape[-1])
            pad_K = (-K_total) % _SPARSE_PREFILL_TOPK_ALIGN
            if pad_K > 0:
                indices_i32 = F.pad(indices_i32, (0, pad_K), value=-1)
            indices_i32 = indices_i32.unsqueeze(1).contiguous()

            o3, _, _ = flash_mla_sparse_fwd(
                q=q_flat,
                kv=kv_flat,
                indices=indices_i32,
                sm_scale=self.softmax_scale,
                attn_sink=self.attn_sink,
            )
            o = o3.unsqueeze(0)

        return self._prefill_output_proj_vllm(o, common)

    # ------------------------------------------------------------------
    # Output projection — inv-RoPE + FP8 quant + wo_a einsum + wo_b lin
    # + TP all-reduce. Mirrors the production path in ``_forward_body``
    # (lines ~2604-2680, the ``elif o.is_cuda`` branch).
    # ------------------------------------------------------------------
    def _prefill_output_proj_vllm(
        self, o: torch.Tensor, common: "_VLLMPrefillCommon"
    ) -> torch.Tensor:
        bsz, seqlen = common.bsz, common.seqlen
        with record_function_range("dsv4.attn.out_proj"):
            if o.is_cuda and o.numel() > 0:
                o_3d = o.reshape(bsz * seqlen, self.n_heads, self.head_dim)
                if common.freqs_cis.dim() == 2:
                    freqs_per_token = (
                        common.freqs_cis.unsqueeze(0)
                        .expand(bsz, -1, -1)
                        .reshape(bsz * seqlen, -1)
                        .contiguous()
                    )
                else:
                    freqs_per_token = common.freqs_cis.contiguous()
                o_fp8, o_scale = fused_inv_rope_fp8_quant(
                    o_3d,
                    freqs_per_token,
                    n_groups=self.n_groups,
                    heads_per_group=self.n_heads // self.n_groups,
                    nope_dim=self.head_dim - self.rope_head_dim,
                    rope_head_dim=self.rope_head_dim,
                )
                del o, o_3d, freqs_per_token
                o = self._wo_a_einsum_from_fp8(o_fp8, o_scale, bsz, seqlen)
            else:
                # CPU / empty fallback (unit tests).
                apply_rotary_emb(
                    o[..., -self.rope_head_dim :], common.freqs_cis, inverse=True
                )
                o = o.reshape(bsz, seqlen, self.n_groups, -1)
                wo_a_bf16 = _fp8_dequant_to_fp32(self.wo_a_w, self.wo_a_s).to(o.dtype)
                wo_a = wo_a_bf16.view(self.n_groups, self.o_lora_rank, -1)
                o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
            out = self._lin(self.wo_b, o.flatten(2))
        if self.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            with record_function_range("dsv4.attn.tp_all_reduce"):
                all_reduce(out, Group.TP)
        return out
