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

import math
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
from rtp_llm.models_py.modules.dsv4.compressor import Compressor
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    cp_all_gather_full,
    cp_freqs_cis_local,
)
from rtp_llm.models_py.modules.dsv4.indexer import Indexer
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

_DSV4_FP8_KV_ENTRY_BYTES = 584
_DSV4_FP8_INDEXER_ENTRY_BYTES = 132


def _is_fp8_kv_cache_dtype(kv_cache_dtype: Any) -> bool:
    if kv_cache_dtype is None:
        return False
    name = getattr(kv_cache_dtype, "name", None)
    if isinstance(name, str):
        return name.upper() == "FP8"
    value = str(kv_cache_dtype).lower()
    return value == "fp8" or value.endswith(".fp8")


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


class PrefillMeta(NamedTuple):
    """Per-call prefill metadata, layer-invariant within a
    ``compress_ratio`` bucket. Built once per (forward, ratio) by
    :meth:`Attention._build_shared_prefill_meta` and broadcast to every
    same-ratio layer via :meth:`Attention._set_prefill_meta_shared`.
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
    swa_meta: Optional[SwaPrefillMeta] = None
    indexer_meta: Optional[Any] = None  # IndexerFP8PrefillMeta when applicable


class PrefillQKV(NamedTuple):
    """Q/KV intermediate produced by ``_prefill_compute_qkv``.

    ``qr`` is fed to the indexer (CSA layers); ``q`` is the dense Q.
    ``kv_full`` is the all-gathered KV under CP; equals ``kv`` otherwise.
    The CP-aware sequence length lives on ``PrefillMeta.seqlen_full``.
    """

    qr: torch.Tensor
    q: torch.Tensor
    kv_full: torch.Tensor


class Attention(nn.Module):
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
        kv_cache_dtype: Any = None,
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
        self._kv_cache_dtype = kv_cache_dtype
        self._kv_cache_is_fp8 = _is_fp8_kv_cache_dtype(kv_cache_dtype)
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
            if self._kv_cache_is_fp8:
                from rtp_llm.models_py.modules.dsv4.compressor_fp8 import CompressorFP8

                self.compressor = CompressorFP8(
                    dim=dim,
                    head_dim=head_dim,
                    rope_head_dim=rope_head_dim,
                    compress_ratio=compress_ratio,
                    max_batch_size=max_batch_size,
                    norm_eps=norm_eps,
                    compressor_weights=outer_cmp_weights,
                )
            else:
                self.compressor = Compressor(
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
                if self._kv_cache_is_fp8:
                    from rtp_llm.models_py.modules.dsv4.indexer_fp8 import IndexerFP8

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
                else:
                    self.indexer = Indexer(
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
        kv_spec = (
            (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES)
            if self._kv_cache_is_fp8
            else (torch.bfloat16, head_dim)
        )
        indexer_kv_spec = (
            (torch.uint8, _DSV4_FP8_INDEXER_ENTRY_BYTES)
            if self._kv_cache_is_fp8
            else (torch.bfloat16, idx_hd)
        )
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
        from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
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
        if self._kv_cache_is_fp8:
            pool_view = None
        else:
            pool_view = self._pool_view(SWA_KV)
            if pool_view is None:
                return

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
        if self._kv_cache_is_fp8:
            # FP8 SWA pool: 584B per slot (fp8 NoPE 448 + bf16 RoPE 128 +
            # ue8m0 scale 8). 3D view honoring TMA per-block padding.
            from rtp_llm.models_py.modules.dsv4._swa_fp8_kv_insert_triton import (
                quantize_and_insert_k_cache,
            )

            pool_3d = self._pool_view_3d_fp8(SWA_KV)
            assert (
                pool_3d is not None
            ), "FP8 SWA pool view unavailable in _prefill_write_swa_to_pool"
            quantize_and_insert_k_cache(
                source_flat.to(torch.bfloat16), pool_3d, slot_mapping
            )
        else:
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
            if kv_at is not None and self._kv_cache_is_fp8:
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
            if self._kv_cache_is_fp8:
                kv_view = self._pool_view_3d_fp8(INDEXER_KV)
            else:
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
        if self._kv_cache_is_fp8:
            pool_3d = self._pool_view_3d_fp8(attn_type)
            if pool_3d is None or pool_3d.shape[-1] != _DSV4_FP8_KV_ENTRY_BYTES:
                return None
            from rtp_llm.models_py.modules.dsv4._swa_fp8_dequant_triton import (
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
        from rtp_llm.models_py.modules.dsv4.decode.sparse_attn_decode_op import (
            SparseAttnV4DecodeOp,
        )

        bsz, q_len, _ = x.size()
        assert q_len == 1, "Phase 2: q_len==1 only (MTP/spec-decode is later)"
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        device = x.device
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
        # KV path (single MQA head)
        kv = fused_rmsnorm_rope(
            self._lin(self.wkv, x),
            self.kv_norm,
            freqs_cis_per_req,
            rd,
            eps=self.eps,
        )
        # Phase E5b: direct SWA pool write (register_buffer retired).
        kv_flat = kv.reshape(bsz * q_len, self.head_dim)  # [T, head_dim]
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        swa_pool_slots = attn_metadata.pool_write_slot_mappings.get(SWA_KV)
        swa_view = self._pool_view(SWA_KV)
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
                    self.indexer.forward_decode_vectorized(
                        x,
                        qr,
                        start_pos,
                        topk_buf_cmp,
                    )
                else:
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
                    self.compressor.forward_decode_vectorized(x, start_pos)
                else:
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
                if use_vec:
                    self.compressor.forward_decode_vectorized(x, start_pos)
                else:
                    self.compressor.forward_decode(x, start_pos)
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
        swa_view_3d_fp8 = (
            self._pool_view_3d_fp8(SWA_KV) if self._kv_cache_is_fp8 else None
        )

        # Decide which paged read variant to use, if any.
        use_paged_swa_read = (
            not self.compress_ratio
            and (swa_view_cache is not None or swa_view_3d_fp8 is not None)
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
        cmp_view_3d_fp8 = (
            self._pool_view_3d_fp8(cmp_attn_type)
            if (self._kv_cache_is_fp8 and cmp_attn_type is not None)
            else None
        )
        use_paged_dual_read = (
            self.compress_ratio in (4, 128)
            and (swa_view_cache is not None or swa_view_3d_fp8 is not None)
            and (cmp_view_cache is not None or cmp_view_3d_fp8 is not None)
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
            swa_global = translate_local_to_global_slots(
                req_id,
                swa_pool_bt[:bsz],
                swa_local,
                swa_eb,
            )
            if use_paged_swa_read and swa_view_3d_fp8 is not None:
                # FP8 path: FlashMLA reads the packed FP8 pool directly
                # (block_table + per-request virtual indices). swa_abs_idx
                # already holds per-request abs positions with -1 padding,
                # which is the FlashMLA `indices` contract.
                from rtp_llm.models_py.modules.dsv4.decode.fp8_sparse_attn_decode_op import (
                    SparseAttnV4DecodeFp8Op,
                )

                fp8_sparse_op = SparseAttnV4DecodeFp8Op(
                    n_heads=self.n_heads,
                    head_dim=self.head_dim,
                    softmax_scale=self.softmax_scale,
                )
                indices_fp8 = (
                    attn_metadata.swa_abs_idx[:bsz].to(torch.int32).contiguous()
                )
                cache_seqlens_fp8 = (attn_metadata.start_pos[:bsz] + 1).to(torch.int32)
                o = fp8_sparse_op.forward(
                    q,
                    swa_view_3d_fp8,
                    self.attn_sink,
                    indices_fp8,
                    cache_seqlens=cache_seqlens_fp8,
                    block_table=swa_pool_bt[:bsz].to(torch.int32).contiguous(),
                )
            elif use_paged_swa_read:
                # Zero-copy: pool view fed straight to TileLang kernel.
                q_packed = (
                    q.transpose(0, 1).contiguous() if q_len > 1 else q.transpose(0, 1)
                )
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
                cmp_global = translate_local_to_global_slots(
                    req_id,
                    cmp_pool_bt[:bsz],
                    cmp_local,
                    cmp_eb,
                )
                assert q_len == 1, (
                    "Phase 2B-2b dual-pool paged read currently supports "
                    f"q_len=1 only (got {q_len})"
                )
                if swa_view_3d_fp8 is not None and cmp_view_3d_fp8 is not None:
                    # FP8 dual-pool: per-slot dequant via existing kernel
                    # then concat → bf16 packed buffer for sparse_attn.
                    from rtp_llm.models_py.modules.dsv4._swa_fp8_dequant_triton import (
                        dequantize_slots_to_bf16,
                    )

                    swa_global_flat = swa_global.reshape(-1)
                    cmp_global_flat = cmp_global.reshape(-1)
                    swa_bf16 = dequantize_slots_to_bf16(
                        swa_view_3d_fp8, swa_global_flat
                    ).view(bsz, win, self.head_dim)
                    cmp_bf16 = dequantize_slots_to_bf16(
                        cmp_view_3d_fp8, cmp_global_flat
                    ).view(bsz, K_cmp, self.head_dim)
                    kv_packed = torch.cat([swa_bf16, cmp_bf16], dim=1)
                else:
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
                identity_topk = (
                    torch.arange(
                        win + K_cmp,
                        device=kv_packed.device,
                        dtype=torch.int32,
                    )
                    .view(1, 1, win + K_cmp)
                    .expand(bsz, q_len, win + K_cmp)
                    .contiguous()
                )
                o = sparse_op.forward(q, kv_packed, self.attn_sink, identity_topk)
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
        if o.is_cuda and o.numel() > 0:
            o_fp8, o_scale = fused_inv_rope_fp8_quant(
                o,
                freqs_cis_per_req,
                n_groups=self.n_groups,
                heads_per_group=self.n_heads // self.n_groups,
                nope_dim=self.head_dim - self.rope_head_dim,
                rope_head_dim=self.rope_head_dim,
            )
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

            all_reduce(out, Group.TP)
        return out

    # ==================================================================
    # Prefill — flat ``[T, dim]`` input, B==1 invariant.
    # ==================================================================
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        kv_cache: Optional[Any] = None,
        block_tables_by_type: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Prefill entry point.

        ``x``: flat ``[T, dim]`` (single-request, B==1 — enforced by
        the FIFO scheduler's ``max_context_batch_size=1`` setting and
        ``DeepSeekV4Model.forward``). ``start_pos``: scalar absolute
        position of the first new token (block.py extracts it once via
        ``int(positions[0].item())``). ``kv_cache`` and
        ``block_tables_by_type`` are stashed on ``self`` for the
        duration of the call so the many ``_prefill_*`` / pool helpers
        can resolve via ``self._kv_cache`` without threading the
        handles through every signature.
        """
        assert (
            x.dim() == 2
        ), f"DSv4 Attention prefill expects flat [T, dim]; got shape {tuple(x.shape)}"
        assert isinstance(
            start_pos, int
        ), f"DSv4 Attention prefill expects int start_pos; got {type(start_pos).__name__}"
        prev_kv = self._kv_cache
        prev_bt = self._block_tables_by_type
        if kv_cache is not None:
            self._kv_cache = kv_cache
        if block_tables_by_type is not None:
            self._block_tables_by_type = block_tables_by_type
        try:
            self._set_compressor_pool_context()
            try:
                return self._forward_prefill(x, start_pos)
            finally:
                self._clear_compressor_pool_context()
        finally:
            self._kv_cache = prev_kv
            self._block_tables_by_type = prev_bt

    def _forward_prefill(self, x: torch.Tensor, start_pos) -> torch.Tensor:
        """Prefill body. ``x`` is flat ``[T, dim]``; output flat ``[T, dim]``.

        Pipeline:
          1. ``_prefill_common_setup`` → freqs_cis + window topk + per-call
             scalars / debug closure (everything that's the same across
             the rest of the body).
          2. ``_prefill_compute_qkv`` → qr / q / kv / kv_full (CP all-gather).
          3. ``_compute_compress_idxs`` (HCA/CSA layers only) → extend
             topk_idxs with compressed-block tail.
          4. ``_prefill_write_swa_to_pool`` → write fresh SWA KV (BF16/FP8
             internal dispatch).
          5. Compressor → write compressed pool (BF16/FP8 internal).
          6. ``_build_kv_cat`` → ``[sliding | compressed]`` BF16 view
             consumed by sparse attn (FP8 reads dequant from pool).
          7. ``_prefill_attn`` → ``flash_mla_sparse_fwd`` (FP8) or
             tilelang/_sparse_attn (BF16).
          8. ``_prefill_output_proj`` → inverse-RoPE + wo_a + wo_b + AR.
        """
        assert self._kv_cache_is_fp8, (
            "_forward_prefill is FP8-only; BF16 KV-cache prefill is no "
            "longer supported on this branch"
        )
        common = self._prefill_common_setup(x, start_pos)
        qkv = self._prefill_compute_qkv(x, common)

        # SWA pool write — single dispatch point regardless of attention
        # type. FP8 paged-tail Triton kernel (pre-built slot_mapping in
        # swa_meta). Safe to do before attention because new K (abs pos
        # [sp, sp+S)) and any cont-prefill prefix tail (abs pos
        # [sp-P, sp)) target disjoint slots.
        self._prefill_write_swa_fp8_paged(common, qkv.kv_full)

        # FP8 SWA-only fast path: skip the kv_cat / sparse_attn flow
        # entirely. Cold/warmup attends over the BF16 ``kv_full``
        # directly; continuation builds [prefix_tail | new_K_bf16] in a
        # workspace and runs flash_mla_sparse_fwd over it.
        if self.compress_ratio == 0:
            if common.sp_int == 0 or self._kv_cache is None:
                o = self._attn_fp8_swa_via_kv_full(qkv, common)
            else:
                o = self._attn_fp8_swa_via_concat(qkv, common)
            out_3d = self._prefill_output_proj(o, common)
            return out_3d.squeeze(0)

        topk_idxs = common.topk_idxs
        if self.compress_ratio:
            compress_idxs = self._compute_compress_idxs(x, qkv.qr, common, qkv)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
        topk_idxs = topk_idxs.long()

        kv_full_3d = qkv.kv_full.unsqueeze(0)  # [1, T, head_dim] for pool helpers
        # Continuation prefill rebuilds the dense absolute SWA view from
        # the pool (prefix + new K) so the cont-prefill read can see the
        # full window.
        prefill_swa_dense_for_attn = None
        if common.any_cont:
            prefill_swa_dense_for_attn = self._prefill_read_swa_dense_abs_from_pool(
                1,
                start_pos,
                common.row_seqlens_full,
                common.seqlen_full + common.sp_int,
                current_kv_full=kv_full_3d,
            )

        kv_compress = None
        if self.compress_ratio:
            kv_compress = self.compressor(x.unsqueeze(0), start_pos)

        kv_cat = self._build_kv_cat(
            qkv,
            kv_compress,
            common,
            prefill_swa_dense_for_attn,
            start_pos,
            common.row_seqlens_full,
        )

        o = self._prefill_attn(qkv.q.unsqueeze(0), kv_cat, topk_idxs, common)

        out_3d = self._prefill_output_proj(o, common)
        return out_3d.squeeze(0)

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
        self, x: torch.Tensor, start_pos: int
    ) -> "PrefillMeta":
        """Build the layer-invariant (within compress_ratio bucket) part
        of per-call prefill metadata. All host-side prep work that
        doesn't depend on ``self.layer_id`` lives here so the upper layer
        can run it once per ratio and broadcast to every same-ratio
        attention via :meth:`_set_prefill_meta_shared`. Standalone path
        falls back to running this per-layer.
        """
        seqlen = int(x.shape[0])
        rd = self.rope_head_dim
        device = x.device

        cp_ctx = self._cp_ctx
        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1

        # CP rank-local view overrides the per-request start_pos. Otherwise
        # ``forward`` guarantees ``start_pos: int`` (block.py extracts the
        # scalar via ``int(positions[0].item())`` for B==1 prefill).
        sp_int = int(cp_ctx.prefix_length) if cp_on else start_pos
        any_cont = sp_int > 0

        if cp_on:
            freqs_cis = cp_freqs_cis_local(self.freqs_cis, cp_ctx)
        else:
            freqs_cis = self.freqs_cis[sp_int : sp_int + seqlen]

        # Bind freqs_cis to this layer's compressor / indexer chain
        # (idempotent — safe to call from both standalone and meta-broadcast paths).
        self._ensure_freqs_cis_bound()

        win = self.window_size
        seqlen_full = cp_ctx.seq_len_full if cp_on else seqlen
        if cp_on:
            topk_idxs = _get_window_topk_idxs_cp(
                win,
                1,
                seqlen_full,
                cp_ctx.global_positions,
            )
        else:
            topk_idxs = _get_window_topk_idxs(win, 1, seqlen, sp_int, device)

        # row_seqlens_full: [1] long tensor. Reused by SWA pool read/write
        # helpers (BF16 path) — they refuse a None for the per-row seqlens.
        row_seqlens_full = torch.tensor([seqlen_full], device=device, dtype=torch.long)

        swa_meta: Optional[SwaPrefillMeta] = None
        if self._kv_cache_is_fp8:
            swa_meta = self._build_swa_prefill_meta(seqlen, sp_int, device)

        indexer_meta: Optional[Any] = None
        if self.indexer is not None:
            from rtp_llm.models_py.modules.dsv4.attn_type import INDEXER_KV
            from rtp_llm.models_py.modules.dsv4.indexer_fp8 import IndexerFP8

            if isinstance(self.indexer, IndexerFP8):
                # Pass INDEXER_KV block_table + entries_per_block directly
                # so prepare() doesn't depend on a prior set_pool_context
                # call. Both default to ``None``/``0`` during warmup, which
                # IndexerFP8.prepare handles by emitting an empty
                # ``block_table_i32`` (matching the warmup short-circuit
                # in IndexerFP8.forward).
                idx_bt = (
                    self._block_tables_by_type.get(INDEXER_KV)
                    if self._block_tables_by_type is not None
                    else None
                )
                idx_eb = self._pool_entries_per_block(INDEXER_KV)
                indexer_meta = self.indexer.prepare(
                    bsz=1,
                    seqlen=seqlen,
                    sp_int=sp_int,
                    device=device,
                    kv_block_table=idx_bt,
                    kv_eb=idx_eb,
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
            swa_meta=swa_meta,
            indexer_meta=indexer_meta,
        )

    def _prefill_common_setup(self, x: torch.Tensor, start_pos: int) -> PrefillMeta:
        """Return the (possibly upper-layer-injected) shared prefill meta.
        Standalone callers (no upper-layer broadcast) build it per-layer here.
        """
        meta = self._prefill_meta_shared
        if meta is None:
            meta = self._build_shared_prefill_meta(x, start_pos)
        return meta

    def _build_swa_prefill_meta(
        self, seqlen: int, sp_int: int, device: torch.device
    ) -> SwaPrefillMeta:
        """Build the per-call FP8 prefill metadata bundle (B==1 invariant).

        Group 1 (write meta): always populated for FP8 layers when the
        SWA pool is bound. ``None`` on warmup forward.

        Group 2 (attention meta): only populated for SWA-only layers
        (``compress_ratio == 0``). ``topk_length_kv_full`` is cache-
        independent so it's set even on warmup (so the warmup path
        can consume it without a fallback). ``cache_*`` / ``combined_*``
        only on continuation prefill (``sp > 0``).
        """
        from rtp_llm.models_py.modules.dsv4 import _swa_prefill_ops_triton as _swa_ops
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV

        bsz = 1
        win = self.window_size
        num_tokens = bsz * seqlen
        is_swa_only = self.compress_ratio == 0

        topk_length_kv_full: Optional[torch.Tensor] = None
        if is_swa_only:
            positions = torch.arange(num_tokens, device=device, dtype=torch.int32)
            topk_length_kv_full = torch.clamp(positions + 1, max=win)

        slot_mapping: Optional[torch.Tensor] = None
        query_start_loc: Optional[torch.Tensor] = None
        combined_seq_lens: Optional[torch.Tensor] = None
        combined_gather_lens: Optional[torch.Tensor] = None
        combined_gather_len_max = 0
        M = 0
        cache_seq_lens: Optional[torch.Tensor] = None
        cache_gather_lens: Optional[torch.Tensor] = None
        prefix_len_max = 0
        combined_indices: Optional[torch.Tensor] = None
        combined_lens: Optional[torch.Tensor] = None

        bt = (
            self._block_tables_by_type.get(SWA_KV)
            if self._block_tables_by_type is not None
            else None
        )
        eb = self._pool_entries_per_block(SWA_KV)
        if self._kv_cache is not None and bt is not None and bt.numel() > 0 and eb > 0:
            seq_total = sp_int + seqlen
            query_start_loc = torch.tensor(
                [0, num_tokens], device=device, dtype=torch.int32
            )
            combined_seq_lens = torch.tensor(
                [seq_total], device=device, dtype=torch.int32
            )
            bt_swa = bt[:bsz].to(device=device, dtype=torch.int32).contiguous()
            slot_mapping = _swa_ops.compute_swa_slot_mapping(
                block_table=bt_swa,
                query_start_loc=query_start_loc,
                seq_lens=combined_seq_lens,
                block_size=eb,
                num_tokens=num_tokens,
            )

            if is_swa_only:
                combined_gather_lens = _swa_ops.compute_prefill_gather_lens(
                    seq_lens=combined_seq_lens,
                    query_start_loc=query_start_loc,
                    num_prefills=bsz,
                    num_decodes=0,
                    window_size=win,
                )
                # B==1: gather_len = query_len + min(prefix, win-1) by formula;
                # avoid an .item() sync on combined_gather_lens.max().
                combined_gather_len_max = seqlen + min(sp_int, win - 1)
                M = max(combined_gather_len_max, 1)

                if sp_int > 0:
                    prefix_len = min(sp_int, win - 1)
                    cache_seq_lens = torch.tensor(
                        [sp_int], device=device, dtype=torch.int32
                    )
                    cache_gather_lens = torch.tensor(
                        [prefix_len], device=device, dtype=torch.int32
                    )
                    prefix_len_max = prefix_len

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
        qr = self._rmsnorm_weighted(
            self._lin(self.wq_a, x_3d), self.q_norm
        )  # [1, T, q_lora_rank]
        q = self._lin(self.wq_b, qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = fused_rmsnorm_rope(q, None, common.freqs_cis, rd, eps=self.eps)

        # KV path (single MQA head) — rank-local under CP.
        kv_in = self._lin(self.wkv, x_3d)
        kv = fused_rmsnorm_rope(kv_in, self.kv_norm, common.freqs_cis, rd, eps=self.eps)

        if common.cp_on:
            kv_full = cp_all_gather_full(
                kv, common.cp_ctx
            )  # [1, seq_len_full, head_dim]
        else:
            kv_full = kv

        return PrefillQKV(
            qr=qr.squeeze(0),
            q=q.squeeze(0),
            kv_full=kv_full.squeeze(0),
        )

    # ------------------------------------------------------------------
    # Compressed-block topk (HCA / CSA layers)
    # ------------------------------------------------------------------
    def _compute_compress_idxs(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        common: PrefillMeta,
        qkv: PrefillQKV,
    ) -> torch.Tensor:
        """Build the compressed-block topk tail concatenated to the SWA topk.

        Layouts:
          * fresh prefill: KV is ``[sliding (kv_full) | compressed tail]``
            so the compressed offset is ``seqlen_full``.
          * continuation prefill: dense absolute SWA view of length
            ``sp + seqlen``, then compressed tail; offset is the same.
        """

        ratio = self.compress_ratio
        seqlen = common.seqlen
        device = common.device
        offset = common.seqlen_full + common.sp_int  # absolute SWA stream end

        if self.indexer is not None:
            from rtp_llm.models_py.modules.dsv4.indexer_fp8 import IndexerFP8

            if isinstance(self.indexer, IndexerFP8):
                # IndexerFP8._compute_indexer_q feeds q to apply_rotary_emb
                # which only handles the 4D q layout correctly; pass 3D
                # ``[1, T, dim]`` so q ends up ``[1, T, H, D]``.
                # ``common.indexer_meta`` was prebuilt in
                # ``_prefill_common_setup`` (same isinstance gate).
                raw = self.indexer(x.unsqueeze(0), qr.unsqueeze(0), common.indexer_meta)
                # Indexer returns int32 raw compressed-pool offsets
                # with -1 past the per-row valid count; rebase to
                # global sparse-attn coords + cast to long for cat.
                compress_idxs = torch.where(raw >= 0, raw + offset, raw).long()
            else:
                compress_idxs = self.indexer(
                    x.unsqueeze(0), qr.unsqueeze(0), common.sp_int, offset
                )
            # Indexer outputs may be 2D [T, K] (FP8 flat) or 3D [B, S, K].
            if compress_idxs.dim() == 2:
                compress_idxs = compress_idxs.unsqueeze(0)
            return compress_idxs

        if common.cp_on:
            return _get_compress_topk_idxs_cp(
                ratio, 1, common.seqlen_full, offset, common.cp_ctx.global_positions
            )
        return _get_compress_topk_idxs(ratio, 1, seqlen, common.sp_int, offset, device)

    # ------------------------------------------------------------------
    # KV concat / gather
    # ------------------------------------------------------------------
    def _build_kv_cat(
        self,
        qkv: PrefillQKV,
        kv_compress: Optional[torch.Tensor],
        common: PrefillMeta,
        prefill_swa_dense_for_attn: Optional[torch.Tensor],
        start_pos,
        row_seqlens_full: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the ``[sliding | compressed]`` BF16 KV tensor consumed
        by sparse_attn (3D ``[1, T_kv, head_dim]``).

        - Fresh prefill (sp == 0): use ``kv_full``; if compressor returned
          a bf16 tail, concat it. (FP8 compressor returns ``None`` —
          re-read the just-written suffix from the pool to materialize a
          bf16 tail for the kv_cat.)
        - Continuation prefill (sp > 0): always read SWA + compressed back
          from the framework pool (BF16 read, FP8 dequant — handled inside
          ``_prefill_paged_read_kv``).
        """
        seqlen = common.seqlen
        sp_int = common.sp_int
        ratio = self.compress_ratio
        kv_full_3d = qkv.kv_full.unsqueeze(0)
        prefill_swa_dense_len = common.seqlen_full + sp_int  # absolute end

        # FP8 compressor handled the pool write itself; recover a bf16
        # tail for the fresh-prefill concat by dequant-reading the
        # just-written suffix.
        fp8_compressor_handled_pool_write = False
        if self.compress_ratio and kv_compress is None and self._kv_cache_is_fp8:
            from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, HCA_KV

            cmp_at = CSA_KV if ratio == 4 else HCA_KV
            cmp_write_start = sp_int // ratio
            NB_total = (sp_int + seqlen) // ratio
            NB_new = NB_total - cmp_write_start
            if NB_new > 0:
                full_range = self._prefill_paged_read_kv(
                    cmp_at, 1, NB_total, self.head_dim, torch.bfloat16, common.device
                )
                if full_range is not None:
                    kv_compress = full_range[
                        :, cmp_write_start:NB_total, :
                    ].contiguous()
                    fp8_compressor_handled_pool_write = True

        # BF16 compressor wrote nothing — mirror its tail into the
        # CSA/HCA pool (FP8 already handled).
        if (
            self.compress_ratio
            and kv_compress is not None
            and not fp8_compressor_handled_pool_write
        ):
            from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, HCA_KV

            cmp_at = CSA_KV if ratio == 4 else HCA_KV
            cmp_write_start = sp_int // ratio
            self._prefill_paged_write_kv_range(cmp_at, kv_compress, 1, cmp_write_start)

        if not common.any_cont:
            if kv_compress is not None:
                return torch.cat([kv_full_3d, kv_compress], dim=1)
            return kv_full_3d

        # Continuation prefill: gather SWA + compressed dense view from pool.
        cmp_T = (common.seqlen_full + sp_int) // ratio if self.compress_ratio else None
        kv_cat = self._gather_kv_cache_dense_from_pool(
            1,
            start_pos,
            row_seqlens_full,
            swa_dense_len=prefill_swa_dense_len,
            swa_dense_override=prefill_swa_dense_for_attn,
            cmp_T=cmp_T,
        )
        assert kv_cat is not None, (
            "DSv4 continuation prefill requires paged ctx (pass kv_cache + "
            "block_tables_by_type to Attention.forward via V4Transformer)."
        )
        # Overlay freshly compressed K into the cmp tail (matches BF16
        # round-trip when sp==0 rows were just written above).
        if self.compress_ratio and kv_compress is not None:
            cmp_base = prefill_swa_dense_len + (sp_int // ratio)
            cmp_end = min(cmp_base + kv_compress.shape[1], kv_cat.shape[1])
            if cmp_end > cmp_base:
                kv_cat[:, cmp_base:cmp_end] = kv_compress[:, : cmp_end - cmp_base].to(
                    kv_cat.dtype
                )
        return kv_cat

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
        from rtp_llm.models_py.modules.dsv4 import _swa_fp8_kv_insert_triton as _ins
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV

        meta = common.swa_meta
        if meta is None or meta.slot_mapping is None:
            return
        packed_3d = self._pool_view_3d_fp8(SWA_KV)
        if packed_3d is None:
            return

        k_bf16 = kv_full.reshape(-1, self.head_dim)
        if k_bf16.dtype != torch.bfloat16:
            k_bf16 = k_bf16.to(torch.bfloat16)
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

        o3, _, _ = flash_mla_sparse_fwd(
            q=qkv.q,
            kv=qkv.kv_full.unsqueeze(1),
            indices=common.topk_idxs.squeeze(0).unsqueeze(1).to(torch.int32),
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
        """Continuation prefill (sp>0): prefix-from-cache + new-K-bf16 concat.

        Pipeline:
          1. ``dequantize_and_gather_k_cache`` reads only the trailing
             ``P = min(sp, win-1)`` cached tokens (the SWA prefix tail,
             abs pos ``[sp - P, sp)``) into ``workspace[:, :P, :]``.
             Caller must have written new K via
             ``_prefill_write_swa_fp8_paged`` *before* this method runs;
             new K targets abs pos ``[sp, sp+S)``, distinct slot_mapping
             from prefix tail, so prefix read is safe.
          2. BF16-copy ``kv_full`` (the freshly computed new K) into
             ``workspace[:, P:P+S, :]``.
          3. ``flash_mla_sparse_fwd`` over the concat workspace using
             ``combined_indices`` from ``combine_topk_swa_indices``
             (TOP_K=0, N=0 ⇒ SWA-only rows; indices map abs positions
             ``[sp - P, sp + S)`` onto workspace ``[0, P+S)``).

        Avoids the legacy round-trip's 2*eb capacity wall: cache only
        stores prefix tail (``P <= win-1``, fits in 2*eb=512), and
        new K stays BF16 in the workspace — no FP8 quant loss either.
        """
        from flash_mla import flash_mla_sparse_fwd  # type: ignore[import-not-found]

        from rtp_llm.models_py.modules.dsv4 import _swa_fp8_dequant_triton as _swa_dq
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV

        meta = common.swa_meta
        assert (
            meta is not None
            and meta.combined_indices is not None
            and meta.cache_seq_lens is not None
            and meta.cache_gather_lens is not None
        ), (
            "via_concat path requires swa_meta with cache_* and combined_* "
            "fields (only built for sp>0 continuation prefill)"
        )

        bsz = 1
        bt = self._block_tables_by_type[SWA_KV][:bsz]
        packed_3d = self._pool_view_3d_fp8(SWA_KV)
        assert packed_3d is not None, "FP8 SWA pool unavailable"
        eb = int(packed_3d.shape[1])

        D = self.head_dim
        S = common.seqlen
        P = meta.prefix_len_max
        workspace = torch.zeros(
            (bsz, meta.M, D),
            dtype=torch.bfloat16,
            device=qkv.q.device,
        )
        bt_swa = bt.to(dtype=torch.int32)

        if P > 0:
            _swa_dq.dequantize_and_gather_k_cache(
                out=workspace,
                k_cache=packed_3d,
                seq_lens=meta.cache_seq_lens,
                gather_lens=meta.cache_gather_lens,
                block_table=bt_swa,
                block_size=eb,
                offset=0,
            )

        # B==1: kv_full is [T, D]; copy into workspace[0, P:P+S, :].
        workspace[:, P : P + S, :].copy_(qkv.kv_full.to(torch.bfloat16).unsqueeze(0))

        o3, _, _ = flash_mla_sparse_fwd(
            q=qkv.q,
            kv=workspace.view(-1, 1, D),
            indices=meta.combined_indices.unsqueeze(1),
            sm_scale=self.softmax_scale,
            attn_sink=self.attn_sink,
            topk_length=meta.combined_lens,
        )
        return o3.unsqueeze(0)

    # ------------------------------------------------------------------
    # Sparse attention dispatch
    # ------------------------------------------------------------------
    def _prefill_attn(
        self,
        q_3d: torch.Tensor,  # [1, T, H, D]
        kv_cat: torch.Tensor,  # [1, T_kv, D] bf16
        topk_idxs: torch.Tensor,  # [1, T, K] long
        common: PrefillMeta,
    ) -> torch.Tensor:
        """Sparse attention call — FP8 KV-cache layers go through FlashMLA's
        sparse_fwd (the FP8-aware kernel that's also fed BF16 inputs);
        BF16 KV-cache layers fall through to TileLang or the PyTorch
        reference implementation. Returns ``[1, T, H, D]`` BF16.
        """
        if self._kv_cache_is_fp8 and q_3d.is_cuda:
            from flash_mla import flash_mla_sparse_fwd  # type: ignore[import-not-found]

            topk_i32 = topk_idxs.to(torch.int32)
            topk_length = (
                (topk_i32 >= 0).sum(dim=-1).reshape(-1).to(torch.int32).contiguous()
            )
            B_TOPK = 128
            K = topk_i32.shape[-1]
            if K % B_TOPK != 0:
                pad = B_TOPK - (K % B_TOPK)
                topk_i32 = torch.nn.functional.pad(topk_i32, (0, pad), value=-1)
            o3, _, _ = flash_mla_sparse_fwd(
                q=q_3d.squeeze(0),
                kv=kv_cat.squeeze(0).unsqueeze(1),
                indices=topk_i32.squeeze(0).unsqueeze(1).contiguous(),
                sm_scale=self.softmax_scale,
                attn_sink=self.attn_sink,
                topk_length=topk_length,
            )
            return o3.unsqueeze(0)
        if _tl_kernels.tilelang_available():
            return _tl_kernels.sparse_attn(
                q_3d, kv_cat, self.attn_sink, topk_idxs, self.softmax_scale
            )
        return _sparse_attn(q_3d, kv_cat, self.attn_sink, topk_idxs, self.softmax_scale)

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
            o_3d = o.reshape(seqlen, self.n_heads, self.head_dim)
            if freqs_cis.dim() == 2:
                freqs_per_token = freqs_cis.contiguous()
            else:
                freqs_per_token = freqs_cis.reshape(seqlen, -1).contiguous()
            o_fp8, o_scale = fused_inv_rope_fp8_quant(
                o_3d,
                freqs_per_token,
                n_groups=self.n_groups,
                heads_per_group=self.n_heads // self.n_groups,
                nope_dim=self.head_dim - self.rope_head_dim,
                rope_head_dim=self.rope_head_dim,
            )
            o = self._wo_a_einsum_from_fp8(o_fp8, o_scale, 1, seqlen)
        else:
            apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
            o = o.reshape(1, seqlen, self.n_groups, -1)
            wo_a_bf16 = _fp8_dequant_to_fp32(self.wo_a_w, self.wo_a_s).to(o.dtype)
            wo_a = wo_a_bf16.view(self.n_groups, self.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        out = self._lin(self.wo_b, o.flatten(2))
        if self.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            all_reduce(out, Group.TP)
        return out
