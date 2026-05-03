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
from typing import Any, Dict, Optional, Tuple, Union

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
from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
    precompute_freqs_cis,
)
from rtp_llm.models_py.modules.dsv4.weight_loader import _repack_v4_fp8_scale_to_int32
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


def _v4_fp8_linear_from_dict(
    weights: dict,
    weight_key: str,
    scale_key: str,
):
    """Build a CudaFp8DeepGEMMLinear from V4 ckpt tensors.

    Repacks the UE8M0 float8_e8m0fnu scale into DeepGEMM's int32 layout
    in place in the weights dict so subsequent callers see the packed form.
    """
    s = weights.get(scale_key)
    assert s is not None, f"expected FP8 scale at {scale_key}"
    if s.dtype == torch.float8_e8m0fnu:
        s = _repack_v4_fp8_scale_to_int32(s)
        weights[scale_key] = s
    # Build via factory (CudaFp8DeepGEMMLinear matches FP8_PER_BLOCK +
    # float8_e4m3fn weight + int32 scale).
    return LinearFactory.create_linear_from_weights(
        weights,
        weight_key,
        scale_key,
        quant_config=_V4_FP8_BLOCK_CFG,
    )


class _NormHolder(nn.Module):
    """Wraps a BF16 norm-weight parameter so that ckpt key `.weight` matches.

    BF16 dtype is required by ``rtp_llm_ops.rmsnorm`` (silent NaN with fp32).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))


def _get_window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int, device
) -> torch.Tensor:
    """Returns int64 [bsz, seqlen, window_size] giving the (cyclic) absolute slot indices
    in the sliding-window KV ring buffer that each query position should read."""
    if start_pos > 0 and seqlen > 1:
        # Continuation prefill: per-position ring buffer indices
        # Each query at global position g = start_pos + i reads from ring buffer
        global_pos = torch.arange(
            start_pos, start_pos + seqlen, device=device
        )  # [seqlen]
        sp = global_pos % window_size  # [seqlen]
        offsets = torch.arange(window_size, device=device)  # [win]
        idxs = (
            sp.unsqueeze(1) + 1 + offsets.unsqueeze(0)
        ) % window_size  # [seqlen, win]
        valid_count = torch.clamp(global_pos + 1, max=window_size)  # [seqlen]
        invalid = offsets.unsqueeze(0) < (window_size - valid_count.unsqueeze(1))
        matrix = torch.where(invalid, -1, idxs)
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
    seq_len_full: int,
    global_positions: torch.Tensor,
) -> torch.Tensor:
    """CP-prefill variant: each rank-local Q token at local index i sits
    at GLOBAL position g = global_positions[i].  Its sliding window
    reads KV at global positions [max(0, g-win+1), g+1), which — after
    the attention-side all-gather has stripped padding — live at
    indices [max(0, g-win+1), g+1) in the full uncompressed KV tensor
    ``kv[:, :seq_len_full]``.

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
    W = min(window_size, max(seq_len_full, 1))
    # Per-row window start, clamped to 0 for early Q positions whose
    # left edge would be negative.  Matches `_get_window_topk_idxs`'s
    # `(base - window_size + 1).clamp(0)` at start_pos == 0.
    base = global_positions.unsqueeze(1)  # [S_local, 1]
    window_start = (base - W + 1).clamp_min(0)  # [S_local, 1]
    offs = torch.arange(W, device=device)  # [W]
    matrix = window_start + offs  # [S_local, W] — valid kv indices left-aligned
    # Mask out positions that are causally future (matrix > g) or land
    # past the unpadded sequence end (matrix >= seq_len_full, which
    # protects rank-local padding slots whose global_position == padded_len).
    invalid = (matrix > base) | (matrix >= seq_len_full)
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
    if start_pos > 0:
        n = (start_pos + 1) // ratio
        matrix = (
            (torch.arange(0, n, device=device) + offset).unsqueeze(0).expand(seqlen, -1)
        )
    else:
        matrix = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
        mask = (
            matrix >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        )
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _get_compress_topk_idxs_cp(
    ratio: int,
    bsz: int,
    seq_len_full: int,
    offset: int,
    global_positions: torch.Tensor,
) -> torch.Tensor:
    """CP-prefill variant of the dense HCA compressed-KV index list
    (the branch used when no Indexer is present — compress_ratio == 128).
    Q at GLOBAL position g reads compressed KV blocks [0, (g+1)//ratio),
    which live at offsets [offset, offset + (g+1)//ratio) inside the
    attention-side concatenated [sliding | compressed] tensor.  Return
    shape [bsz, S_local, seq_len_full // ratio]."""
    device = global_positions.device
    S_local = int(global_positions.shape[0])
    T_comp = max(seq_len_full // ratio, 0)
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
        weights: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
        tp_size: int = 1,
        tp_rank: int = 0,
        kv_cache_dtype: Any = None,
    ):
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
        self._factory_mode = weights is not None
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

        if self._factory_mode:
            # Q / KV / O — FP8 linears go through LinearFactory →
            # CudaFp8DeepGEMMLinear → DeepGEMM fp8_gemm_nt.
            def _fp8(name: str):
                return _v4_fp8_linear_from_dict(
                    weights,
                    f"{prefix}.{name}.weight",
                    f"{prefix}.{name}.scale",
                )

            def _fp8_sliced(
                name: str, row_slice: slice = None, col_slice: slice = None
            ) -> "torch.nn.Module":
                """Build a CudaFp8DeepGEMMLinear from a sliced view of the
                ckpt FP8 weight + UE8M0 block-128 scale.  Slices the scale
                along the *block-128* axis correspondingly: row_slice with
                stride/start divisible by 128 maps to a row_slice on the
                scale; same for col_slice."""
                wkey = f"{prefix}.{name}.weight"
                skey = f"{prefix}.{name}.scale"
                w = weights[wkey]
                s = weights[skey]
                if row_slice is not None:
                    rs = row_slice.start // 128
                    re = row_slice.stop // 128
                    w = w[row_slice]
                    s = s[rs:re]
                if col_slice is not None:
                    cs = col_slice.start // 128
                    ce = col_slice.stop // 128
                    w = w[:, col_slice]
                    s = s[:, cs:ce]
                # Local dict so the factory call sees the slice
                local = dict(weights)
                local[wkey] = w.contiguous()
                local[skey] = s.contiguous()
                return _v4_fp8_linear_from_dict(local, wkey, skey)

            self.wq_a = _fp8("wq_a")  # [q_lora, dim] — replicate
            # wq_b is row-split along N (n_heads * head_dim)
            self.wq_b = (
                _fp8_sliced("wq_b", row_slice=wq_b_row_slice)
                if tp_size > 1
                else _fp8("wq_b")
            )
            self.wkv = _fp8("wkv")  # MQA single KV head — replicate

            # wo_a grouped projection: row-split along (n_groups*o_lora_rank).
            # Stays on QuantizedLinear (grouped einsum, no factory equivalent yet).
            assert (n_heads * head_dim) % o_groups == 0
            self.wo_a = QuantizedLinear(
                n_heads_local * head_dim // n_groups_local,
                n_groups_local * o_lora_rank,
                storage="fp8",
            )
            with torch.no_grad():
                wo_a_w = weights[f"{prefix}.wo_a.weight"]
                wo_a_s = weights[f"{prefix}.wo_a.scale"]
                if tp_size > 1:
                    wo_a_w = wo_a_w[wo_a_row_slice].contiguous()
                    wo_a_s = wo_a_s[
                        wo_a_row_slice.start // 128 : wo_a_row_slice.stop // 128
                    ].contiguous()
                self.wo_a.weight = nn.Parameter(wo_a_w, requires_grad=False)
                self.wo_a.scale = nn.Parameter(wo_a_s, requires_grad=False)
                # Pre-stack into the ``fp8_einsum`` layout once.  Stored
                # as plain buffers so they don't enter nn.Module state_dict
                # (would confuse the ckpt loader — the raw wo_a.weight /
                # wo_a.scale are the real params).
                K_local = n_heads_local * head_dim // n_groups_local
                _stk_w, _stk_s = _prepare_wo_a_stacked(
                    wo_a_w, wo_a_s, n_groups_local, o_lora_rank, K_local
                )
                self.register_buffer("_wo_a_stk_w", _stk_w, persistent=False)
                self.register_buffer("_wo_a_stk_s", _stk_s, persistent=False)

            # wo_b row-split along K (cols), all_reduce after forward
            self.wo_b = (
                _fp8_sliced("wo_b", col_slice=wo_b_col_slice)
                if tp_size > 1
                else _fp8("wo_b")
            )

            # Non-quantized params copy straight from the dict.
            self.q_norm = _NormHolder(q_lora_rank)
            self.q_norm.weight = nn.Parameter(
                weights[f"{prefix}.q_norm.weight"].to(torch.bfloat16),
                requires_grad=False,
            )
            self.kv_norm = _NormHolder(head_dim)
            self.kv_norm.weight = nn.Parameter(
                weights[f"{prefix}.kv_norm.weight"].to(torch.bfloat16),
                requires_grad=False,
            )
            attn_sink_full = weights[f"{prefix}.attn_sink"].float()
            self.attn_sink = nn.Parameter(
                (
                    attn_sink_full[attn_sink_slice].contiguous()
                    if tp_size > 1
                    else attn_sink_full
                ),
                requires_grad=False,
            )
        else:
            # Legacy meta-tensor + load_v4_safetensors path.
            self.wq_a = QuantizedLinear(dim, q_lora_rank, storage="fp8")
            self.q_norm = _NormHolder(q_lora_rank)
            self.wq_b = QuantizedLinear(q_lora_rank, n_heads * head_dim, storage="fp8")
            self.wkv = QuantizedLinear(dim, head_dim, storage="fp8")
            self.kv_norm = _NormHolder(head_dim)
            assert (n_heads * head_dim) % o_groups == 0
            self.wo_a = QuantizedLinear(
                n_heads * head_dim // o_groups, o_groups * o_lora_rank, storage="fp8"
            )
            self.wo_b = QuantizedLinear(o_groups * o_lora_rank, dim, storage="fp8")
            self.attn_sink = nn.Parameter(torch.empty(n_heads, dtype=torch.float32))

        assert (n_heads * head_dim) % o_groups == 0

        # Compressor + Indexer (only for compressed layers)
        if compress_ratio:
            self.compressor = Compressor(
                dim=dim,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                compress_ratio=compress_ratio,
                max_batch_size=max_batch_size,
                norm_eps=norm_eps,
                weights=weights,
                prefix=f"{prefix}.compressor" if self._factory_mode else "",
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
                    weights=weights,
                    prefix=f"{prefix}.indexer" if self._factory_mode else "",
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

    def _prefill_write_swa_to_pool(
        self,
        bsz: int,
        kv_full: torch.Tensor,
        sp: Union[int, torch.Tensor],
        row_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Phase E5b: direct SWA ring write to the framework SWA_KV pool
        (replaces the retired ``self.kv_cache[:bsz, :win]`` register
        _buffer ring-write + ``_prefill_paged_write_swa`` mirror pair).

        For each of a row's valid new tokens at local index i:
            global_pos[b] = sp[b] + src_start[b] + i
            ring_pos[b]   = global_pos[b] % win
            slot[b]       = bt[b, ring_pos[b] // eb] * eb + (ring_pos[b] % eb)

        Two write shapes depending on per-row seqlen vs win:
          * seqlen <= win: write all seqlen tokens (each to a unique
            ring slot).
          * seqlen  > win: the first ``seqlen - win`` tokens would have
            been overwritten by the ring — skip them and only write the
            last ``win`` tokens (each to a unique ring slot).  Matches
            the retired ring-write's final state without relying on the
            non-deterministic behavior of ``index_copy_`` under
            duplicate slot_mapping entries.

        Supports ``bsz >= 1``.  ``sp`` accepts scalar (legacy) or ``[B]``
        int64 tensor.  ``row_seqlens`` (optional ``[B]``) marks the valid
        new-token count per row; ``None`` means ``kv_full.shape[1]`` for
        every row.  Per-row rows with ``n_write[b] < max_n_write`` emit
        sentinel ``-1`` slots past their valid range so write_kv_to_pool
        skips them.  bsz==1 + scalar ``sp`` + ``row_seqlens is None``
        produces byte-equal slot_mapping to the historical scalar path.
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

        win = self.window_size
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

        # Per-row src_start / n_write.  seqlen <= win => write all; seqlen > win
        # => write last win only.
        n_write_per_row = torch.clamp(seq_t, max=win)  # [B]
        src_start_per_row = seq_t - n_write_per_row  # [B] >= 0
        max_n_write = int(n_write_per_row.max().item()) if bsz > 0 else 0
        if max_n_write == 0:
            return

        j = torch.arange(max_n_write, device=device, dtype=torch.long)  # [max_n_write]
        # Row-local validity: j < n_write[b] -> valid position.
        row_valid = j.unsqueeze(0) < n_write_per_row.unsqueeze(1)  # [B, max_n_write]
        # Global positions per (row, j): sp[b] + src_start[b] + j.
        global_pos = (
            sp_t.unsqueeze(1) + src_start_per_row.unsqueeze(1) + j.unsqueeze(0)
        )  # [B, max_n_write]
        ring_pos = global_pos % win
        block_in_seq = ring_pos // eb
        in_block = ring_pos % eb
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

        # Gather source rows: src[b, j] = kv_full[b, src_start[b] + j].
        src_pos = src_start_per_row.unsqueeze(1) + j.unsqueeze(0)  # [B, max_n_write]
        # Clamp to [0, max_seqlen-1] for safe gather; invalid positions are
        # masked by slot == -1 so the gathered value doesn't matter.
        src_pos_safe = src_pos.clamp(min=0, max=max(max_seqlen - 1, 0))
        gather_idx = src_pos_safe.unsqueeze(-1).expand(-1, -1, self.head_dim)
        source = torch.gather(kv_full[:bsz], 1, gather_idx)  # [B, max_n_write, D]
        source_flat = source.reshape(bsz * max_n_write, self.head_dim)
        slot_mapping = slot.reshape(-1)
        write_kv_to_pool(source_flat, slot_mapping, pool_view, mask_negative=True)

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

    def _gather_kv_cache_dense_from_pool(self, bsz: int) -> Optional[torch.Tensor]:
        """Phase E1: reconstruct the ``[bsz, kv_cache_size, head_dim]``
        dense tensor that ``self.kv_cache[:bsz]`` presents, but sourced
        from the framework pools instead of the register_buffer mirror.

        Layout (matches register_buffer):
          ``[:, :win, :]``        -- SWA_KV pool (ring-buffered)
          ``[:, win:win+T_cmp, :]`` -- CSA_KV or HCA_KV pool (compressed)

        Returns ``None`` when ctx not bound — caller falls back to
        register_buffer.  SWA-only layers (compress_ratio == 0) get a
        bare ``[bsz, win, hd]`` read.
        """
        from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, HCA_KV, SWA_KV

        if self._kv_cache is None or self._block_tables_by_type is None:
            return None
        win = self.window_size
        hd = self.head_dim
        dtype = torch.bfloat16
        device = self.freqs_cis.device
        T_cmp = (
            self.compressor._kv_cache_t
            if (self.compressor is not None and self.compress_ratio)
            else 0
        )

        swa_dense = self._prefill_paged_read_kv(SWA_KV, bsz, win, hd, dtype, device)
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
        if self._factory_mode and x.dim() > 2:
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
        rd = self.rope_head_dim
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
            self._lin(self.wq_a, x), self.q_norm.weight
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
            self.kv_norm.weight,
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
        # Always use the vectorized compressor/indexer decode variants —
        # the loop variants did per-request ``.item()`` D2H syncs which
        # serialize the GPU stream.  Vectorized is math-equivalent and
        # CUDA-graph-capturable.
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
                self.indexer.forward_decode_vectorized(
                    x,
                    qr,
                    start_pos,
                    topk_buf_cmp,
                )
                # Stitch indexer output into the metadata's topk_total
                # compressed half (with +win offset) — in-place on the view
                # so downstream consumers (sparse_op, refill paths) see the
                # stitched values.  The result is consumed via attn_metadata
                # later, so no local variable binding is needed here.
                topk_total_view = attn_metadata.topk_total_by_ratio[4][:bsz]
                idx_with_off = torch.where(
                    topk_buf_cmp >= 0,
                    topk_buf_cmp + win,
                    topk_buf_cmp,
                )
                topk_total_view[:, :, win:] = idx_with_off
            else:
                # HCA layer (ratio=128): dense topk lives in
                # ``attn_metadata.topk_total_by_ratio[ratio]`` already
                # (builder-side), so all we owe here is the compressor
                # side-effect — it self-binds/writes/scatters compressed-K
                # into the HCA_KV + HCA_STATE pool via set_pool_context
                # lifecycle.
                self.compressor.forward_decode_vectorized(x, start_pos)

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
            swa_global = translate_local_to_global_slots(
                req_id,
                swa_pool_bt[:bsz],
                swa_local,
                swa_eb,
            )

            if use_paged_swa_read:
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
        if self._factory_mode and o.is_cuda and o.numel() > 0:
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
            wo_a_bf16 = self.wo_a.dequant_weight(out_dtype=o.dtype)
            wo_a = wo_a_bf16.view(self.n_groups, self.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        out = self._lin(self.wo_b, o.flatten(2))
        if self.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

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
        for the existing ``_forward_prefill`` and squeezed on the way out
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
                out = self._forward_prefill(x_in, start_pos, sequence_lengths)
            finally:
                self._clear_compressor_pool_context()
        finally:
            self._kv_cache = prev_kv
            self._block_tables_by_type = prev_bt
        return out.squeeze(0) if was_flat else out

    def _compute_prefill_indices(
        self,
        start_pos,
        seqlen: int,
        bsz: int,
        device: torch.device,
        cp_ctx,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-token RoPE freqs + SWA-window topk indices for the current
        prefill call.  Two modes share the same dispatch axis:

        - CP enabled       : rank-local positions via ``cp_ctx``
        - scalar prefill   : contiguous ``[sp, sp+seqlen)``

        Both outputs depend only on positional metadata (start_pos,
        cp_ctx.global_positions), so it's safe to compute them at the
        top of the body before the Q/KV path.

        Caller guarantees ``bsz == 1`` (asserted in ``_forward_prefill``);
        batched prefill is gated by ``max_context_batch_size = 1`` in
        the FIFO scheduler — see ``_forward_prefill`` for details."""
        win = self.window_size
        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1
        if cp_on:
            freqs_cis = cp_freqs_cis_local(self.freqs_cis, cp_ctx)
            topk_idxs = _get_window_topk_idxs_cp(
                win,
                bsz,
                cp_ctx.seq_len_full,
                cp_ctx.global_positions,
            )
        else:
            sp = int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
            freqs_cis = self.freqs_cis[sp : sp + seqlen]
            topk_idxs = _get_window_topk_idxs(win, bsz, seqlen, sp, device)
        return freqs_cis, topk_idxs

    def _compute_compress_idxs(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos,
        seqlen: int,
        seqlen_full: int,
        bsz: int,
        device: torch.device,
        cp_ctx,
        _dbg: bool,
    ) -> torch.Tensor:
        """Compressed-block topk indices for HCA layers (``compress_ratio
        != 0``).  Concatenated prefill KV is ``[sliding (seqlen_full) |
        compressed tail]``, so the compressed-block start-index is
        ``seqlen_full``.

        Caller guarantees ``bsz == 1`` (asserted in ``_forward_prefill``).

        Branches by what's available:

        - ``self.indexer is not None`` (HCA layer): defer to the indexer
        - ``cp_on`` and no indexer: ``_get_compress_topk_idxs_cp``
        - otherwise (scalar prefill, no indexer):
          ``_get_compress_topk_idxs``
        """
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        ratio = self.compress_ratio
        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1
        offset = seqlen_full

        if self.indexer is not None:
            if _dbg:
                self.indexer._dbg_prefix = f"L{self.layer_id:02d}_attn_idx"
            compress_idxs = self.indexer(x, qr, start_pos, offset)
            if _dbg:
                self.indexer._dbg_prefix = None
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_compress_idxs",
                    compress_idxs,
                )
            return compress_idxs
        if cp_on:
            return _get_compress_topk_idxs_cp(
                ratio,
                bsz,
                seqlen_full,
                offset,
                cp_ctx.global_positions,
            )
        sp_int = int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
        return _get_compress_topk_idxs(ratio, bsz, seqlen, sp_int, offset, device)

    def _build_kv_cat(
        self,
        kv_full: torch.Tensor,
        kv_compress: Optional[torch.Tensor],
        any_cont: bool,
        bsz: int,
    ) -> torch.Tensor:
        """Assemble the ``[sliding | compressed]`` KV tensor consumed by
        sparse_attn.

        Two regimes (orthogonal to compress vs no-compress):

        - Fresh prefill (``any_cont == False``): use the just-computed
          ``kv_full``; if HCA produced a ``kv_compress`` tail, concat it.
        - Continuation prefill (any sp > 0): always read SWA + compressed
          back from the framework pool — by this point ``kv_full`` has
          already been written to the pool too, so the gather is byte-
          equal even for mixed batches with sp==0 / sp>0 rows.
        """
        if any_cont:
            kv_cat = self._gather_kv_cache_dense_from_pool(bsz)
            assert kv_cat is not None, (
                "Phase E5b: continuation prefill requires paged "
                "ctx (pass kv_cache=... + block_tables_by_type=... "
                "to Attention.forward via V4Transformer)."
            )
            return kv_cat
        if kv_compress is not None:
            return torch.cat([kv_full, kv_compress], dim=1)
        return kv_full

    def _forward_prefill(
        self, x: torch.Tensor, start_pos, sequence_lengths=None
    ) -> torch.Tensor:
        """Prefill-only forward.  Decode goes through ``forward_decode`` /
        ``_forward_decode_body`` with paged metadata; this body is entered
        only when ``seqlen > 1``.  ``start_pos`` is int or 1-element tensor.

        DSv4 TEMP — bsz==1 invariant: the FIFO scheduler is configured
        with ``max_context_batch_size = 1`` (see
        ``cpp/engine_base/schedulers/FIFOScheduler.cc::evaluateRunningMemory``)
        so only one prefill request enters per scheduling round.  All
        per-row "batched prefill" branches in the helpers were removed
        on this premise — re-enabling B>1 prefill requires reinstating
        them (and the per-row sequence_lengths threading) along with
        cu_seqlens-aware compressor / indexer / SWA-pool kernels."""
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        # Master switch: when MOEDBG=0 the AND short-circuits so neither the
        # layer_id compare nor any record_if_level call site below runs.
        _dbg = _rt.ENABLED and self.layer_id <= 2

        def _rec(tag: str, t):
            if _dbg:
                _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_{tag}", t)

        bsz, seqlen, _ = x.size()
        # DSv4 TEMP: see docstring — prefill kernels assume bsz==1.
        assert bsz == 1, (
            f"DSv4 prefill expects bsz==1; got {bsz}.  Set "
            "MAX_CONTEXT_BATCH_SIZE=1 (default) so the scheduler does "
            "not batch multiple prefill requests in one round."
        )
        rd = self.rope_head_dim
        device = x.device
        device = x.device

        cp_ctx = self._cp_ctx
        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1

        # Positional metadata: per-token RoPE freqs + SWA-window topk.
        # Both depend only on start_pos / cp_ctx, so pre-compute here
        # before the Q/KV path.
        freqs_cis, topk_idxs = self._compute_prefill_indices(
            start_pos, seqlen, bsz, device, cp_ctx
        )

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
        qr = self._rmsnorm_weighted(
            self._lin(self.wq_a, x), self.q_norm.weight
        )  # [B, S, q_lora_rank]
        _rec("qr_norm", qr)
        q = self._lin(self.wq_b, qr).unflatten(-1, (self.n_heads, self.head_dim))
        _rec("q_pre_rmsnorm", q)
        # Per-head QK RMSNorm (no learnable scale, per official code) + partial
        # RoPE, single Triton launch.  ``fused_rmsnorm_rope`` handles both
        # prefill (``freqs_cis`` per-position ``[S, rd/2]``) and batched-
        # decode-in-prefill (``[B, 1, rd/2]``) via the unified freq_stride.
        q = fused_rmsnorm_rope(q, None, freqs_cis, rd, eps=self.eps)
        _rec("q_post_rope", q)
        _rec("freqs_cis", freqs_cis)

        # KV path (single KV head) — rank-local under CP.
        kv_in = self._lin(self.wkv, x)
        kv = fused_rmsnorm_rope(kv_in, self.kv_norm.weight, freqs_cis, rd, eps=self.eps)
        _rec("kv_post_rope_local", kv)

        # Under CP prefill, all-gather KV across the CP (== TP) group
        # and strip padding so every rank has the FULL uncompressed
        # sliding KV in logical order; attention then runs with rank-
        # local Q × full-KV.
        if cp_on:
            kv_full = cp_all_gather_full(kv, cp_ctx)  # [1, seq_len_full, head_dim]
            seqlen_full = cp_ctx.seq_len_full
        else:
            kv_full = kv
            seqlen_full = seqlen
        _rec("kv_full", kv_full)

        # HCA layers extend topk_idxs with the compressed-block tail.
        if self.compress_ratio:
            compress_idxs = self._compute_compress_idxs(
                x,
                qr,
                start_pos,
                seqlen,
                seqlen_full,
                bsz,
                device,
                cp_ctx,
                _dbg,
            )
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
        topk_idxs = topk_idxs.long()

        # any_cont drives the kv_cat fork: fresh prefill (sp==0) uses
        # the just-computed kv_full; continuation prefill (sp>0) gathers
        # SWA + compressed back from the framework pool.  bsz==1 is
        # enforced above, so a 1-element tensor reduces to a scalar
        # comparison.
        sp_int = (
            int(start_pos) if isinstance(start_pos, torch.Tensor) else int(start_pos)
        )
        any_cont = sp_int > 0

        # Phase E5b: direct SWA pool write from kv_full (no register_buffer
        # intermediary).  Ring-buffered addressing handled inside
        # ``_prefill_write_swa_to_pool``; prefix KV for continuation prefill
        # already lives in the pool from prior calls.
        self._prefill_write_swa_to_pool(bsz, kv_full, start_pos, sequence_lengths)
        if self.compress_ratio:
            # #50: compressor self-binds kv_state / score_state /
            # kv_cache from the framework pool at entry, runs body,
            # scatters back at exit.  Pool context was set by the
            # outer ``forward`` wrapper in its try/finally.
            if _dbg:
                self.compressor._dbg_prefix = f"L{self.layer_id:02d}_attn_cmp"
            kv_compress = self.compressor(
                x, start_pos, sequence_lengths=sequence_lengths
            )
            if _dbg:
                self.compressor._dbg_prefix = None
            if kv_compress is not None:
                _rec("kv_compress", kv_compress)
        else:
            kv_compress = None
        kv_cat = self._build_kv_cat(kv_full, kv_compress, any_cont, bsz)
        _rec("kv_cat", kv_cat)

        if _tl_kernels.tilelang_available():
            o = _tl_kernels.sparse_attn(
                q, kv_cat, self.attn_sink, topk_idxs, self.softmax_scale
            )
        else:
            o = _sparse_attn(q, kv_cat, self.attn_sink, topk_idxs, self.softmax_scale)
        _rec("sparse_out", o)
        _rec("topk_idxs", topk_idxs)

        # Inverse RoPE on output (cancels K's absolute position)
        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
        _rec("o_post_inv_rope", o)

        # Grouped output projection: split heads into n_groups groups
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        # wo_a storage is native FP8; dequant on-the-fly and view into group-wise form.
        wo_a_bf16 = self.wo_a.dequant_weight(out_dtype=o.dtype)
        wo_a = wo_a_bf16.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        _rec("wo_a_out", o)
        out = self._lin(self.wo_b, o.flatten(2))
        _rec("wo_b_out_pre_ar", out)
        if self.tp_size > 1:
            # wo_b is row-split along K — each rank produces a partial
            # sum; AR combines across the tp group.
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            all_reduce(out, Group.TP)
            _rec("wo_b_out_post_ar", out)
        return out
