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
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.dsv4.compressor import Compressor
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    cp_all_gather_full,
    cp_freqs_cis_local,
)
from rtp_llm.models_py.modules.dsv4.indexer import Indexer
from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb, precompute_freqs_cis
from rtp_llm.models_py.modules.dsv4.weight_loader import _repack_v4_fp8_scale_to_int32
from rtp_llm.models_py.modules.factory.linear import LinearFactory

# P4 (prefill_opt/final_plan.md, minimal C): fused RMSNorm Triton kernel.
# Replaces the 6-launch ``.float().square().mean(-1).rsqrt().mul().to()``
# chain at all 3 attention RMSNorm sites.  Set DSV4_QK_RMSNORM_FAST=0
# to force REF (debug only).
try:
    from rtp_llm.models_py.modules.dsv4._qk_rmsnorm_triton import v4_rmsnorm

    _QK_RMSNORM_FAST_OK = True
except Exception:  # pragma: no cover
    v4_rmsnorm = None
    _QK_RMSNORM_FAST_OK = False


def _use_qk_rmsnorm_fast() -> bool:
    if not _QK_RMSNORM_FAST_OK:
        return False
    return os.environ.get("DSV4_QK_RMSNORM_FAST", "1") != "0"


# P3 (prefill_opt/final_plan.md): wo_a native FP8 fast path.  Replaces
# ``dequant_weight()`` + bf16 ``einsum("bsgd,grd->bsgr")`` with per-group
# DeepGEMM ``fp8_gemm_nt`` calls (no BF16 weight materialization).  wo_b
# is already on the FP8 native path via ``_v4_fp8_linear_from_dict``.
# Set DSV4_WO_FP8_FAST=0 to force the dequant REF (debug only).
def _use_wo_fp8_fast() -> bool:
    return os.environ.get("DSV4_WO_FP8_FAST", "1") != "0"


_V4_FP8_BLOCK_CFG = Fp8BlockWiseQuantConfig()

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
    w = weights[weight_key]
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
    """Wraps an FP32 norm-weight parameter so that ckpt key `.weight` matches."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))


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

    Returns [bsz, S_local, min(window_size, seq_len_full)] int64.
    Entries whose global position is out-of-range (padding local idxs
    with g >= seq_len_full, or ``(g-win+1)+j > g`` for i/w edge rows)
    are set to -1 so the sparse_attn kernel masks them out.
    """
    device = global_positions.device
    S_local = int(global_positions.shape[0])
    W = min(window_size, max(seq_len_full, 1))
    # base: [S_local, 1] the global KV "right edge" (= g for row i).
    base = global_positions.unsqueeze(1)  # [S_local, 1]
    offs = torch.arange(W, device=device)  # [W]
    # kv_pos[i, j] = (g_i - W + 1) + j
    kv_pos = (base - W + 1) + offs  # [S_local, W]
    # Mask-out:
    #   - kv_pos < 0   (Q too early, window begins before seq start)
    #   - kv_pos > g   (impossible given our window-length design, but
    #                   included for defense)
    #   - kv_pos >= seq_len_full (Q at a padding slot, or short seq)
    valid = (kv_pos >= 0) & (kv_pos <= base) & (kv_pos < seq_len_full)
    kv_pos = torch.where(valid, kv_pos, torch.full_like(kv_pos, -1))
    if W < window_size:
        # Pad trailing columns with -1 so concat with compressed topk is
        # a fixed width across calls (sparse_attn_kernel takes symbolic
        # topk but we keep layout consistent with the non-CP path).
        pad = torch.full(
            (S_local, window_size - W), -1, dtype=kv_pos.dtype, device=device
        )
        kv_pos = torch.cat([kv_pos, pad], dim=1)
    return kv_pos.unsqueeze(0).expand(bsz, -1, -1).contiguous()


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

            # wo_b row-split along K (cols), all_reduce after forward
            self.wo_b = (
                _fp8_sliced("wo_b", col_slice=wo_b_col_slice)
                if tp_size > 1
                else _fp8("wo_b")
            )

            # Non-quantized params copy straight from the dict.
            self.q_norm = _NormHolder(q_lora_rank)
            self.q_norm.weight = nn.Parameter(
                weights[f"{prefix}.q_norm.weight"].float(), requires_grad=False
            )
            self.kv_norm = _NormHolder(head_dim)
            self.kv_norm.weight = nn.Parameter(
                weights[f"{prefix}.kv_norm.weight"].float(), requires_grad=False
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
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        # KV cache: [B, window_size + max_seq_len // ratio, head_dim]
        kv_cache_size = window_size + (
            max_seq_len // compress_ratio if compress_ratio else 0
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size, kv_cache_size, head_dim),
            persistent=False,
        )

        # Phase 4D FP8 KV cache (num_blocks=max_B, block_size=kv_cache_size, 584B/slot).
        # Request r owns block r; write slot = r * kv_cache_size + position_in_block.
        # Identical topk_idxs as the BF16 path (both use request-local [0, kv_cache_size)).
        self._fp8_kv_enabled = False
        self.register_buffer(
            "kv_cache_fp8",
            torch.zeros(max_batch_size, kv_cache_size, 584, dtype=torch.uint8),
            persistent=False,
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
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # CP context bound per-forward by V4Transformer.  None = no CP.
        self._cp_ctx: Optional[CPContext] = None

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
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.eps)
        return (weight * x32).to(dtype)

    def _lin(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Linear call that tolerates both the legacy QuantizedLinear
        (3-D input OK via F.linear) and factory LinearBase (expects 2-D)."""
        if self._factory_mode and x.dim() > 2:
            shape = x.shape
            y = layer(x.reshape(-1, shape[-1]))
            return y.view(*shape[:-1], y.shape[-1])
        return layer(x)

    def _wo_a_grouped_fp8(self, o: torch.Tensor) -> torch.Tensor:
        """Per-group native FP8 GEMM, replaces dequant + bf16 einsum.

        ``o`` is ``[B, S, G, K]`` BF16 from sparse-attn output (after
        inv-RoPE).  ``self.wo_a.weight`` is the full ``[G*R, K]`` FP8
        tensor with UE8M0 block-128 scale.  We slice per-group, build
        ``CudaFp8DeepGEMMLinear`` lazily on first call (so the repack
        runs on-device), then loop over groups calling DeepGEMM's
        ``fp8_gemm_nt`` — no [G*R, K] BF16 dequant, no full transpose.

        Returns ``[B, S, G, R]`` BF16, identical math (within block-FP8
        quant noise) to the dequant + einsum REF path.
        """
        B, S, G, K = o.shape
        R = self.o_lora_rank
        if getattr(self, "_wo_a_groups_lazy", None) is None:
            from rtp_llm.models_py.modules.dsv4.weight_loader import (
                _repack_v4_fp8_scale_to_int32,
            )

            groups = []
            for g in range(G):
                w_g = self.wo_a.weight[g * R : (g + 1) * R].contiguous()
                s_g_raw = self.wo_a.scale[
                    g * R // 128 : (g + 1) * R // 128
                ].contiguous()
                if s_g_raw.dtype == torch.float8_e8m0fnu:
                    s_g = _repack_v4_fp8_scale_to_int32(s_g_raw)
                else:
                    s_g = s_g_raw
                local = {f"_g.weight": w_g, f"_g.scale": s_g}
                lin = LinearFactory.create_linear_from_weights(
                    local,
                    f"_g.weight",
                    f"_g.scale",
                    quant_config=_V4_FP8_BLOCK_CFG,
                )
                groups.append(lin)
            self._wo_a_groups_lazy = groups

        out_full = torch.empty(B, S, G, R, dtype=o.dtype, device=o.device)
        for g in range(G):
            x_g = o[:, :, g, :].contiguous().view(B * S, K)
            out_g = self._wo_a_groups_lazy[g](x_g)
            out_full[:, :, g, :].copy_(out_g.view(B, S, R))
        return out_full

    def forward_decode(
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
        from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
            write_compressed_k_decode,
            write_swa_k_decode,
        )
        from rtp_llm.models_py.modules.dsv4.decode.sparse_attn_decode_op import (
            SparseAttnV4DecodeOp,
        )

        bsz, q_len, _ = x.size()
        assert q_len == 1, "Phase 2: q_len==1 only (MTP/spec-decode is later)"
        if self._fp8_kv_enabled:
            return self._forward_decode_fp8(x, attn_metadata)
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

        # bind compressor cache + freqs lazily (mirrors prefill arm).
        # Indexer owns its OWN nested compressor (indexer.compressor) with
        # its OWN kv_cache; the indexer prefill path binds it inside
        # indexer.forward(), but the decode_vectorized path doesn't —
        # bind it here so the captured graph sees the buffer.
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
                if self.indexer.compressor.kv_cache is None:
                    self.indexer.compressor.kv_cache = self.indexer.kv_cache
                    self.indexer.compressor.freqs_cis = self.freqs_cis

        # Q path
        qr = self._rmsnorm_weighted(
            self._lin(self.wq_a, x), self.q_norm.weight
        )  # [B, 1, q_lora]
        q = self._lin(self.wq_b, qr).unflatten(
            -1, (self.n_heads, self.head_dim)
        )  # [B, 1, H, D]
        if _use_qk_rmsnorm_fast() and q.is_cuda and q.numel() > 0:
            q = v4_rmsnorm(q, None, eps=self.eps)
        else:
            q = q * torch.rsqrt(
                q.float().square().mean(-1, keepdim=True) + self.eps
            ).to(q.dtype)
        # Per-request RoPE on q_pe — each req has its own start_pos.
        freqs_cis_per_req = self.freqs_cis[start_pos.long()]  # [B, freqs_dim]
        for r in range(bsz):
            apply_rotary_emb(q[r : r + 1, :, :, -rd:], freqs_cis_per_req[r : r + 1])

        # KV path (single MQA head)
        kv = self._rmsnorm_weighted(
            self._lin(self.wkv, x), self.kv_norm.weight
        )  # [B, 1, head_dim]
        for r in range(bsz):
            apply_rotary_emb(kv[r : r + 1, :, -rd:], freqs_cis_per_req[r : r + 1])

        # Write SWA K — flat slot mapping over [B*q_len].
        # Slice metadata to actual bsz (allocated for max_bs by graph impl;
        # captured graph at BS=k must read only [:k] slots, else PyTorch
        # broadcast-assigns the bsz-sized k_state across all max_bs slot
        # positions, corrupting the KV cache).
        kv_flat = kv.reshape(bsz * q_len, self.head_dim)  # [T, head_dim]
        swa_buffer = self.kv_cache[:, :win]  # [max_B, win, head_dim]
        slot_mapping_swa = attn_metadata.slot_mapping_swa[: bsz * q_len]
        write_swa_k_decode(kv_flat, slot_mapping_swa, swa_buffer)

        # CSA / HCA: build / fill compressed topk; write compressed-K.
        # Stage 3B: when the metadata is from CUDA-graph capture, dispatch
        # to the vectorized (Python-branch-free) variants so the captured
        # forward holds no data-dependent control flow. Eager Phase 2 path
        # keeps the loop variants for byte-equal regression safety.
        use_vec = bool(getattr(attn_metadata, "is_cuda_graph", False))
        topk_idxs: torch.Tensor
        if self.compress_ratio:
            # Slice topk_buffer_compressed to actual bsz so indexer writes
            # only the [:bsz] prefix (graph impl allocates [max_bs, ...]).
            topk_buf_cmp = attn_metadata.topk_buffer_compressed[:bsz]
            if self.indexer is not None:
                # CSA layer (ratio=4): indexer fills its own buffer slot in
                # topk_buf_cmp; we then stitch into topk_total_by_ratio[4][..., win:].
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
                # The builder already fills [win+K_dense) with [0..lens) but
                # WITHOUT the +win offset — apply it here.
                topk_total = attn_metadata.topk_total_by_ratio[ratio][:bsz].clone()
                cmp_part = topk_total[:, :, win:]
                cmp_part = torch.where(cmp_part >= 0, cmp_part + win, cmp_part)
                topk_total[:, :, win:] = cmp_part
                topk_idxs = topk_total
                # HCA also writes the compressor's compressed-K output via
                # the compressor.kv_cache view bound earlier.
                if use_vec:
                    self.compressor.forward_decode_vectorized(x, start_pos)
                else:
                    self.compressor.forward_decode(x, start_pos)
        else:
            # SWA-only layer: just window topk. Already request-local ring slots.
            topk_idxs = attn_metadata.topk_window_idxs[:bsz]

        # Sparse attn over per-request KV view.
        # NOTE: kv_cache layout is [max_B, win + max_seq_len/ratio, head_dim].
        # For SWA-only layers, only the [:, :win, :] slice carries valid data
        # but the buffer is allocated as [max_B, win, head_dim] (no compressed
        # tail). For CSA/HCA, the full buffer is used.
        kv_view = self.kv_cache[:bsz]  # [B, T, head_dim]
        sparse_op = SparseAttnV4DecodeOp(
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            softmax_scale=self.softmax_scale,
        )
        o = sparse_op.forward(q, kv_view, self.attn_sink, topk_idxs)  # [B, 1, H, D]

        # Inverse RoPE per request
        for r in range(bsz):
            apply_rotary_emb(
                o[r : r + 1, :, :, -rd:], freqs_cis_per_req[r : r + 1], inverse=True
            )

        # Grouped output projection (same as prefill)
        o = o.reshape(bsz, q_len, self.n_groups, -1)
        if self._factory_mode and _use_wo_fp8_fast() and o.is_cuda and o.numel() > 0:
            o = self._wo_a_grouped_fp8(o)
        else:
            wo_a_bf16 = self.wo_a.dequant_weight(out_dtype=o.dtype)
            wo_a = wo_a_bf16.view(self.n_groups, self.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        out = self._lin(self.wo_b, o.flatten(2))
        if self.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            all_reduce(out, Group.TP)
        return out

    # ------------------------------------------------------------------
    # Phase 4D — FP8 KV decode path
    # ------------------------------------------------------------------

    def enable_fp8_kv_cache(self, bsz: int) -> None:
        """Convert BF16 kv_cache → FP8 after prefill.

        Call once after the prefill step when switching to the FP8 decode
        path. All slots for requests [0, bsz) are bulk-converted; zeros
        in unwritten slots are harmless (topk masking prevents reads).
        """
        from rtp_llm.models_py.modules.dsv4.decode.fp8_kv_quant_decode_op import (
            quantize_v4_kv_decode,
        )

        block_size = self.kv_cache_fp8.shape[1]  # win + max_seq_len // ratio
        T = bsz * block_size
        # Flat identity mapping: BF16 kv_cache[r, p, :] → FP8 kv_cache_fp8[r, p, :]
        # Both have the same [max_B, block_size] layout, so slot index = r * block_size + p.
        slot_map = torch.arange(T, device=self.kv_cache.device, dtype=torch.long)
        kv_flat = self.kv_cache[:bsz].reshape(T, self.head_dim)
        quantize_v4_kv_decode(kv_flat, slot_map, self.kv_cache_fp8)
        self._fp8_kv_enabled = True

    def _forward_decode_fp8(
        self,
        x: torch.Tensor,  # [B, 1, dim] bf16
        attn_metadata: "DSv4DecodeAttnMetadata",  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """Phase 4D: decode forward with FP8 KV cache + FlashMLA.

        Replaces the BF16 register_buffer + SparseAttnV4DecodeOp path with:
          * quantize_v4_kv_decode  — write new K into kv_cache_fp8
          * SparseAttnV4DecodeFp8Op — FlashMLA is_fp8_kvcache=True sparse attn

        For CSA layers the Attention's compressed-K (kv_cache_fp8[:, win:, :])
        was populated once by enable_fp8_kv_cache() and is not re-written here
        (the Indexer updates its own compressor's kv_cache for scoring only).
        For HCA layers the compressor returns a fresh BF16 compressed-K each
        step which we quantize and write to FP8.
        """
        from rtp_llm.models_py.modules.dsv4.decode.fp8_kv_quant_decode_op import (
            quantize_v4_kv_decode,
        )
        from rtp_llm.models_py.modules.dsv4.decode.fp8_sparse_attn_decode_op import (
            SparseAttnV4DecodeFp8Op,
        )

        bsz, q_len, _ = x.size()
        win = self.window_size
        ratio = self.compress_ratio
        block_size = self.kv_cache_fp8.shape[1]  # win + max_seq_len // ratio (or win)
        rd = self.rope_head_dim
        device = x.device
        start_pos = attn_metadata.start_pos  # [B] int32

        # Bind compressor/indexer caches lazily (same as BF16 forward_decode).
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # Q path (identical to BF16 forward_decode).
        qr = self._rmsnorm_weighted(self._lin(self.wq_a, x), self.q_norm.weight)
        q = self._lin(self.wq_b, qr).unflatten(-1, (self.n_heads, self.head_dim))
        if _use_qk_rmsnorm_fast() and q.is_cuda and q.numel() > 0:
            q = v4_rmsnorm(q, None, eps=self.eps)
        else:
            q = q * torch.rsqrt(
                q.float().square().mean(-1, keepdim=True) + self.eps
            ).to(q.dtype)
        freqs_cis_per_req = self.freqs_cis[start_pos.long()]
        for r in range(bsz):
            apply_rotary_emb(q[r : r + 1, :, :, -rd:], freqs_cis_per_req[r : r + 1])

        # KV path.
        kv = self._rmsnorm_weighted(self._lin(self.wkv, x), self.kv_norm.weight)
        for r in range(bsz):
            apply_rotary_emb(kv[r : r + 1, :, -rd:], freqs_cis_per_req[r : r + 1])

        # Write SWA K to FP8 cache.
        # BF16 slot_mapping_swa uses stride=win: slot = r * win + ring_pos.
        # FP8 kv_cache_fp8 has stride=block_size, so: fp8_slot = r * block_size + ring_pos.
        # Remap: fp8_slot = (bf16_slot // win) * block_size + (bf16_slot % win).
        kv_flat = kv.reshape(bsz * q_len, self.head_dim)
        swa_slot_bf16 = attn_metadata.slot_mapping_swa.to(torch.long)
        swa_slot_fp8 = (swa_slot_bf16 // win) * block_size + (swa_slot_bf16 % win)
        quantize_v4_kv_decode(kv_flat, swa_slot_fp8, self.kv_cache_fp8)

        use_vec = bool(getattr(attn_metadata, "is_cuda_graph", False))
        topk_idxs: torch.Tensor
        if self.compress_ratio:
            if self.indexer is not None:
                # CSA: Indexer scores against its own compressor's kv_cache;
                # the Attention's compressed-K FP8 slots were written at enable_fp8_kv_cache()
                # and don't change during decode.
                if use_vec:
                    self.indexer.forward_decode_vectorized(
                        x,
                        qr,
                        start_pos,
                        attn_metadata.topk_buffer_compressed,
                    )
                else:
                    self.indexer.forward_decode(
                        x,
                        qr,
                        start_pos,
                        attn_metadata.topk_buffer_compressed,
                    )
                topk_total = attn_metadata.topk_total_by_ratio[4]
                idx_with_off = torch.where(
                    attn_metadata.topk_buffer_compressed >= 0,
                    attn_metadata.topk_buffer_compressed + win,
                    attn_metadata.topk_buffer_compressed,
                )
                topk_total[:, :, win:] = idx_with_off
                topk_idxs = topk_total
            else:
                # HCA: compressor produces new compressed-K each step → write to FP8.
                if use_vec:
                    kv_compressed = self.compressor.forward_decode_vectorized(
                        x, start_pos
                    )
                else:
                    kv_compressed = self.compressor.forward_decode(x, start_pos)

                if kv_compressed is not None:
                    # Remap: BF16 compressed slot = r * stride_bf16 + c
                    # FP8 compressed slot = r * block_size + win + c
                    stride_bf16 = attn_metadata.compressed_buffer_t_dim_per_ratio[ratio]
                    cmp_slot_bf16 = attn_metadata.slot_mapping_compressed[ratio].to(
                        torch.long
                    )
                    valid_mask = cmp_slot_bf16 >= 0
                    cmp_r = cmp_slot_bf16 // stride_bf16
                    cmp_c = cmp_slot_bf16 % stride_bf16
                    cmp_slot_fp8 = torch.where(
                        valid_mask,
                        cmp_r * block_size + win + cmp_c,
                        torch.full_like(cmp_slot_bf16, -1),
                    )
                    kv_c_flat = kv_compressed.reshape(bsz * q_len, self.head_dim)
                    quantize_v4_kv_decode(kv_c_flat, cmp_slot_fp8, self.kv_cache_fp8)

                topk_total = attn_metadata.topk_total_by_ratio[ratio].clone()
                cmp_part = topk_total[:, :, win:]
                cmp_part = torch.where(cmp_part >= 0, cmp_part + win, cmp_part)
                topk_total[:, :, win:] = cmp_part
                topk_idxs = topk_total
        else:
            topk_idxs = attn_metadata.topk_window_idxs

        # FP8 sparse attention via FlashMLA (or reference fallback on non-SM100).
        # block_table[r, 0] = r so FlashMLA reads kv_cache_fp8[r, topk_slot, :].
        cache_seqlens = (start_pos + 1).to(torch.int32)
        block_table = torch.arange(bsz, device=device, dtype=torch.int32).unsqueeze(1)
        sparse_fp8 = SparseAttnV4DecodeFp8Op(
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            softmax_scale=self.softmax_scale,
        )
        o = sparse_fp8.forward(
            q,
            self.kv_cache_fp8[:bsz],
            self.attn_sink,
            topk_idxs,
            cache_seqlens,
            block_table,
        )

        # Inverse RoPE per request.
        for r in range(bsz):
            apply_rotary_emb(
                o[r : r + 1, :, :, -rd:], freqs_cis_per_req[r : r + 1], inverse=True
            )

        # Grouped output projection (identical to BF16 path).
        o = o.reshape(bsz, q_len, self.n_groups, -1)
        if self._factory_mode and _use_wo_fp8_fast() and o.is_cuda and o.numel() > 0:
            o = self._wo_a_grouped_fp8(o)
        else:
            wo_a_bf16 = self.wo_a.dequant_weight(out_dtype=o.dtype)
            wo_a = wo_a_bf16.view(self.n_groups, self.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        out = self._lin(self.wo_b, o.flatten(2))
        if self.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            all_reduce(out, Group.TP)
        return out

    def forward(
        self, x: torch.Tensor, start_pos, sequence_lengths=None
    ) -> torch.Tensor:
        """Forward pass. start_pos can be int (B=1) or tensor [B] for batched decode."""
        bsz, seqlen, _ = x.size()
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        device = x.device

        cp_ctx = self._cp_ctx
        is_batched_decode = (
            isinstance(start_pos, torch.Tensor) and start_pos.numel() > 1
        )
        is_prefill_attn = (not is_batched_decode) and seqlen > 1
        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1 and is_prefill_attn

        # Per-token RoPE angles.  Non-CP uses the contiguous window
        # freqs_cis[start_pos:start_pos+seqlen]; CP selects at each
        # rank-local token's GLOBAL position.
        is_batched_decode = (
            isinstance(start_pos, torch.Tensor) and start_pos.numel() > 1
        )
        if cp_on:
            freqs_cis = cp_freqs_cis_local(self.freqs_cis, cp_ctx)
        elif is_batched_decode:
            # Batched decode: each batch element at different position, seqlen=1
            # Gather freqs for each batch element's position
            positions = start_pos.long()
            freqs_cis = self.freqs_cis[positions]  # [B, rope_dim//2]
            freqs_cis = freqs_cis.unsqueeze(1)  # [B, 1, rope_dim//2]
        else:
            sp = int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
            freqs_cis = self.freqs_cis[sp : sp + seqlen]

        # bind compressor cache + freqs on first call
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # Q path
        qr = self._rmsnorm_weighted(
            self._lin(self.wq_a, x), self.q_norm.weight
        )  # [B, S, q_lora_rank]
        q = self._lin(self.wq_b, qr).unflatten(-1, (self.n_heads, self.head_dim))
        # QK RMSNorm (no learnable scale here, per official code)
        if _use_qk_rmsnorm_fast() and q.is_cuda and q.numel() > 0:
            q = v4_rmsnorm(q, None, eps=self.eps)
        else:
            q = q * torch.rsqrt(
                q.float().square().mean(-1, keepdim=True) + self.eps
            ).to(q.dtype)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # KV path (single KV head) — rank-local under CP.
        kv = self._rmsnorm_weighted(
            self._lin(self.wkv, x), self.kv_norm.weight
        )  # [B, S_local, head_dim]
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

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

        # Build topk_idxs — rows = rank-local Q; columns reference the
        # concatenated [sliding | compressed] KV tensor (under CP the
        # sliding portion has ``seqlen_full`` entries, not
        # ``chunk_length``).
        if cp_on:
            topk_idxs = _get_window_topk_idxs_cp(
                win,
                bsz,
                seqlen_full,
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
        else:
            sp = int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
            topk_idxs = _get_window_topk_idxs(win, bsz, seqlen, sp, device)
        if self.compress_ratio:
            # The concatenated prefill KV is [sliding (seqlen_full), compressed tail].
            # Compressed block start-index is seqlen_full in prefill,
            # win in decode (ring-buffer layout).
            if is_prefill_attn:
                offset = seqlen_full
            else:
                offset = win
            if self.indexer is not None:
                if is_batched_decode:
                    # Indexer now supports tensor start_pos directly
                    compress_idxs = self.indexer(x, qr, start_pos, offset)
                else:
                    compress_idxs = self.indexer(x, qr, start_pos, offset)
            elif cp_on:
                compress_idxs = _get_compress_topk_idxs_cp(
                    ratio,
                    bsz,
                    seqlen_full,
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
                    c_idxs = torch.where(valid, entry_range.unsqueeze(0) + offset, -1)
                    compress_idxs = c_idxs.unsqueeze(1)  # [B, 1, max_entries]
                else:
                    compress_idxs = torch.full(
                        (bsz, 1, 0), -1, device=device, dtype=torch.long
                    )
            else:
                sp_int = (
                    int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
                )
                compress_idxs = _get_compress_topk_idxs(
                    ratio, bsz, seqlen, sp_int, offset, device
                )
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
        topk_idxs = topk_idxs.long()

        # Write KV cache + sparse attn
        sp_int = (
            int(start_pos)
            if isinstance(start_pos, torch.Tensor) and start_pos.numel() == 1
            else (start_pos if isinstance(start_pos, int) else 0)
        )

        if is_prefill_attn:
            if sp_int == 0:
                # Fresh prefill: write all tokens into ring buffer
                if seqlen_full <= win:
                    self.kv_cache[:bsz, :seqlen_full] = kv_full
                else:
                    cutoff = seqlen_full % win
                    self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = (
                        kv_full[:, -win:].split([win - cutoff, cutoff], dim=1)
                    )
            else:
                # Continuation prefill: prefix KV already in kv_cache (gathered from BlockPool).
                # Write new tokens into ring buffer starting at start_pos.
                total_len = sp_int + seqlen
                if total_len <= win:
                    self.kv_cache[:bsz, sp_int:total_len] = kv_full
                else:
                    # Ring buffer wrap: place last `win` tokens of [prefix..prefix+seqlen]
                    # But we only have the new tokens (kv_full). Prefix is already in cache.
                    # Write new tokens at their ring positions.
                    for t in range(seqlen):
                        pos = (sp_int + t) % win
                        self.kv_cache[:bsz, pos] = kv_full[:, t]
            if self.compress_ratio:
                kv_compress = self.compressor(x, sp_int)
                if kv_compress is not None:
                    if sp_int == 0:
                        kv_cat = torch.cat([kv_full, kv_compress], dim=1)
                    else:
                        # Continuation: use ring buffer for attention (has prefix + new KV)
                        kv_cat = self.kv_cache[:bsz]
                else:
                    if sp_int == 0:
                        kv_cat = kv_full
                    else:
                        kv_cat = self.kv_cache[:bsz]
            else:
                if sp_int == 0:
                    kv_cat = kv_full
                else:
                    kv_cat = self.kv_cache[:bsz]
            if _tl_kernels.tilelang_available():
                o = _tl_kernels.sparse_attn(
                    q, kv_cat, self.attn_sink, topk_idxs, self.softmax_scale
                )
            else:
                o = _sparse_attn(
                    q, kv_cat, self.attn_sink, topk_idxs, self.softmax_scale
                )
        else:
            # Decode: write each batch element to its own ring position
            if is_batched_decode:
                batch_idx = torch.arange(bsz, device=device)
                pos = start_pos % win  # [B]
                self.kv_cache[batch_idx, pos] = kv.squeeze(1)  # vectorized write
                if self.compress_ratio:
                    self.compressor(x, start_pos)  # compressor handles tensor start_pos
            else:
                sp = (
                    int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
                )
                self.kv_cache[:bsz, sp % win] = kv.squeeze(1)
                if self.compress_ratio:
                    self.compressor(x, sp)
            if _tl_kernels.tilelang_available():
                o = _tl_kernels.sparse_attn(
                    q,
                    self.kv_cache[:bsz],
                    self.attn_sink,
                    topk_idxs,
                    self.softmax_scale,
                )
            else:
                o = _sparse_attn(
                    q,
                    self.kv_cache[:bsz],
                    self.attn_sink,
                    topk_idxs,
                    self.softmax_scale,
                )

        # Inverse RoPE on output (cancels K's absolute position)
        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        # Grouped output projection: split heads into n_groups groups
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        # wo_a storage is native FP8; dequant on-the-fly and view into group-wise form.
        wo_a_bf16 = self.wo_a.dequant_weight(out_dtype=o.dtype)
        wo_a = wo_a_bf16.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        out = self._lin(self.wo_b, o.flatten(2))
        if self.tp_size > 1:
            # wo_b is row-split along K — each rank produces a partial
            # sum; AR combines across the tp group.
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            all_reduce(out, Group.TP)
        return out
