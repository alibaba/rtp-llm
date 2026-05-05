"""DeepSeek-V4 lightning Indexer for CSA.

Faithful BF16 port of `inference/model.py:Indexer`. Skips Hadamard rotation
+ FP4 quant (BF16-only path for M2/M3 correctness validation).

Has its own dedicated Compressor (rotate=True in official code; we keep
the parameter for ckpt-loader symmetry but don't apply Hadamard).
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4._indexer_cp_gather_triton import (
    gather_indexer_k_for_prefill,
)
from rtp_llm.models_py.modules.dsv4._indexer_fp8_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    dequantize_indexer_k,
)
from rtp_llm.models_py.modules.dsv4._indexer_q_fp8_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4._indexer_score_fp8 import (
    fp8_mqa_indexer_score,
    fp8_paged_indexer_score,
    has_fp8_mqa_logits,
    has_fp8_paged_mqa_logits,
)
from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score
from rtp_llm.models_py.modules.dsv4.compressor import Compressor
from rtp_llm.models_py.modules.dsv4.cp import CPContext, cp_freqs_cis_local
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    PoolBackedModule,
    is_fp8_indexer_pool,
)
from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def _as_bf16_contig(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


# ---------------------------------------------------------------------------
# Persistent radix-select TopK (vendored from vLLM, see
# rtp_llm/models_py/bindings/cuda/kernels/dsv4_persistent_topk.{h,cu,cuh}).
# Replaces ``score.topk + masked_fill`` on the indexer decode path:
#   K=512 / T_max=2048: torch.topk ~30us  →  this kernel ~5-10us per call.
# Off by default until the .so is rebuilt with the new binding.  Set
# ``DSV4_PERSISTENT_TOPK=1`` to enable.
# ---------------------------------------------------------------------------
_PERSISTENT_TOPK_OK = hasattr(rtp_llm_ops, "dsv4_persistent_topk")
_PERSISTENT_TOPK_WORKSPACE_SIZE = 1024 * 1024  # 1 MB, matches vLLM
_persistent_topk_workspace_cache: Dict[torch.device, torch.Tensor] = {}


def _persistent_topk_enabled() -> bool:
    if not _PERSISTENT_TOPK_OK:
        return False
    # Default ON when the binding is built in. Set DSV4_PERSISTENT_TOPK=0
    # to fall back to ``torch.topk`` (e.g. for golden-tensor diff when the
    # set-equal-but-order-different output trips a strict comparator).
    return os.environ.get("DSV4_PERSISTENT_TOPK", "1") != "0"


def _fp8_deepgemm_score_enabled() -> bool:
    """DeepGEMM FP8 paged MQA logits path. Default ON when the deep_gemm
    Python pkg has the API (it's bundled with our prod CUDA image). Set
    ``DSV4_FP8_DEEPGEMM_SCORE=0`` to force the bf16-dequant Triton path
    (e.g. for golden-tensor comparison)."""
    if not has_fp8_paged_mqa_logits():
        return False
    return os.environ.get("DSV4_FP8_DEEPGEMM_SCORE", "1") != "0"


def _fp8_deepgemm_prefill_score_enabled() -> bool:
    """DeepGEMM FP8 non-paged MQA logits path for prefill. Same env gate
    as the decode/paged path (``DSV4_FP8_DEEPGEMM_SCORE``) so a single
    knob disables both."""
    if not has_fp8_mqa_logits():
        return False
    return os.environ.get("DSV4_FP8_DEEPGEMM_SCORE", "1") != "0"


def _get_topk_workspace(device: torch.device) -> torch.Tensor:
    ws = _persistent_topk_workspace_cache.get(device)
    if ws is None:
        ws = torch.empty(
            _PERSISTENT_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=device
        )
        _persistent_topk_workspace_cache[device] = ws
    return ws


class Indexer(PoolBackedModule):
    def __init__(
        self,
        dim: int,
        q_lora_rank: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        compress_ratio: int,
        max_batch_size: int,
        max_seq_len: int,
        norm_eps: float = 1e-6,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio
        self._factory_mode = weights is not None

        if self._factory_mode:
            from rtp_llm.models_py.modules.dsv4.attention import (
                _v4_fp8_linear_from_dict,
            )

            self.wq_b = _v4_fp8_linear_from_dict(
                weights,
                f"{prefix}.wq_b.weight",
                f"{prefix}.wq_b.scale",
            )
            self.weights_proj = nn.Linear(dim, index_n_heads, bias=False)
            self.weights_proj.weight = nn.Parameter(
                weights[f"{prefix}.weights_proj.weight"],
                requires_grad=False,
            )
        else:
            self.wq_b = QuantizedLinear(
                q_lora_rank, index_n_heads * index_head_dim, storage="fp8"
            )
            self.weights_proj = nn.Linear(dim, index_n_heads, bias=False)

        self.compressor = Compressor(
            dim=dim,
            head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            max_batch_size=max_batch_size,
            norm_eps=norm_eps,
            rotate=True,
            weights=weights,
            prefix=f"{prefix}.compressor" if self._factory_mode else "",
        )
        self.max_batch_size = max_batch_size
        self._kv_cache_t = max_seq_len // compress_ratio
        self._kv_cache_d = index_head_dim
        self.freqs_cis: Optional[torch.Tensor] = None
        self._cp_ctx: Optional[CPContext] = None
        self._dbg_prefix: Optional[str] = None

    # ------------------------------------------------------------------
    # Pool propagation to nested compressor
    # ------------------------------------------------------------------

    def _propagate_pool_to_nested(self) -> None:
        if self.compressor is None:
            return
        self.compressor.set_pool_context(
            self._kv_pool_view,
            self._kv_block_table,
            self._kv_eb,
            self._state_pool_view,
            self._state_block_table,
            self._state_eb,
        )

    def _clear_nested_pool(self) -> None:
        if self.compressor is None:
            return
        self.compressor.clear_pool_context()

    def set_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        self._cp_ctx = cp_ctx

    # ------------------------------------------------------------------
    # Shared Q-projection + RoPE helper
    # ------------------------------------------------------------------

    def _compute_indexer_q(
        self,
        qr: torch.Tensor,
        freqs_cis: torch.Tensor,
        batched_rope: bool = False,
    ) -> torch.Tensor:
        """qr -> wq_b -> unflatten -> RoPE -> q [B, S, H, D]."""
        if self._factory_mode and qr.dim() > 2:
            shape = qr.shape
            q = self.wq_b(qr.reshape(-1, shape[-1])).view(
                *shape[:-1],
                self.n_heads * self.head_dim,
            )
        else:
            q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        if batched_rope:
            apply_rotary_emb_batched(q[..., -self.rope_head_dim :], freqs_cis)
        else:
            apply_rotary_emb(q[..., -self.rope_head_dim :], freqs_cis)
        return q

    # ------------------------------------------------------------------
    # FP8-aware kv_cache bind: under FP8 indexer cache the pool view is
    # uint8/132B per slot. ``PoolBackedModule._bind_kv_cache_from_pool``
    # would do a value-cast (uint8 → bf16) which reinterprets bytes as
    # bf16 values — broken. Detect that layout and dequant via the
    # Triton kernel instead. Falls back to the base class for the
    # bf16-cache (non-FP8) path.
    # ------------------------------------------------------------------
    def _bind_kv_cache_for_indexer(
        self, bsz: int, is_fresh_prefill: bool, device: torch.device
    ) -> None:
        if not is_fp8_indexer_pool(self._kv_pool_view):
            self._bind_kv_cache_from_pool(
                bsz,
                is_fresh_prefill=is_fresh_prefill,
                device=device,
                dtype=torch.bfloat16,
            )
            return

        T = (
            self._kv_cache_t
            if self._kv_cache_t > 0
            else (
                self._kv_block_table.shape[1] * self._kv_eb
                if self._kv_block_table is not None
                else 0
            )
        )
        if T <= 0 or is_fresh_prefill or self._kv_block_table is None:
            self.kv_cache = torch.zeros(
                bsz, T, INDEXER_HEAD_DIM, dtype=torch.bfloat16, device=device
            )
            return
        valid, safe_slot = self._compute_pool_slots(
            bsz, T, self._kv_block_table, self._kv_eb, device
        )
        # -1 in slot_mapping → kernel writes zeros for that slot.
        slot_mapping = torch.where(
            valid, safe_slot, torch.full_like(safe_slot, -1)
        ).reshape(-1)
        # Pool view shape from framework: [num_blocks * eb, 132] flat. The
        # dequant kernel expects [num_blocks, block_size, 132]; we feed it
        # a single virtual block view — slot indices are already absolute
        # into the flat pool, so block_size=1 makes ``slot // bs * bs``
        # identity.
        pool_blocks = self._kv_pool_view.view(-1, 1, INDEXER_ENTRY_BYTES)
        flat = dequantize_indexer_k(
            pool_blocks,
            slot_mapping,
            out_dtype=torch.bfloat16,
        )
        self.kv_cache = flat.view(bsz, T, INDEXER_HEAD_DIM).contiguous()

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def forward_decode_vectorized(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: torch.Tensor,
        out_topk_buffer: torch.Tensor,
    ) -> torch.Tensor:
        assert x.shape[1] == 1, "decode-only: q_len must be 1"

        bsz = x.size(0)
        ratio = self.compress_ratio
        K = self.index_topk

        self._propagate_pool_to_nested()
        try:
            self.compressor.forward_decode_vectorized(x, start_pos)

            freqs_per_b = self.freqs_cis[start_pos.long()]
            q = self._compute_indexer_q(qr, freqs_per_b, batched_rope=True)

            weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)

            compressed_len = ((start_pos + 1) // ratio).to(torch.int64).view(bsz, 1, 1)

            # FP8 fast path: paged FP8 cache + DeepGEMM
            # ``fp8_paged_mqa_logits`` consumes the cache directly. Skips
            # the bf16 dequant materialization entirely.
            if (
                is_fp8_indexer_pool(self._kv_pool_view)
                and self._kv_block_table is not None
                and _fp8_deepgemm_score_enabled()
            ):
                T_cache = self._kv_block_table.shape[1] * self._kv_eb
                # 32-align T_live for kernel uniformity (DeepGEMM is fine
                # with arbitrary T but the topk grid likes power-of-2-ish).
                sp_max = int(start_pos.max().item())
                T_live = (((sp_max + 1) // ratio) + 31) & ~31
                T_max = max(32, min(T_cache, T_live))

                q_fp8, w_fold = indexer_q_fp8_quant_fold(
                    _as_bf16_contig(q), _as_bf16_contig(weights)
                )
                ctx_lens_2d = compressed_len.view(bsz, 1).to(torch.int32)
                bt_i32 = self._kv_block_table[:bsz].to(torch.int32).contiguous()
                logits = fp8_paged_indexer_score(
                    q_fp8,
                    w_fold.view(bsz * 1, self.n_heads),
                    self._kv_pool_view,
                    bt_i32,
                    ctx_lens_2d,
                    block_size=self._kv_eb,
                    max_ctx_len=T_max,
                )  # [B, T_max] fp32
                score = logits.view(bsz, 1, T_max)
            else:
                # bf16 fallback: dequant materialization + Triton score.
                self._bind_kv_cache_for_indexer(
                    bsz, is_fresh_prefill=False, device=x.device
                )
                T_cache = self.kv_cache.shape[1]
                # P0: crop T to live compressed length (32-aligned).
                sp_max = int(start_pos.max().item())
                T_live = (((sp_max + 1) // ratio) + 31) & ~31
                T_max = max(32, min(T_cache, T_live))
                kv_view = self.kv_cache[:bsz, :T_max]
                q_pos = (compressed_len.view(bsz) * ratio - 1).to(torch.int32)
                score = v4_indexer_score(
                    _as_bf16_contig(q),
                    _as_bf16_contig(kv_view),
                    _as_bf16_contig(weights),
                    q_pos=q_pos.unsqueeze(1),
                    compress_ratio=ratio,
                )

            K_eff = min(K, T_max)
            if K_eff > 0 and K in (512, 1024, 2048) and _persistent_topk_enabled():
                # Fused radix-select + length-mask + -1 padding in one kernel.
                # Kernel writes -1 past lengths[r] directly into the output.
                rtp_llm_ops.dsv4_persistent_topk(
                    score.view(bsz, T_max),
                    compressed_len.view(bsz).to(torch.int32),
                    out_topk_buffer.view(bsz, K),
                    _get_topk_workspace(score.device),
                    K,
                    T_max,
                )
            else:
                out_topk_buffer.fill_(-1)
                if K_eff > 0:
                    topk_idxs = score.topk(K_eff, dim=-1)[1].to(torch.int32)
                    out_topk_buffer[:, :, :K_eff].copy_(topk_idxs)
                    k_arange = torch.arange(K, device=out_topk_buffer.device).view(
                        1, 1, K
                    )
                    out_topk_buffer.masked_fill_(
                        k_arange >= compressed_len,
                        -1,
                    )

            return out_topk_buffer
        finally:
            self._clear_nested_pool()
            self.kv_cache = None

    # ------------------------------------------------------------------
    # Forward (prefill only — decode uses forward_decode_vectorized)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos,
        offset: int,
    ) -> torch.Tensor:
        """Prefill-only entry point.  Decode goes through
        :meth:`forward_decode_vectorized` which handles batched decode
        natively.  Attention enforces ``bsz==1`` for prefill."""
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        cp_ctx = self._cp_ctx

        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1 and start_pos == 0
        _dbg = self._dbg_prefix if _rt.ENABLED else None

        if cp_on:
            freqs_cis = cp_freqs_cis_local(self.freqs_cis, cp_ctx)
            end_pos = cp_ctx.seq_len_full
            sp = 0
        else:
            sp = int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
            freqs_cis = self.freqs_cis[sp : sp + seqlen]
            end_pos = sp + seqlen

        is_fresh_prefill = sp == 0

        if self.compressor.freqs_cis is None:
            self.compressor.freqs_cis = self.freqs_cis

        self._propagate_pool_to_nested()
        try:
            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_freqs_cis", freqs_cis)
            q = self._compute_indexer_q(qr, freqs_cis)
            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_q_post_rope", q)

            if _dbg is not None:
                self.compressor._dbg_prefix = f"{_dbg}_cmp"
            self.compressor(x, start_pos)
            if _dbg is not None:
                self.compressor._dbg_prefix = None

            weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_weights", weights)

            S = q.size(1)
            T = end_pos // ratio

            if is_fresh_prefill:
                if cp_on:
                    qpos_row = cp_ctx.global_positions.to(torch.int32)
                else:
                    qpos_row = torch.arange(S, device=q.device, dtype=torch.int32)
                q_pos_kernel = qpos_row.view(1, S).expand(bsz, S).contiguous()
            else:
                q_pos_kernel = None

            # FP8 fast path: paged FP8 cache + DeepGEMM ``fp8_mqa_logits``.
            # Mirrors the decode ``is_fp8_pool`` branch — skips the bf16
            # dequant materialization entirely, gathers (k_quant, k_scale)
            # straight from the per-block grouped pool.
            is_fp8_pool_prefill = (
                is_fp8_indexer_pool(self._kv_pool_view)
                and self._kv_block_table is not None
                and not cp_on  # CP path keeps bf16 (positions split across ranks)
                and _fp8_deepgemm_prefill_score_enabled()
            )

            if is_fp8_pool_prefill and T > 0:
                # Build slot_mapping for the live K range [0, T).
                valid, safe_slot = self._compute_pool_slots(
                    bsz, T, self._kv_block_table, self._kv_eb, x.device
                )
                slot_mapping = torch.where(
                    valid, safe_slot, torch.full_like(safe_slot, -1)
                ).reshape(
                    -1
                )  # [bsz * T]

                pool_blocks = self._kv_pool_view.view(-1, 1, INDEXER_ENTRY_BYTES)
                k_quant_flat, k_scale_flat = gather_indexer_k_for_prefill(
                    pool_blocks,
                    slot_mapping,
                    head_dim=INDEXER_HEAD_DIM,
                )  # [bsz*T, D], [bsz*T]

                # Q FP8 quant + scale-fold into weights.
                q_fp8, w_fold = indexer_q_fp8_quant_fold(
                    _as_bf16_contig(q), _as_bf16_contig(weights)
                )  # [B, S, H, D] / [B, S, H]

                # cu_seqlen_ks/ke per Q row — flatten over (b, s).
                # No-mask (continuation): ke=T_live, ks=0.
                # Causal: ke=(q_pos+1)//ratio, ks=0.
                M = bsz * S
                if q_pos_kernel is not None:
                    ke = ((q_pos_kernel.to(torch.int32) + 1) // ratio).clamp_max(T)
                    cu_ke = ke.reshape(M).contiguous()
                else:
                    cu_ke = torch.full((M,), T, dtype=torch.int32, device=x.device)
                cu_ks = torch.zeros(M, dtype=torch.int32, device=x.device)

                # bsz==1 enforced by Attention prefill, so the gathered K
                # workspace is exactly [T, D] for the single sequence.
                # (For bsz>1, the cu_seqlen_ks/ke per-row would need to
                #  reference each request's own K window in the gathered
                #  workspace — out of scope here.)
                assert bsz == 1, (
                    "FP8 prefill indexer score currently only supports bsz==1 "
                    f"(got {bsz}); fall back to bf16 path for batched prefill."
                )
                k_quant = k_quant_flat  # [T, D]
                k_scale = k_scale_flat  # [T]

                logits = fp8_mqa_indexer_score(
                    q_fp8.view(M, self.n_heads, INDEXER_HEAD_DIM),
                    w_fold.view(M, self.n_heads),
                    k_quant,
                    k_scale,
                    cu_ks,
                    cu_ke,
                    clean_logits=False,
                )  # [M, T]
                index_score = logits.view(bsz, S, T)
            else:
                # bf16 fallback: dequant materialization + Triton score.
                self._bind_kv_cache_for_indexer(
                    bsz, is_fresh_prefill=False, device=x.device
                )
                if _dbg is not None:
                    _rt.record_if_level(
                        2,
                        f"{_dbg}_compressor_kv_cache",
                        self.kv_cache[:bsz, : end_pos // ratio],
                    )
                kv = self.kv_cache[:bsz, : end_pos // ratio]
                index_score = v4_indexer_score(
                    _as_bf16_contig(q),
                    _as_bf16_contig(kv),
                    _as_bf16_contig(weights),
                    q_pos=q_pos_kernel,
                    compress_ratio=ratio,
                )

            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_score_post_mask", index_score)

            topk_idxs = index_score.topk(
                min(self.index_topk, end_pos // ratio), dim=-1
            )[1]
            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_topk_pre_offset", topk_idxs)
            if is_fresh_prefill:
                if cp_on:
                    q_pos_1b = (cp_ctx.global_positions + 1).unsqueeze(1)
                else:
                    q_pos_1b = torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1)
                mask = topk_idxs >= (q_pos_1b // ratio)
                topk_idxs = torch.where(mask, -1, topk_idxs + offset)
            else:
                topk_idxs = topk_idxs + offset
            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_topk_final", topk_idxs)
            return topk_idxs
        finally:
            self._clear_nested_pool()
            self.kv_cache = None
