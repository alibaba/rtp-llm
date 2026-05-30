"""DeepSeek-V4 lightning Indexer — FP8 KV pool path.

Companion to ``indexer.py`` (BF16 path). Always uses DeepGEMM for the
FP8 indexer score (paged for decode, non-paged for prefill); the bf16
``v4_indexer_score`` fallback path lives in the BF16 class.

Nested compressor is ``CompressorFP8(head_dim=128)`` which writes the
132B grouped UE8M0 layout that DeepGEMM consumes directly.

What this class does NOT do — by design:

  * NO ``_bind_kv_cache_for_indexer`` materialization. DeepGEMM reads
    the FP8 pool directly via ``fp8_paged_mqa_logits`` /
    ``fp8_mqa_logits``. No bf16 dequant intermediate.
  * NO env gate (``_fp8_deepgemm_score_enabled`` / equivalent). FP8
    class always uses DeepGEMM. If DeepGEMM is unavailable at runtime,
    the layer construction itself should fail loudly.
  * NO bf16 / FP8 runtime branching. Pick the right indexer class at
    attention construction time.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.chunk_env import dsv4_chunk_tokens_from_env
from rtp_llm.models_py.modules.dsv4.cp import CPContext, build_cp_full_prefill_positions
from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    fp8_mqa_indexer_score,
    fp8_paged_indexer_score,
    has_fp8_mqa_logits,
    has_fp8_paged_mqa_logits,
)
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import PoolBackedModule
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorFP8,
    CompressorMeta,
    _CompressorPending,
)


def _use_varlen_prefill() -> bool:
    return os.environ.get("DSV4_VARLEN_PREFILL", "1") != "0"


from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear
from rtp_llm.ops.compute_ops import rtp_llm_ops


def _as_bf16_contig(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def _flat_1d(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).contiguous()


# Persistent radix-select TopK — vendored CUDA kernel binding. Same gate
# as the BF16 class so a single env knob disables both paths.
_PERSISTENT_TOPK_OK = hasattr(rtp_llm_ops, "dsv4_persistent_topk")
_FAST_PREFILL_TOPK_OK = hasattr(rtp_llm_ops, "fast_topk_v2_variable")
_PERSISTENT_TOPK_WORKSPACE_SIZE = 1024 * 1024  # 1 MB
_FAST_PREFILL_TOPK_MAX_INPUT_TOKENS = 12 * 1024
_persistent_topk_workspace_cache: Dict[torch.device, torch.Tensor] = {}


def _persistent_topk_enabled() -> bool:
    if not _PERSISTENT_TOPK_OK:
        return False
    return os.environ.get("DSV4_PERSISTENT_TOPK", "1") != "0"


def _fp8_prefill_fast_topk_enabled() -> bool:
    if not _FAST_PREFILL_TOPK_OK:
        return False
    return os.environ.get("DSV4_PREFILL_FAST_TOPK", "1") != "0"


def _fp8_prefill_topk_force_radix_sort() -> bool:
    return os.environ.get("DSV4_PREFILL_TOPK_FORCE_RADIX", "1") != "0"


def _fp8_prefill_topk_use_torch() -> bool:
    return (
        os.environ.get("DSV4_INDEXER_TOPK_BACKEND", "auto").strip().lower() == "torch"
    )


def _fp8_prefill_topk_canonicalize() -> bool:
    return os.environ.get("DSV4_INDEXER_TOPK_CANONICALIZE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _run_prefill_topk_torch(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out: torch.Tensor,
    topk: int,
) -> None:
    M, T = logits.shape
    out.fill_(-1)
    col = torch.arange(T, device=logits.device, dtype=torch.int32).unsqueeze(0)
    ks = row_starts.unsqueeze(1)
    ke = row_ends.unsqueeze(1)
    mask = (col >= ks) & (col < ke)
    masked = torch.where(mask, logits, torch.full_like(logits, float("-inf")))
    k_eff = min(topk, T)
    _, indices = masked.topk(k_eff, dim=-1)
    indices = indices.to(torch.int32) - row_starts.unsqueeze(1)
    lengths = (row_ends - row_starts).unsqueeze(1)
    indices = torch.where(indices < lengths, indices, torch.full_like(indices, -1))
    if _fp8_prefill_topk_canonicalize():
        sentinel = torch.iinfo(torch.int32).max
        sortable = torch.where(
            indices >= 0, indices, torch.full_like(indices, sentinel)
        )
        sorted_idx = torch.sort(sortable, dim=-1).values
        indices = torch.where(
            sorted_idx == sentinel, torch.full_like(sorted_idx, -1), sorted_idx
        )
    out[:, :k_eff].copy_(indices)


def _run_prefill_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    compress_ratio: int,
) -> None:
    if _fp8_prefill_topk_use_torch():
        _run_prefill_topk_torch(logits, row_starts, row_ends, out, topk)
        return

    # ``logits`` is over compressed K tokens. Convert back to input-token
    # length so the 16k policy matches the user-visible prefill length.
    estimated_input_tokens = int(logits.size(1)) * int(compress_ratio)
    if (
        _fp8_prefill_fast_topk_enabled()
        and int(topk) in (512, 1024, 2048)
        and estimated_input_tokens <= _FAST_PREFILL_TOPK_MAX_INPUT_TOKENS
    ):
        lengths = (row_ends - row_starts).contiguous()
        rtp_llm_ops.fast_topk_v2_variable(
            logits, out, lengths, row_starts.contiguous(), int(topk)
        )
        return

    rtp_llm_ops.dsv4_top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        out,
        logits.size(0),
        logits.stride(0),
        logits.stride(1),
        int(topk),
        _fp8_prefill_topk_force_radix_sort(),
    )


def _fp8_prefill_score_chunk_rows() -> int:
    """Rows per non-paged FP8 prefill score chunk.

    ``fp8_mqa_indexer_score`` returns dense ``[M, T]`` logits. Long-context
    CP prefill can have hundreds of thousands of local query rows and K rows,
    so the one-shot output is not viable. Keep the default aligned with the
    other DSV4 chunk knobs; lower it with the env var on tighter HBM budgets.
    """
    return dsv4_chunk_tokens_from_env(
        "DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS",
        min_value=0,
    )


def _get_topk_workspace(device: torch.device) -> torch.Tensor:
    ws = _persistent_topk_workspace_cache.get(device)
    if ws is None:
        ws = torch.empty(
            _PERSISTENT_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=device
        )
        _persistent_topk_workspace_cache[device] = ws
    return ws


class _IndexerFP8PrefillMeta(NamedTuple):
    """Per-call FP8 prefill metadata for :class:`IndexerFP8`.

    Built once by :meth:`IndexerFP8.prepare` from the caller's
    per-step state (``bsz``, ``seqlen``, ``sp_int``, ``device``) and the
    bound INDEXER_KV pool. Replaces the old in-forward slot construction and
    masking chain so :meth:`IndexerFP8.forward` is kernel-only.

    All ``torch.Tensor`` fields are device-side, contiguous, and
    immediately consumable by the gather / score / topk kernels.

    Coordinate system:
      * **single-request**: ``ks=0``, ``ke = clamp((positions_d+1)//ratio,
        max=T)`` — request-local indices directly into the single-request
        flat K. Returned topk is request-local.
      * **varlen**: K is request-concatenated on the flat axis
        (``cu_kv_seqlens``). ``ks/ke`` are GLOBAL flat-K coords with the
        per-token request offset baked in (``ks[t] = cu_kv_seqlens[req_t]``,
        ``ke[t] = ks[t] + min((pos_t+1)//ratio, T_b)``). The TopK kernel
        processes columns ``[ks, ke)`` but returns indices relative to ``ks``
        — already in request-local ``[0, N_b)`` workspace column space,
        same contract as the B==1 path.
    """

    # ── geometry (Python scalars; cheap) ──
    bsz: int
    seqlen: int
    M: int  # = bsz * seqlen  (= T_total for flat 2D)
    sp_int: int  # legacy compat: prefix_lengths[0] under varlen, sp_int otherwise
    end_pos: int  # legacy compat: sp_per_req[0] + S_0 under varlen
    is_fresh_prefill: bool  # legacy compat: any-cont negation under varlen
    T: int  # compressed K count: total across batch (sum T_b) under varlen

    # ── Q-side ──
    freqs_cis_slice: torch.Tensor  # self.freqs_cis[sp:sp+seqlen]; pre-sliced view
    positions_d: torch.Tensor  # [M] int32 — per-token global absolute positions

    # ── per-row visible-K window for fp8_mqa_logits ──
    # See class docstring for the legacy vs varlen coord contract.
    ks: torch.Tensor  # [M] int32
    ke: torch.Tensor  # [M] int32

    # ── K-side (paged) ──
    block_table_i32: torch.Tensor  # [B, max_blocks] int32 contig
    cu_kv_seqlens: torch.Tensor  # [B+1] int32 (compressed)
    # Per-token global K offset (= ``cu_kv_seqlens[req_id_per_token]``).
    # ``None`` on the legacy B==1 path (no offset needed); populated under
    # varlen so ``forward()`` can convert global TopK indices back to
    # request-local in one ``torch.where(idx >= 0, idx - off, idx)`` launch.
    cu_kv_per_token: Optional[torch.Tensor]

    # ── Nested CompressorFP8 metadata, hoisted out of the per-call hot
    # path. Without this, ``CompressorFP8.forward(meta=None)`` rebuilds
    # ~30 small kernels (state/kv slot mapping) per layer between the
    # F.linear SGEMM and ``run_save_partial_states``. ``None`` during
    # warmup (pool not bound); forward then falls back to in-band rebuild.
    compressor_meta: Optional[CompressorMeta]


class IndexerFP8(PoolBackedModule):
    """FP8 lightning indexer. DeepGEMM-only score; nested
    ``CompressorFP8(head_dim=128)`` writes the 132B pool."""

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
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """``layer_weights`` is the framework's per-layer dict
        (``ModelWeights.weights[layer_id]``), keyed by ``W.v4_*`` enum.
        Indexer reads ``W.v4_indexer_wq_b_{w,s}``, ``W.v4_indexer_weights_proj_w``,
        and the four ``W.v4_indexer_compressor_*`` keys for its nested
        ``CompressorFP8``."""
        super().__init__()
        assert index_head_dim == INDEXER_HEAD_DIM, (
            f"IndexerFP8 locked to index_head_dim={INDEXER_HEAD_DIM} "
            f"(matches CompressorFP8 132B layout); got {index_head_dim}"
        )
        assert has_fp8_paged_mqa_logits(), (
            "deep_gemm.fp8_paged_mqa_logits not available — IndexerFP8 cannot "
            "operate without DeepGEMM. Use IndexerBF16 (or install deep_gemm)."
        )
        assert layer_weights is not None, (
            "IndexerFP8 requires layer_weights — meta-tensor / stand-alone "
            "construction is not supported (use the BF16 path for that)."
        )
        self.dim = dim
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio

        from rtp_llm.models_py.modules.dsv4.fp8.attention import _v4_fp8_linear
        from rtp_llm.utils.model_weight import W

        self.wq_b = _v4_fp8_linear(
            layer_weights[W.v4_indexer_wq_b_w],
            layer_weights[W.v4_indexer_wq_b_s],
        )
        # weights_proj is plain BF16. Pre-fold the runtime ``softmax_scale *
        # n_heads^-0.5`` into the weight at load time so prefill / decode can
        # do a single ``F.linear`` (cuBLAS GEMM) without a trailing elementwise
        # mul. New tensor — never mutate ``layer_weights`` in place.
        _wp_scale = self.softmax_scale * self.n_heads**-0.5
        self.weights_proj = (
            layer_weights[W.v4_indexer_weights_proj_w] * _wp_scale
        ).contiguous()

        # Nested compressor: 132B layout (head_dim=128).
        inner_cmp_weights = {
            "ape": layer_weights[W.v4_indexer_compressor_ape],
            "wkv": layer_weights[W.v4_indexer_compressor_wkv],
            "wgate": layer_weights[W.v4_indexer_compressor_wgate],
            "norm": layer_weights[W.v4_indexer_compressor_norm],
        }
        self.compressor = CompressorFP8(
            dim=dim,
            head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            max_batch_size=max_batch_size,
            norm_eps=norm_eps,
            rotate=True,
            compressor_weights=inner_cmp_weights,
        )
        self.max_batch_size = max_batch_size
        self._kv_cache_t = max_seq_len // compress_ratio
        self._kv_cache_d = index_head_dim
        self.freqs_cis: Optional[torch.Tensor] = None
        # Context-Parallel context bound by ``V4Transformer._propagate_cp_ctx``
        # before each prefill forward (Phase F). Decode runs with this
        # cleared. Under cp_size > 1, ``prepare()`` swaps in
        # ``cp_ctx.input_lengths_global`` for the per-request length math
        # so T (compressed-K count) reflects the gather'd global view, and
        # the nested compressor rebuilds its slot mappings post-gather
        # (CompressorFP8.forward handles that internally).
        self._cp_ctx: Optional[CPContext] = None

    # --------------------------------------------------------------
    # Pool propagation to nested compressor
    # --------------------------------------------------------------
    def _propagate_pool_to_nested(self) -> None:
        self.compressor.set_pool_context(
            self._kv_pool_view,
            self._kv_block_table,
            self._kv_eb,
            self._state_pool_3d,
            self._state_block_table,
            self._state_eb,
            state_tokens_per_block=self._state_tokens_per_block,
            kv_tokens_per_block=self._kv_tokens_per_block,
        )

    def _clear_nested_pool(self) -> None:
        compressor = getattr(self, "compressor", None)
        if compressor is not None:
            compressor.clear_pool_context()

    def clear_pool_context(self) -> None:
        super().clear_pool_context()
        self._clear_nested_pool()

    def set_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        """Bind the per-forward CPContext (Phase F / Phase-2).

        Phase-2 lifts the prior B==1 restriction: ``prepare`` already
        consumes ``cp_ctx.input_lengths_global`` as a per-request array
        (computing ``T_per_req`` elementwise), so multi-request CP just
        works at this layer. The nested compressor's CP context is
        bound by ``V4Transformer._propagate_cp_ctx`` separately (it walks
        ``indexer.compressor`` directly) so we don't need to forward
        here. Accepting ``None`` with cp_size <= 1 is the no-op.
        """
        if cp_ctx is not None and cp_ctx.cp_size > 1:
            assert (
                cp_ctx.input_lengths_global is not None
                and cp_ctx.input_lengths_global.numel() >= 1
            ), "IndexerFP8 CP requires cp_ctx.input_lengths_global"
        self._cp_ctx = cp_ctx

    def _gather_prefill_k_cache(
        self,
        attention_inputs: _IndexerFP8PrefillMeta,
        k_quant_flat: torch.Tensor,
        k_scale_buf: torch.Tensor,
    ) -> None:
        """Fill prefill indexer-K buffers from the FP8 pool.

        Shared by ``forward`` and ``forward_with_pending_nested`` so the overlap
        path keeps the same CP-sharded assembler behavior as the baseline path.
        """
        cp_ctx = getattr(self, "_cp_ctx", None)
        kv_cache_sharded = bool(
            cp_ctx is not None
            and getattr(cp_ctx, "kv_cache_sharded", False)
            and cp_ctx.cp_size > 1
        )
        if not kv_cache_sharded:
            rtp_llm_ops.cp_gather_indexer_k_quant_cache(
                self._kv_pool_view,
                k_quant_flat,
                k_scale_buf,
                attention_inputs.block_table_i32,
                attention_inputs.cu_kv_seqlens,
            )
            return

        from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler import (
            assemble_indexer_k,
            build_indexer_cp_chunk_plan,
            build_local_cu_kv_seqlens,
        )

        cu = attention_inputs.cu_kv_seqlens.to(torch.int64)
        per_req_T = (cu[1:] - cu[:-1]).contiguous()
        plan = build_indexer_cp_chunk_plan(
            cp_ctx=cp_ctx,
            per_req_total_kv_lens=per_req_T,
            block_size=self._kv_eb,
            device=k_quant_flat.device,
        )
        local_cu = build_local_cu_kv_seqlens(plan)
        local_q = torch.empty(
            (plan.total_local_T, k_quant_flat.shape[-1]),
            dtype=k_quant_flat.dtype,
            device=k_quant_flat.device,
        )
        local_s = torch.empty(
            (plan.total_local_T, k_scale_buf.shape[-1]),
            dtype=k_scale_buf.dtype,
            device=k_scale_buf.device,
        )
        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            self._kv_pool_view,
            local_q,
            local_s,
            attention_inputs.block_table_i32,
            local_cu,
        )
        assemble_indexer_k(
            plan=plan,
            local_k_quant=local_q,
            local_k_scale=local_s,
            out_k_quant=k_quant_flat,
            out_k_scale=k_scale_buf,
        )

    # --------------------------------------------------------------
    # Q-projection + RoPE helper (shared between prefill & decode)
    # --------------------------------------------------------------
    def _compute_indexer_q(
        self,
        qr: torch.Tensor,
        freqs_cis: torch.Tensor,
        batched_rope: bool = False,
    ) -> torch.Tensor:
        """qr → wq_b → unflatten → RoPE → q.

        Output layout mirrors qr's leading dims:
          * qr ``[B, S, q_lora]``  → q ``[B, S, H, D]``  (legacy 3D)
          * qr ``[T_total, q_lora]`` → q ``[T_total, H, D]`` (Phase-3a flat)

        FP8 decode/indexer runs on CUDA. RoPE is always applied through the
        shared Triton kernel, which supports both ``[B, S, H, D]`` and flat
        ``[T, H, D]`` decode shapes.
        """
        # Framework FP8 linear (CudaFp8DeepGEMMLinear) requires 2D input;
        # flatten leading dims and restore.
        with record_function_range("dsv4.fp8.indexer.compute_q.wq_b"):
            if qr.dim() > 2:
                shape = qr.shape
                q = self.wq_b(qr.reshape(-1, shape[-1])).view(
                    *shape[:-1], self.n_heads * self.head_dim
                )
            else:
                q = self.wq_b(qr)
            q = q.unflatten(-1, (self.n_heads, self.head_dim))
        with record_function_range("dsv4.fp8.indexer.compute_q.rope"):
            rope_view = q[..., -self.rope_head_dim :]
            if not rope_view.is_cuda:
                raise RuntimeError("IndexerFP8._compute_indexer_q expects CUDA tensors")
            from rtp_llm.models_py.modules.dsv4._rope_only_triton import (
                rope_only_inplace,
            )

            rope_only_inplace(rope_view, freqs_cis)
        return q

    # --------------------------------------------------------------
    # Decode (vectorized over B)
    # --------------------------------------------------------------
    def forward_decode_vectorized(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: torch.Tensor,
        out_topk_buffer: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        compressor_meta: Optional[CompressorMeta] = None,
    ) -> torch.Tensor:
        bsz, q_len = int(x.size(0)), int(x.size(1))
        if position_ids is None:
            assert q_len == 1, "decode q_len > 1 requires flat position_ids"
        ratio = self.compress_ratio
        K = self.index_topk

        self._propagate_pool_to_nested()
        try:
            # Nested compressor writes its compressed token to the 132B pool.
            self.compressor.forward_decode_vectorized(
                x,
                start_pos,
                meta=compressor_meta,
                position_ids=position_ids,
            )

            if position_ids is None:
                freqs = self.freqs_cis[start_pos.long()]
                q = self._compute_indexer_q(qr, freqs, batched_rope=True)
                compressed_len = ((start_pos + 1) // ratio).view(bsz, 1, 1)
            else:
                assert compressor_meta is not None
                pos_flat = compressor_meta.positions.reshape(-1)
                freqs = self.freqs_cis.index_select(0, pos_flat).contiguous()
                q = self._compute_indexer_q(qr, freqs, batched_rope=False)
                assert compressor_meta.compressed_lens_per_token is not None
                compressed_len = compressor_meta.compressed_lens_per_token.view(
                    bsz, q_len, 1
                )
            # ``softmax_scale * n_heads^-0.5`` is pre-folded into weights_proj at __init__.
            weights = F.linear(x, self.weights_proj)

            # Always use DeepGEMM (FP8 path). Decode uses a static score
            # width from the cache/block-table upper bound; replay updates
            # block tables and per-row lengths in-place before captured
            # kernels run. Do not derive this from live positions via a GPU
            # scalar read: that would synchronize the device and is
            # illegal during CUDA graph capture.
            T_cache = self._kv_block_table.shape[1] * self._kv_eb
            T_static = self._kv_cache_t if self._kv_cache_t > 0 else T_cache
            T_max = max(32, min(T_cache, T_static))

            q_fp8, w_fold = indexer_q_fp8_quant_fold(
                _as_bf16_contig(q), _as_bf16_contig(weights)
            )
            ctx_lens_2d = compressed_len.view(bsz, q_len)
            bt_i32 = self._kv_block_table[:bsz].to(torch.int32).contiguous()
            # ``_kv_pool_view`` is 3D ``[num_blocks, eb, 132]`` from production
            # (set by ``Attention._set_compressor_pool_context``); standalone
            # tests still pass flat 2D. Flatten to ``[total_slots, 132]``
            # (no copy — INDEXER pool is contiguous, no padding) for DeepGEMM.
            pool_2d = (
                self._kv_pool_view.flatten(0, 1)
                if self._kv_pool_view.dim() == 3
                else self._kv_pool_view
            )
            logits = fp8_paged_indexer_score(
                q_fp8,
                w_fold.view(bsz * q_len, self.n_heads),
                pool_2d,
                bt_i32,
                ctx_lens_2d,
                block_size=self._kv_eb,
                max_ctx_len=T_max,
            )  # [B*q_len, T_max] fp32
            score = logits.view(bsz, q_len, T_max)

            # TopK (with optional persistent radix-select)
            K_eff = min(K, T_max)
            score_2d = score.view(bsz * q_len, T_max)
            lengths_i32 = compressed_len.view(bsz * q_len)
            out_topk_2d = out_topk_buffer.view(bsz * q_len, K)
            if K_eff > 0 and K in (512, 1024, 2048) and _persistent_topk_enabled():
                rtp_llm_ops.dsv4_persistent_topk(
                    score_2d,
                    lengths_i32,
                    out_topk_2d,
                    _get_topk_workspace(score.device),
                    K,
                    T_max,
                )
            else:
                out_topk_buffer.fill_(-1)
                if K_eff > 0:
                    t_range = torch.arange(T_max, device=score.device).view(1, T_max)
                    score_masked = torch.where(
                        t_range < lengths_i32.view(-1, 1),
                        score_2d,
                        torch.full_like(score_2d, float("-inf")),
                    )
                    topk_idxs = score_masked.topk(K_eff, dim=-1)[1].to(torch.int32)
                    out_topk_2d[:, :K_eff].copy_(topk_idxs)
                    k_arange = torch.arange(K, device=out_topk_buffer.device).view(1, K)
                    out_topk_2d.masked_fill_(k_arange >= lengths_i32.view(-1, 1), -1)
            return out_topk_buffer
        finally:
            self._clear_nested_pool()

    # --------------------------------------------------------------
    # Prefill prepare — caller invokes once per layer-call, before forward
    # --------------------------------------------------------------
    def prepare(
        self,
        bsz: int,
        seqlen: int,
        sp_int: int,
        device: torch.device,
        kv_block_table: Optional[torch.Tensor] = None,
        kv_eb: int = 0,
        *,
        use_varlen: bool,
        batch_size: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
        write_skip_restore_window: int = 0,
    ) -> _IndexerFP8PrefillMeta:
        """Build per-call FP8 prefill metadata.

        Caller (``Attention._build_shared_prefill_meta``) invokes this once
        per layer-call. The returned bundle carries every device-side
        tensor and Python scalar :meth:`forward` needs, so the hot path
        has zero ``arange`` / ``where`` / ``zeros`` / ``contiguous``
        launches.

        ``kv_block_table`` and ``kv_eb`` describe the INDEXER_KV pool layout
        for this forward; pass them explicitly so this method does not
        rely on a stale ``set_pool_context`` snapshot. Warmup callers can
        pass ``kv_block_table=None`` / ``kv_eb=0`` and the per-row
        ``block_table_i32`` is emitted empty (matches the warmup short-
        circuit in :meth:`forward`).

        Today's caller still hands us a single-request layout (``bsz==1``
        with the full request flattened into ``seqlen``), so the
        ``cu_kv_seqlens`` we emit is ``[0, T]``. This method is otherwise
        layout-agnostic — once the caller starts feeding per-request
        ``cu_seqlens`` we'll fan that out here.
        """
        assert self.freqs_cis is not None, "IndexerFP8.prepare needs freqs_cis bound"
        ratio = self.compress_ratio
        # ``use_varlen`` is required — set by
        # ``Attention._build_csa_prefill_meta`` which receives it from
        # ``Attention._build_shared_prefill_meta`` (the single env-read
        # point + contract guard for the whole prefill stack). UT helpers
        # must pass it explicitly.
        # Phase F: under CP the framework's ``input_lengths`` is rank-local
        # (chunk_length per req). The nested ``CompressorFP8.forward``
        # all-gathers KV/score to the full ``seq_len_full`` per req, so the
        # compressed-K count ``T_b = (prefix + S_b) // ratio`` must use the
        # GLOBAL per-req length to size the score axis (ks/ke/cu_kv_seqlens
        # all index into a per-rank pool that holds compressed entries for
        # the entire global sequence). ``prefix_lengths`` is rank-invariant
        # so it doesn't need swapping. Q stays rank-local — ``positions_d``
        # uses ``position_ids`` which already carries GLOBAL absolute
        # positions on each rank-local token.
        cp_ctx = getattr(self, "_cp_ctx", None)
        cp_active = cp_ctx is not None and cp_ctx.cp_size > 1
        eff_input_lengths = input_lengths
        if cp_active and cp_ctx.input_lengths_global is not None:
            eff_input_lengths = cp_ctx.input_lengths_global
        if use_varlen:
            assert cu_seqlens is not None
            assert eff_input_lengths is not None
            assert prefix_lengths is not None
            assert position_ids is not None
            assert req_id_per_token is not None
            cu_seqlens = _flat_1d(cu_seqlens)
            eff_input_lengths = _flat_1d(eff_input_lengths)
            prefix_lengths = _flat_1d(prefix_lengths)
            position_ids = _flat_1d(position_ids)
            req_id_per_token = _flat_1d(req_id_per_token)
            assert position_ids.numel() == req_id_per_token.numel(), (
                "position_ids / req_id_per_token must have matching token counts: "
                f"{position_ids.numel()} vs {req_id_per_token.numel()}"
            )
            assert (
                cu_seqlens.numel() == batch_size + 1
            ), f"cu_seqlens must be [B+1={batch_size + 1}], got {cu_seqlens.shape}"
            assert (
                eff_input_lengths.numel() == batch_size
            ), f"input_lengths must be [B={batch_size}], got {eff_input_lengths.shape}"
            assert (
                prefix_lengths.numel() == batch_size
            ), f"prefix_lengths must be [B={batch_size}], got {prefix_lengths.shape}"
            # Per-request compressed-K count: T_b = (sp_b + S_b) // ratio.
            with record_function_range("dsv4.fp8.indexer.prepare.varlen_lengths"):
                seq_total_per_req = prefix_lengths.to(
                    device=device, dtype=torch.int64
                ) + eff_input_lengths.to(
                    device=device, dtype=torch.int64
                )  # [B]
                T_per_req = (seq_total_per_req // ratio).to(torch.int32)  # [B]
                cu_kv_seqlens = torch.zeros(
                    batch_size + 1, dtype=torch.int32, device=device
                )
                cu_kv_seqlens[1:] = torch.cumsum(T_per_req.to(torch.int64), dim=0).to(
                    torch.int32
                )
                T = int(cu_kv_seqlens[-1].item())  # total compressed K across batch
                M = int(position_ids.numel())  # T_total

                positions_d = position_ids.to(
                    device=device, dtype=torch.int32
                ).contiguous()

            # Under varlen, ``fp8_mqa_indexer_score`` consumes a single
            # request-concatenated K axis ``k_quant_flat[0..T)`` whose
            # per-request offset is ``cu_kv_seqlens[b]``. ``ks/ke`` MUST
            # be in this GLOBAL flat-K coord (kernel docstring:
            # "K start, inclusive / K end, exclusive" — same 1D K axis
            # across rows). Per-token visible window for token t in
            # request b at absolute position pos_t:
            #     ks[t] = cu_kv_seqlens[b]
            #     ke[t] = cu_kv_seqlens[b] + min((pos_t+1)//ratio, T_b)
            # The TopK kernel processes [ks, ke) but returns indices
            # relative to ks — already request-local [0, N_b).
            with record_function_range("dsv4.fp8.indexer.prepare.varlen_bounds"):
                req_id_long = req_id_per_token.to(device=device, dtype=torch.int64)
                T_per_token = T_per_req.to(torch.int64).index_select(0, req_id_long)
                cu_kv_per_token_i64 = cu_kv_seqlens.to(torch.int64).index_select(
                    0, req_id_long
                )
                ks = cu_kv_per_token_i64.to(torch.int32).contiguous()
                ke = (
                    (
                        cu_kv_per_token_i64
                        + ((positions_d.to(torch.int64) + 1) // ratio).clamp_max(
                            T_per_token
                        )
                    )
                    .to(torch.int32)
                    .contiguous()
                )
                cu_kv_per_token = cu_kv_per_token_i64.to(torch.int32).contiguous()

            # ``freqs_cis_slice`` per-token gather — RoPE angles for ``q``
            # in ``_compute_indexer_q``. Equivalent to ``self.freqs_cis[
            # sp:sp+S]`` for B == 1 contiguous range; per-token gather is
            # required when requests interleave on the flat axis.
            with record_function_range("dsv4.fp8.indexer.prepare.freqs"):
                freqs_cis_slice = self.freqs_cis.index_select(
                    0, positions_d.to(torch.long)
                )

            if kv_block_table is not None and kv_eb > 0:
                with record_function_range("dsv4.fp8.indexer.prepare.block_table"):
                    block_table_i32 = (
                        kv_block_table[:batch_size]
                        .to(device=device, dtype=torch.int32)
                        .contiguous()
                    )
            else:
                block_table_i32 = torch.empty(
                    (batch_size, 0), dtype=torch.int32, device=device
                )

            # Legacy compat scalar fields. Under varlen ``forward()`` does
            # not consume ``end_pos`` / ``is_fresh_prefill`` (the kernel-
            # native path drives off the per-request tensors); ``sp_int``
            # is still passed to ``self.compressor(x, sp, meta=...)`` but
            # is ignored there because ``meta.is_batched=True``. Keep the
            # request-0 values for diagnostics / B==1 collapse equivalence.
            end_pos = int(seq_total_per_req[0].item())
            is_fresh_prefill = sp_int == 0
        else:
            # Legacy B == 1 scalar path — unchanged, bit-equal to pre-Phase-3a.
            with record_function_range("dsv4.fp8.indexer.prepare.legacy"):
                end_pos = sp_int + seqlen
                is_fresh_prefill = sp_int == 0
                T = end_pos // ratio
                M = bsz * seqlen

                freqs_cis_slice = self.freqs_cis[sp_int : sp_int + seqlen]

                # Global Q positions for this chunk: sp..sp+S-1 (B==1 → flat [M]).
                positions_d = torch.arange(
                    sp_int, sp_int + seqlen, device=device, dtype=torch.int32
                )

                # Per-row visible-K window (causal): row r → [0, (pos+1)//ratio),
                # clamped to T (compressed K count).
                ke = ((positions_d + 1) // ratio).clamp_max(T).to(torch.int32)
                ks = torch.zeros(M, dtype=torch.int32, device=device)

            # K-side block_table — None during warmup (no pool bound).
            if kv_block_table is not None and kv_eb > 0:
                block_table_i32 = (
                    kv_block_table[:bsz]
                    .to(device=device, dtype=torch.int32)
                    .contiguous()
                )
            else:
                block_table_i32 = torch.empty(
                    (bsz, 0), dtype=torch.int32, device=device
                )
            cu_kv_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
            # B==1 → request-0 offset is 0 → request-local ≡ global; the
            # subtract in ``forward()`` becomes a no-op so we leave this
            # field as ``None`` to skip the launch entirely on the legacy path.
            cu_kv_per_token = None

        # Hoist the nested CompressorFP8's slot-mapping metadata. The
        # IndexerFP8's pool context is bound by Attention's
        # ``_set_compressor_pool_context`` BEFORE ``_build_csa_prefill_meta``
        # runs, so ``_propagate_pool_to_nested`` is safe here. Without
        # this, ``CompressorFP8.forward(meta=None)`` rebuilds the slot
        # mappings each call (~30 small kernels per layer between the
        # F.linear SGEMM and ``run_save_partial_states``). ``getattr``
        # defaults keep the stub-based UTs (which expose only the four
        # attrs ``prepare`` historically read) working — they fall through
        # to ``compressor_meta=None`` and the in-band rebuild path.
        compressor_meta: Optional[CompressorMeta] = None
        state_bt = getattr(self, "_state_block_table", None)
        state_eb = getattr(self, "_state_eb", 0)
        compressor = getattr(self, "compressor", None)
        if (
            self._kv_block_table is not None
            and self._kv_eb > 0
            and state_bt is not None
            and state_eb > 0
            and compressor is not None
        ):
            from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
                build_prepare_metadata_args,
            )

            self._propagate_pool_to_nested()
            try:
                if cp_active:
                    # Phase F: under CP the nested compressor all-gathers
                    # KV/score to the full sequence. Build matching
                    # full-sequence slot metadata here, where Attention's
                    # prefill_meta broadcast path calls ``prepare`` once per
                    # CSA ratio bucket, instead of rebuilding in every layer's
                    # nested compressor hot path.
                    assert cp_ctx is not None
                    if cp_ctx.input_lengths_global is not None:
                        with record_function_range(
                            "dsv4.fp8.indexer.prepare.cp_nested_compressor_meta"
                        ):
                            (
                                cp_positions,
                                cp_b_idx,
                                cp_seq_start_per_req,
                                cp_cu_seq_per_req,
                            ) = build_cp_full_prefill_positions(cp_ctx, device)
                            compressor_meta = compressor.prepare_metadata(
                                cp_positions,
                                cp_b_idx,
                                is_batched=True,
                                seq_start_per_req=cp_seq_start_per_req,
                                cu_seq_per_req=cp_cu_seq_per_req,
                                write_skip_restore_window=write_skip_restore_window,
                            )
                    else:
                        compressor_meta = None
                else:
                    with record_function_range(
                        "dsv4.fp8.indexer.prepare.nested_compressor_meta"
                    ):
                        cmp_args = build_prepare_metadata_args(
                            use_varlen=use_varlen,
                            device=device,
                            sp_int=sp_int,
                            seqlen=seqlen,
                            position_ids=position_ids,
                            req_id_per_token=req_id_per_token,
                            seq_start_per_req=prefix_lengths,
                            cu_seqlens=cu_seqlens,
                            write_skip_restore_window=write_skip_restore_window,
                        )
                        compressor_meta = compressor.prepare_metadata(**cmp_args)
            finally:
                self._clear_nested_pool()

        return _IndexerFP8PrefillMeta(
            bsz=bsz,
            seqlen=seqlen,
            M=M,
            sp_int=sp_int,
            end_pos=end_pos,
            is_fresh_prefill=is_fresh_prefill,
            T=T,
            freqs_cis_slice=freqs_cis_slice,
            positions_d=positions_d,
            ks=ks,
            ke=ke,
            block_table_i32=block_table_i32,
            cu_kv_seqlens=cu_kv_seqlens,
            cu_kv_per_token=cu_kv_per_token,
            compressor_meta=compressor_meta,
        )

    # --------------------------------------------------------------
    # Prefill — kernel-only hot path; metadata pre-built in ``prepare``.
    # ``x`` may be either flat ``[T_total, dim]`` (vLLM-native) or
    # legacy ``[B, S, dim]`` — output shape mirrors the input layout
    # (``[..., index_topk]``).
    # Returned indices are **raw** compressed-pool offsets in int32 with
    # ``-1`` past the per-row valid count. The caller is responsible for
    # any downstream coordinate transform / dtype cast — the topk
    # consumer (``flash_mla_sparse_fwd``) accepts ``-1`` directly so no
    # sentinel-aware post-process is needed here.
    # --------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        attention_inputs: _IndexerFP8PrefillMeta,
    ) -> torch.Tensor:
        M = attention_inputs.M
        T = attention_inputs.T
        sp = attention_inputs.sp_int
        K = self.index_topk

        # Output shape mirrors ``x`` layout: ``[T_total, K]`` for flat 2D,
        # ``[B, S, K]`` for 3D. Empty ``K=0`` for warmup / cold-start.
        out_shape = (*x.shape[:-1], K)
        empty_shape = (*x.shape[:-1], 0)

        # Warmup forward (no pool bound by framework): emit empty topk
        # (matches BF16 Indexer's warmup fallback shape). Caller in
        # ``Attention._forward_prefill_fp8_csa_hca`` concats with the SWA
        # topk; an empty trailing dim is a no-op there.
        if (
            self._kv_block_table is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return torch.full(empty_shape, -1, dtype=torch.int32, device=x.device)

        if self.compressor.freqs_cis is None:
            self.compressor.freqs_cis = self.freqs_cis

        self._propagate_pool_to_nested()
        try:
            with record_function_range("dsv4.fp8.indexer.prefill.compute_q"):
                q = self._compute_indexer_q(qr, attention_inputs.freqs_cis_slice)
            with record_function_range("dsv4.fp8.indexer.prefill.nested_compressor"):
                self.compressor(x, sp, meta=attention_inputs.compressor_meta)
            # ``softmax_scale * n_heads^-0.5`` is pre-folded into weights_proj at __init__.
            with record_function_range("dsv4.fp8.indexer.prefill.weights_proj"):
                weights = F.linear(x, self.weights_proj)

            assert (
                has_fp8_mqa_logits()
            ), "deep_gemm.fp8_mqa_logits required for IndexerFP8 prefill"
            assert self._kv_pool_view.dim() == 3, (
                "IndexerFP8 expects 3D ``_kv_pool_view`` "
                "[num_blocks, eb, 132]; got dim="
                f"{self._kv_pool_view.dim()}"
            )

            if T == 0:
                # Cold-start prefill before any compressed tokens — no K
                # to score against; emit empty topk buffer behavior.
                return torch.full(empty_shape, -1, dtype=torch.int32, device=x.device)

            # Vendored CUDA gather: reads paged FP8 K + scale directly
            # via (block_table, cu_kv_seqlens). No host-side slot mapping.
            with record_function_range("dsv4.fp8.indexer.prefill.gather_k_cache"):
                k_quant_flat = torch.empty(
                    (T, INDEXER_HEAD_DIM),
                    dtype=torch.float8_e4m3fn,
                    device=x.device,
                )
                # quant_block_size = head_dim*4/dst_scale.size(1) = 128*4/4 = 128.
                k_scale_buf = torch.empty((T, 4), dtype=torch.uint8, device=x.device)
                self._gather_prefill_k_cache(
                    attention_inputs,
                    k_quant_flat,
                    k_scale_buf,
                )
                # ``deep_gemm.fp8_mqa_logits`` expects k_scale as 1D fp32
                # contig — view uint8 [T, 4] → fp32 [T, 1] → squeeze [T].
                k_scale_flat = k_scale_buf.view(torch.float32).squeeze(-1)

            # Phase-3a part 3 fix: when ``_compute_indexer_q`` was given a
            # flat 2D qr it returns 3D ``[T, H, D]``; ``indexer_q_fp8_quant_fold``
            # requires 4D ``[B, S, H, D]`` (asserts ``q_bf16.dim() == 4``).
            # Wrap with a fake B=1 — downstream consumers immediately reshape
            # to flat ``[M, H, D]`` so the wrap is a free view.
            q_for_quant = q if q.dim() == 4 else q.unsqueeze(0)
            w_for_quant = weights if weights.dim() == 3 else weights.unsqueeze(0)
            with record_function_range("dsv4.fp8.indexer.prefill.quant_q"):
                q_fp8, w_fold = indexer_q_fp8_quant_fold(
                    _as_bf16_contig(q_for_quant), _as_bf16_contig(w_for_quant)
                )

            q_score = q_fp8.view(M, self.n_heads, INDEXER_HEAD_DIM)
            w_score = w_fold.view(M, self.n_heads)
            score_chunk_rows = _fp8_prefill_score_chunk_rows()
            chunked_score = score_chunk_rows > 0 and M > score_chunk_rows
            if not chunked_score:
                score_chunk_rows = M
            out_buf = torch.empty((M, K), dtype=torch.int32, device=x.device)

            # Vendored CUDA per-row TopK over [ks[r], ke[r]). Causal mask is
            # implicit via ke = (q_pos+1)//ratio clamped to T; padding past
            # per-row valid count is ``-1`` from the kernel. For long prefill,
            # score in row chunks because DeepGEMM returns dense [rows, T].
            for row_start in range(0, M, score_chunk_rows):
                row_end = min(M, row_start + score_chunk_rows)
                with record_function_range("dsv4.fp8.indexer.prefill.score"):
                    logits = fp8_mqa_indexer_score(
                        q_score[row_start:row_end],
                        w_score[row_start:row_end],
                        k_quant_flat,
                        k_scale_flat,
                        attention_inputs.ks[row_start:row_end],
                        attention_inputs.ke[row_start:row_end],
                        clean_logits=False,
                    )  # [chunk_rows, T] fp32

                with record_function_range("dsv4.fp8.indexer.prefill.topk"):
                    _run_prefill_topk(
                        logits,
                        attention_inputs.ks[row_start:row_end],
                        attention_inputs.ke[row_start:row_end],
                        out_buf[row_start:row_end],
                        K,
                        self.compress_ratio,
                    )
                del logits

            return out_buf.view(out_shape)
        finally:
            self._clear_nested_pool()

    # --------------------------------------------------------------
    # Overlap orchestration entry points
    #
    # Companion pair to :meth:`forward` for the attention overlap path:
    # ``start_prefill_nested_compressor`` queues the nested compressor's
    # CP all-gather on ``cp_gather_stream`` without waiting, and the
    # returned ``_CompressorPending`` is drained inside
    # :meth:`forward_with_pending_nested` in place of the synchronous
    # ``self.compressor(x, sp, meta=...)`` call that ``forward`` issues.
    #
    # Non-overlap callers must keep using :meth:`forward` unchanged.
    # --------------------------------------------------------------
    def start_prefill_nested_compressor(
        self,
        x: torch.Tensor,
        sp_int: int,
        *,
        meta: Optional[CompressorMeta],
        cp_gather_stream: Optional[Any] = None,
        profile_label: Optional[str] = None,
    ) -> Optional[_CompressorPending]:
        """Begin the nested compressor's CP all-gather without waiting.

        Mirrors the first half of :meth:`forward` up to the synchronous
        compressor call: warmup gate (returns ``None`` when no pool is
        bound), ``freqs_cis`` propagation, and pool propagation to the
        nested compressor. The fused KV/gate projection + NCCL gather
        are then enqueued on the caller's ``cp_gather_stream`` via
        :meth:`CompressorFP8.start_prefill`.

        ``None`` return means warmup — callers must either fall back to
        :meth:`forward` (which also returns the empty-topk shape) or
        pass ``None`` straight to :meth:`forward_with_pending_nested`
        (which detects warmup via the same pool check and returns the
        empty-topk shape, after a no-op ``finish_prefill(None)``).
        """
        if (
            self._kv_block_table is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None
        if self.compressor.freqs_cis is None:
            self.compressor.freqs_cis = self.freqs_cis
        self._propagate_pool_to_nested()
        kwargs = {
            "meta": meta,
            "cp_gather_stream": cp_gather_stream,
        }
        if profile_label is not None:
            kwargs["profile_label"] = profile_label
        return self.compressor.start_prefill(x, sp_int, **kwargs)

    def forward_with_pending_nested(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        attention_inputs: _IndexerFP8PrefillMeta,
        nested_pending: Optional[_CompressorPending],
        before_gather_k: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """Overlap-variant of :meth:`forward`.

        Identical to :meth:`forward` byte-for-byte except the nested
        compressor is drained via ``self.compressor.finish_prefill``
        on the supplied ``nested_pending`` (produced by
        :meth:`start_prefill_nested_compressor`) instead of a fresh
        synchronous call. Same warmup early-return, same try/finally
        pool clear, same gather_k → quant_q → score → topk path.

        ``before_gather_k`` is an optional orchestrator fence. CSA overlap uses
        it to wait the main compressor's CP gather after nested write +
        compute_q/weights projection, but before this method reads the nested
        INDEXER_KV pool and runs the numerically sensitive score/topk chain.

        ``nested_pending`` is ``None`` on the warmup path; the call to
        ``finish_prefill(None)`` is a no-op so the pairing is safe to
        chain unconditionally.
        """
        M = attention_inputs.M
        T = attention_inputs.T
        K = self.index_topk

        out_shape = (*x.shape[:-1], K)
        empty_shape = (*x.shape[:-1], 0)

        if (
            self._kv_block_table is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            # Warmup mirror of ``forward``: emit empty topk. The pending
            # here is always ``None`` (start_prefill_nested_compressor
            # also short-circuits on warmup); the explicit no-op drain
            # keeps the orchestrator's symmetric pairing contract clear.
            self.compressor.finish_prefill(nested_pending)
            return torch.full(empty_shape, -1, dtype=torch.int32, device=x.device)

        if self.compressor.freqs_cis is None:
            self.compressor.freqs_cis = self.freqs_cis
        # ``set_pool_context`` is idempotent — re-propagating is safe and
        # mirrors ``forward``'s unconditional propagate so callers that
        # entered through this path without start_prefill_nested_compressor
        # (e.g. tests) still work.
        self._propagate_pool_to_nested()
        try:
            with record_function_range("dsv4.fp8.indexer.prefill.compute_q"):
                q = self._compute_indexer_q(qr, attention_inputs.freqs_cis_slice)
            with record_function_range("dsv4.fp8.indexer.prefill.nested_compressor"):
                self.compressor.finish_prefill(nested_pending)
            with record_function_range("dsv4.fp8.indexer.prefill.weights_proj"):
                weights = F.linear(x, self.weights_proj)

            assert (
                has_fp8_mqa_logits()
            ), "deep_gemm.fp8_mqa_logits required for IndexerFP8 prefill"
            assert self._kv_pool_view.dim() == 3, (
                "IndexerFP8 expects 3D ``_kv_pool_view`` "
                "[num_blocks, eb, 132]; got dim="
                f"{self._kv_pool_view.dim()}"
            )

            if T == 0:
                return torch.full(empty_shape, -1, dtype=torch.int32, device=x.device)

            if before_gather_k is not None:
                with record_function_range("dsv4.fp8.indexer.prefill.before_gather_k"):
                    before_gather_k()

            with record_function_range("dsv4.fp8.indexer.prefill.gather_k_cache"):
                k_quant_flat = torch.empty(
                    (T, INDEXER_HEAD_DIM),
                    dtype=torch.float8_e4m3fn,
                    device=x.device,
                )
                k_scale_buf = torch.empty((T, 4), dtype=torch.uint8, device=x.device)
                self._gather_prefill_k_cache(
                    attention_inputs,
                    k_quant_flat,
                    k_scale_buf,
                )
                k_scale_flat = k_scale_buf.view(torch.float32).squeeze(-1)

            q_for_quant = q if q.dim() == 4 else q.unsqueeze(0)
            w_for_quant = weights if weights.dim() == 3 else weights.unsqueeze(0)
            with record_function_range("dsv4.fp8.indexer.prefill.quant_q"):
                q_fp8, w_fold = indexer_q_fp8_quant_fold(
                    _as_bf16_contig(q_for_quant), _as_bf16_contig(w_for_quant)
                )

            q_score = q_fp8.view(M, self.n_heads, INDEXER_HEAD_DIM)
            w_score = w_fold.view(M, self.n_heads)
            score_chunk_rows = _fp8_prefill_score_chunk_rows()
            chunked_score = score_chunk_rows > 0 and M > score_chunk_rows
            if not chunked_score:
                score_chunk_rows = M
            out_buf = torch.empty((M, K), dtype=torch.int32, device=x.device)

            for row_start in range(0, M, score_chunk_rows):
                row_end = min(M, row_start + score_chunk_rows)
                with record_function_range("dsv4.fp8.indexer.prefill.score"):
                    logits = fp8_mqa_indexer_score(
                        q_score[row_start:row_end],
                        w_score[row_start:row_end],
                        k_quant_flat,
                        k_scale_flat,
                        attention_inputs.ks[row_start:row_end],
                        attention_inputs.ke[row_start:row_end],
                        clean_logits=False,
                    )

                with record_function_range("dsv4.fp8.indexer.prefill.topk"):
                    _run_prefill_topk(
                        logits,
                        attention_inputs.ks[row_start:row_end],
                        attention_inputs.ke[row_start:row_end],
                        out_buf[row_start:row_end],
                        K,
                        self.compress_ratio,
                    )
                del logits

            return out_buf.view(out_shape)
        finally:
            self._clear_nested_pool()
