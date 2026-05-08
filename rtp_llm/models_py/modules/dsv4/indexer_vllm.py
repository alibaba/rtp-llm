"""DeepSeek-V4 lightning Indexer — vLLM-flow port, BF16 compute path.

Direct adaptation of ``new_ft/.../dsv4/fp8/indexer.py`` (the FP8 / DeepGEMM
indexer) with two changes:

  1. **BF16 score** instead of FP8.  The FP8 path's
     ``indexer_q_fp8_quant_fold`` + ``cp_gather_indexer_k_quant_cache`` +
     ``fp8_mqa_indexer_score`` / ``fp8_paged_indexer_score`` chain is
     replaced with: BF16 K gather from the BF16 INDEXER_KV pool (same one
     :class:`CompressorVLLM` writes) → :func:`v4_indexer_score`
     (BF16 fused score Triton kernel, identical math to the legacy
     :class:`Indexer`).
  2. **Nested compressor is locked to** :class:`CompressorVLLM`
     (head_dim=128).  No env-switch fallback to legacy ``Compressor`` —
     this class is a single-flavor companion to the vLLM compressor.

Kept verbatim from the FP8 source:

  * ``prepare(...)`` builds per-call metadata once, off the hot path.
    ``forward(x, qr, meta)`` then runs as a kernel-only sequence with no
    ``arange`` / ``where`` / ``zeros`` launches in the body.
  * ``forward_decode_vectorized(x, qr, start_pos, out_topk_buffer)`` —
    same in/out signature as the FP8 class, drop-in compatible with the
    decode caller.
  * Vendored TopK kernels — direct calls, byte-for-byte the same as
    the FP8 source:
      - decode → ``rtp_llm_ops.persistent_topk`` (same vendored
        radix-select kernel the source calls as
        ``rtp_llm_ops.dsv4_persistent_topk``; only the op name differs
        in this project's binding). Gated by ``DSV4_PERSISTENT_TOPK``
        with the same fallback to ``score.topk`` + length mask.
      - prefill → ``rtp_llm_ops.dsv4_top_k_per_row_prefill`` (per-row
        ``[ks[r], ke[r])`` topk vendored from vLLM
        ``csrc/sampler.cu::top_k_per_row_prefill``). Same op name as
        the source.

Public API parity:

  * Prefill: ``forward(x, qr, meta)`` with ``meta = prepare(...)``.
    Returns **raw compressed-pool offsets** in int32 with ``-1`` past the
    per-row valid count — exactly the FP8 source's contract; the caller
    is responsible for any downstream coordinate transform / dtype cast.
  * Decode: ``forward_decode_vectorized(x, qr, start_pos, out_topk_buffer)``
    fills ``out_topk_buffer`` in-place; returns the same tensor.
  * No CP support — ``set_cp_ctx`` accepts ``None`` only and refuses any
    real context.  CP scenarios must continue to use the legacy BF16
    :class:`Indexer`.
"""

from __future__ import annotations

import os
from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score
from rtp_llm.models_py.modules.dsv4._pool_triton import masked_gather_from_pool
from rtp_llm.models_py.modules.dsv4.compressor_vllm import CompressorVLLM
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


_INDEXER_HEAD_DIM = 128


# Persistent radix-select TopK — vendored CUDA kernel binding. Mirrors
# the source FP8 indexer's gating: same env var name, same K∈{512,1024,
# 2048} predicate, same workspace cache. Only the op name differs (this
# project exposes the same vendored kernel as ``persistent_topk`` rather
# than ``dsv4_persistent_topk``).
_PERSISTENT_TOPK_OK = hasattr(rtp_llm_ops, "persistent_topk")
_PERSISTENT_TOPK_WORKSPACE_SIZE = 1024 * 1024  # 1 MB
_persistent_topk_workspace_cache: Dict[torch.device, torch.Tensor] = {}


def _persistent_topk_enabled() -> bool:
    if not _PERSISTENT_TOPK_OK:
        return False
    return os.environ.get("DSV4_PERSISTENT_TOPK", "1") != "0"


def _get_topk_workspace(device: torch.device) -> torch.Tensor:
    ws = _persistent_topk_workspace_cache.get(device)
    if ws is None:
        ws = torch.empty(
            _PERSISTENT_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=device
        )
        _persistent_topk_workspace_cache[device] = ws
    return ws


class _IndexerVLLMPrefillMeta(NamedTuple):
    """Per-call BF16 prefill metadata for :class:`IndexerVLLM`.

    Built once by :meth:`IndexerVLLM.prepare` from the caller's per-step
    state (``bsz``, ``seqlen``, ``sp_int``, ``device``) and the bound
    INDEXER_KV pool. Replaces the in-forward ``torch.arange`` / ``where``
    / ``contiguous`` chain so :meth:`IndexerVLLM.forward` is kernel-only.

    All ``torch.Tensor`` fields are device-side, contiguous, and
    immediately consumable by the gather / score / topk kernels.

    Coordinate system: K-pool slots and TopK indices are **raw**
    compressed-pool offsets starting at 0 — no global SWA offset baked in.
    """

    # ── geometry (Python scalars; cheap) ──
    bsz: int
    seqlen: int
    M: int          # = bsz * seqlen
    sp_int: int    # absolute prefix length
    end_pos: int   # = sp_int + seqlen
    is_fresh_prefill: bool  # = (sp_int == 0)
    T: int         # compressed K count = end_pos // ratio

    # ── Q-side ──
    freqs_cis_slice: torch.Tensor  # self.freqs_cis[sp:sp+seqlen]; pre-sliced view
    positions_d: torch.Tensor       # [M] int32 — global positions sp..sp+S-1
    q_pos: torch.Tensor             # [B, S] int32 (== positions_d viewed)

    # ── per-row visible-K window (causal) ──
    # ks[r] = 0; ke[r] = clamp((positions_d[r]+1)//ratio, max=T).
    # Mirrors the FP8 source's ``_IndexerFP8PrefillMeta`` exactly so the
    # ``dsv4_top_k_per_row_prefill`` call site is byte-for-byte identical.
    ks: torch.Tensor                # [M] int32
    ke: torch.Tensor                # [M] int32

    # ── K-side gather (compressed pool slots 0..T-1) ──
    # k_slot_mapping[c] = block_table[b, c // eb] * eb + (c % eb), or -1 / 0
    # depending on `k_valid[c]`. Both are 1D length T (bsz==1 today).
    k_slot_mapping: torch.Tensor    # [T] int64
    k_valid: torch.Tensor           # [T] bool


class IndexerVLLM(nn.Module):
    """vLLM-flow lightning indexer with BF16 score.

    Single-flavor companion to :class:`CompressorVLLM` (head_dim=128). No
    runtime branching on quantization / DeepGEMM availability — pick this
    class at construction time when you want the per-token state-pool
    flow + BF16 score pipeline.
    """

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
        Reads ``W.v4_indexer_wq_b_{w,s}``, ``W.v4_indexer_weights_proj_w``,
        and the four ``W.v4_indexer_compressor_*`` keys for the nested
        :class:`CompressorVLLM`."""
        super().__init__()
        assert index_head_dim == _INDEXER_HEAD_DIM, (
            f"IndexerVLLM locked to index_head_dim={_INDEXER_HEAD_DIM} "
            f"(matches CompressorVLLM 128-dim BF16 KV slot); got {index_head_dim}"
        )
        assert layer_weights is not None, (
            "IndexerVLLM requires layer_weights — meta-tensor / stand-alone "
            "construction is not supported (use the legacy BF16 path for that)."
        )
        self.dim = dim
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio

        from rtp_llm.models_py.modules.dsv4.attention import _v4_fp8_linear
        from rtp_llm.utils.model_weight import W

        # wq_b is a plain weight-only FP8 linear (Q LoRA up-projection).
        # The "FP8" here is a *weight* quantization (storage saving) — the
        # math runs in BF16/FP32 and is independent of the indexer's score
        # precision path. Keep it as-is.
        self.wq_b = _v4_fp8_linear(
            layer_weights[W.v4_indexer_wq_b_w],
            layer_weights[W.v4_indexer_wq_b_s],
        )
        # Pre-fold ``softmax_scale * n_heads^-0.5`` into weights_proj at
        # load time so prefill / decode each do a single ``F.linear``
        # (cuBLAS GEMM) without a trailing elementwise mul. New tensor —
        # never mutate ``layer_weights`` in place.
        _wp_scale = self.softmax_scale * self.n_heads**-0.5
        self.weights_proj = (
            layer_weights[W.v4_indexer_weights_proj_w] * _wp_scale
        ).contiguous()

        # Nested compressor: locked to vLLM flow, head_dim=128.
        inner_cmp_weights = {
            "ape": layer_weights[W.v4_indexer_compressor_ape],
            "wkv": layer_weights[W.v4_indexer_compressor_wkv],
            "wgate": layer_weights[W.v4_indexer_compressor_wgate],
            "norm": layer_weights[W.v4_indexer_compressor_norm],
        }
        self.compressor = CompressorVLLM(
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

        # Pool context — bound per forward call by Attention via
        # ``set_pool_context`` and cleared via ``clear_pool_context``.
        self._kv_pool_view: Optional[torch.Tensor] = None
        self._kv_block_table: Optional[torch.Tensor] = None
        self._kv_eb: int = 0
        self._state_pool_view: Optional[torch.Tensor] = None
        self._state_block_table: Optional[torch.Tensor] = None
        self._state_eb: int = 0

    # ------------------------------------------------------------------
    # Pool context lifecycle (mirrors BF16 Indexer surface)
    # ------------------------------------------------------------------
    def set_pool_context(
        self,
        kv_pool_view: Optional[torch.Tensor],
        kv_block_table: Optional[torch.Tensor],
        kv_eb: int,
        state_pool_view: Optional[torch.Tensor],
        state_block_table: Optional[torch.Tensor],
        state_eb: int,
    ) -> None:
        self._kv_pool_view = kv_pool_view
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb
        self._state_pool_view = state_pool_view
        self._state_block_table = state_block_table
        self._state_eb = state_eb

    def clear_pool_context(self) -> None:
        self._kv_pool_view = None
        self._kv_block_table = None
        self._kv_eb = 0
        self._state_pool_view = None
        self._state_block_table = None
        self._state_eb = 0

    def _propagate_pool_to_nested(self) -> None:
        self.compressor.set_pool_context(
            self._kv_pool_view,
            self._kv_block_table,
            self._kv_eb,
            self._state_pool_view,
            self._state_block_table,
            self._state_eb,
        )

    def _clear_nested_pool(self) -> None:
        self.compressor.clear_pool_context()

    def set_cp_ctx(self, cp_ctx) -> None:
        # Mirrors IndexerFP8: transformer's global ``_propagate_cp_ctx``
        # calls this on every indexer; accept ``None`` (CP off) silently
        # and refuse a real ctx loudly.
        assert cp_ctx is None, "IndexerVLLM does not support Context-Parallel"

    # ------------------------------------------------------------------
    # Q-projection + RoPE helper (shared between prefill & decode)
    # ------------------------------------------------------------------
    def _compute_indexer_q(
        self,
        qr: torch.Tensor,
        freqs_cis: torch.Tensor,
        batched_rope: bool = False,
    ) -> torch.Tensor:
        """qr → wq_b → unflatten → RoPE → q [B, S, H, D]."""
        if qr.dim() > 2:
            shape = qr.shape
            q = self.wq_b(qr.reshape(-1, shape[-1])).view(
                *shape[:-1], self.n_heads * self.head_dim
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
    # K-side gather: dense [bsz, T, D] BF16 from the BF16 INDEXER_KV pool
    # ------------------------------------------------------------------
    def _gather_compressed_k(
        self,
        bsz: int,
        T: int,
        device: torch.device,
        slot_mapping: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dense ``[bsz, T, head_dim]`` BF16 gather from the INDEXER_KV pool.

        When ``slot_mapping`` / ``valid`` are pre-built (prefill via
        :meth:`prepare`), they are used directly; otherwise (decode) we
        compute them here for the full ``T = self._kv_cache_t`` window.
        """
        D = self._kv_cache_d
        if (
            self._kv_pool_view is None
            or self._kv_block_table is None
            or self._kv_eb <= 0
            or T <= 0
        ):
            return torch.zeros(bsz, T, D, dtype=torch.bfloat16, device=device)
        if slot_mapping is None:
            eb = self._kv_eb
            max_blocks = int(self._kv_block_table.shape[1])
            pool_capacity = max_blocks * eb
            pos = torch.arange(T, device=device, dtype=torch.long)
            in_capacity = pos < pool_capacity
            safe_pos = torch.where(in_capacity, pos, torch.zeros_like(pos))
            block_in_seq = safe_pos // eb
            in_block = safe_pos % eb
            bt_long = self._kv_block_table[:bsz].to(device=device, dtype=torch.long)
            b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
            block_id = bt_long[b_idx, block_in_seq.unsqueeze(0).expand(bsz, -1)]
            valid = (block_id > 0) & in_capacity.unsqueeze(0)
            slot_mapping = torch.where(
                valid,
                block_id * eb + in_block.unsqueeze(0),
                torch.zeros_like(block_id),
            )
        return masked_gather_from_pool(
            self._kv_pool_view,
            slot_mapping,
            valid,
            out_shape=(bsz, T, D),
            dtype=torch.bfloat16,
        ).contiguous()

    # ------------------------------------------------------------------
    # Prefill prepare — caller invokes once per layer-call, before forward
    # ------------------------------------------------------------------
    def prepare(
        self,
        bsz: int,
        seqlen: int,
        sp_int: int,
        device: torch.device,
        kv_block_table: Optional[torch.Tensor] = None,
        kv_eb: int = 0,
    ) -> _IndexerVLLMPrefillMeta:
        """Build per-call BF16 prefill metadata.

        Caller invokes this once per layer-call. The returned bundle
        carries every device-side tensor and Python scalar
        :meth:`forward` needs, so the hot path has zero
        ``arange`` / ``where`` / ``zeros`` / ``contiguous`` launches.

        ``kv_block_table`` and ``kv_eb`` describe the INDEXER_KV pool
        layout for this forward; pass them explicitly so this method does
        not rely on a stale ``set_pool_context`` snapshot. Warmup callers
        can pass ``kv_block_table=None`` / ``kv_eb=0`` and the K-side
        gather metadata is emitted as empty placeholders (matches the
        warmup short-circuit in :meth:`forward`).

        Today's caller still hands us a single-request layout (``bsz==1``
        with the full request flattened into ``seqlen``); the gather
        metadata is sized accordingly. Once the caller starts feeding
        per-request batches we'll fan that out here.
        """
        assert (
            self.freqs_cis is not None
        ), "IndexerVLLM.prepare needs freqs_cis bound"
        ratio = self.compress_ratio
        end_pos = sp_int + seqlen
        is_fresh_prefill = sp_int == 0
        T = end_pos // ratio
        M = bsz * seqlen

        freqs_cis_slice = self.freqs_cis[sp_int : sp_int + seqlen]

        # Global Q positions for this chunk: sp..sp+S-1 (B==1 → flat [M]).
        positions_d = torch.arange(
            sp_int, sp_int + seqlen, device=device, dtype=torch.int32
        )
        q_pos = positions_d.view(bsz, seqlen)

        # Per-row visible-K window (causal): row r → (pos+1)//ratio,
        # clamped to T (compressed K count). v4_indexer_score uses q_pos
        # directly for its in-kernel mask; we surface ``ke`` separately
        # because the per-row TopK kernel takes lengths-style input.
        ke = ((positions_d + 1) // ratio).clamp_max(T).to(torch.int32)
        ks = torch.zeros(M, dtype=torch.int32, device=device)

        # K-side gather slot_mapping: maps compressed-token index 0..T-1
        # to BF16 INDEXER_KV pool slot.
        if kv_block_table is not None and kv_eb > 0 and T > 0:
            max_blocks = int(kv_block_table.shape[1])
            pool_capacity = max_blocks * kv_eb
            kpos = torch.arange(T, device=device, dtype=torch.long)
            in_capacity = kpos < pool_capacity
            safe_pos = torch.where(in_capacity, kpos, torch.zeros_like(kpos))
            block_in_seq = safe_pos // kv_eb
            in_block = safe_pos % kv_eb
            bt_long = kv_block_table[:bsz].to(device=device, dtype=torch.long)
            b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
            block_id = bt_long[b_idx, block_in_seq.unsqueeze(0).expand(bsz, -1)]
            k_valid = (block_id > 0) & in_capacity.unsqueeze(0)
            k_slot_mapping = torch.where(
                k_valid,
                block_id * kv_eb + in_block.unsqueeze(0),
                torch.zeros_like(block_id),
            )
        else:
            k_slot_mapping = torch.empty((bsz, 0), dtype=torch.long, device=device)
            k_valid = torch.empty((bsz, 0), dtype=torch.bool, device=device)

        return _IndexerVLLMPrefillMeta(
            bsz=bsz,
            seqlen=seqlen,
            M=M,
            sp_int=sp_int,
            end_pos=end_pos,
            is_fresh_prefill=is_fresh_prefill,
            T=T,
            freqs_cis_slice=freqs_cis_slice,
            positions_d=positions_d,
            q_pos=q_pos,
            ks=ks,
            ke=ke,
            k_slot_mapping=k_slot_mapping,
            k_valid=k_valid,
        )

    # ------------------------------------------------------------------
    # Prefill — kernel-only hot path; metadata pre-built in ``prepare``.
    # Returns **raw** compressed-pool offsets in int32 with ``-1`` past
    # the per-row valid count. The caller is responsible for any
    # downstream coordinate transform / dtype cast (matches the FP8
    # source's contract).
    # ------------------------------------------------------------------
    def forward_with_meta(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        attention_inputs: _IndexerVLLMPrefillMeta,
    ) -> torch.Tensor:
        bsz = attention_inputs.bsz
        T = attention_inputs.T
        sp = attention_inputs.sp_int
        K = self.index_topk

        # Output shape mirrors ``x`` layout: ``[T_total, K]`` for flat 2D,
        # ``[B, S, K]`` for 3D. Empty ``K=0`` for warmup / cold-start.
        out_shape = (*x.shape[:-1], K)
        empty_shape = (*x.shape[:-1], 0)

        # Warmup (no pool bound by framework): emit empty topk (matches
        # FP8 / BF16 Indexer warmup fallback shape). Caller concats with
        # the SWA topk; an empty trailing dim is a no-op there.
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
            q = self._compute_indexer_q(qr, attention_inputs.freqs_cis_slice)
            # Nested compressor writes the current chunk's compressed K
            # into the BF16 INDEXER_KV pool (and per-token state into
            # INDEXER_STATE).
            self.compressor(x, sp)
            # ``softmax_scale * n_heads^-0.5`` is pre-folded into
            # weights_proj at __init__.
            weights = F.linear(x, self.weights_proj)

            if T == 0:
                # Cold-start prefill before any compressed tokens — no K
                # to score against; emit empty topk buffer behavior.
                return torch.full(
                    empty_shape, -1, dtype=torch.int32, device=x.device
                )

            # BF16 K gather: dense [bsz, T, 128] from the INDEXER_KV pool.
            kv = self._gather_compressed_k(
                bsz,
                T,
                x.device,
                slot_mapping=attention_inputs.k_slot_mapping,
                valid=attention_inputs.k_valid,
            )

            # BF16 fused score: applies the causal mask via q_pos
            # (kv_col >= (q_pos+1)//ratio → -inf), no separate masking.
            index_score = v4_indexer_score(
                q.contiguous(),
                kv,
                weights,
                q_pos=attention_inputs.q_pos.contiguous(),
                compress_ratio=self.compress_ratio,
            )

            # Vendored CUDA per-row TopK over [ks[r], ke[r]). Causal
            # mask is implicit via ke = (q_pos+1)//ratio clamped to T;
            # padding past per-row valid count is ``-1`` from the
            # kernel. Byte-for-byte the same call as the FP8 source's
            # ``rtp_llm_ops.dsv4_top_k_per_row_prefill`` block.
            M = attention_inputs.M
            logits = index_score.view(M, T).contiguous()
            out_buf = torch.empty(
                (M, K), dtype=torch.int32, device=logits.device
            )
            rtp_llm_ops.dsv4_top_k_per_row_prefill(
                logits,
                attention_inputs.ks,
                attention_inputs.ke,
                out_buf,
                M,
                logits.stride(0),
                logits.stride(1),
                K,
            )
            return out_buf.view(out_shape)
        finally:
            self._clear_nested_pool()

    # ------------------------------------------------------------------
    # Backward-compat ``forward(x, qr, start_pos, offset, sequence_lengths)``
    # entry. Lets attention.py's existing prefill call sites keep using
    # the legacy ``self.indexer(x, qr, start_pos, offset)`` invocation —
    # we just build the metadata internally and re-add ``offset`` so the
    # returned indices live in the [SWA | compressed] coordinate space
    # the legacy ``Indexer`` produces.
    #
    # Returns ``[B, S, K]`` int64 (matches legacy ``Indexer.forward``);
    # raw int32 indices live behind :meth:`forward_with_meta` for callers
    # that want the FP8 source's contract.
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos,
        offset: int = 0,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        is_batched = (
            isinstance(start_pos, torch.Tensor) and start_pos.numel() > 1
        )
        is_prefill = seqlen > 1

        if is_batched and is_prefill:
            # Source FP8 indexer assumes single-request layout (bsz==1).
            # For BF16 vLLM batched prefill we per-row dispatch, mirroring
            # the legacy Indexer's ``_forward_batched_prefill``.
            return self._forward_batched_prefill(
                x, qr, start_pos, offset, sequence_lengths
            )

        device = x.device
        if isinstance(start_pos, torch.Tensor):
            sp_int = int(start_pos.item()) if start_pos.numel() == 1 else 0
        else:
            sp_int = int(start_pos)

        if self.compressor.freqs_cis is None:
            self.compressor.freqs_cis = self.freqs_cis

        if is_prefill:
            meta = self.prepare(
                bsz=bsz,
                seqlen=seqlen,
                sp_int=sp_int,
                device=device,
                kv_block_table=self._kv_block_table,
                kv_eb=self._kv_eb,
            )
            topk_int32 = self.forward_with_meta(x, qr, meta)
        else:
            # bsz>=1 single-step decode (q_len==1 per request). Route
            # through the vectorized decode path with a temporary buffer.
            K = self.index_topk
            tmp_buf = torch.full(
                (bsz, 1, K), -1, dtype=torch.int32, device=device
            )
            sp_t = (
                start_pos
                if isinstance(start_pos, torch.Tensor)
                else torch.full((bsz,), sp_int, device=device, dtype=torch.int32)
            )
            self.forward_decode_vectorized(x, qr, sp_t, tmp_buf)
            topk_int32 = tmp_buf

        # Lift to long + add SWA prefix offset (legacy Indexer contract).
        topk_long = topk_int32.long()
        if isinstance(offset, torch.Tensor) or offset != 0:
            topk_long = torch.where(
                topk_long >= 0, topk_long + offset, topk_long
            )
        return topk_long

    def forward_decode(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: torch.Tensor,
        out_topk_buffer: torch.Tensor,
    ) -> torch.Tensor:
        """Alias to :meth:`forward_decode_vectorized` — vLLM flow has no
        scalar-loop decode body, so the two are identical."""
        return self.forward_decode_vectorized(x, qr, start_pos, out_topk_buffer)

    def _forward_batched_prefill(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: torch.Tensor,
        offset: int,
        sequence_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-row prefill dispatch for ``bsz>1, seqlen>1``. Each row
        narrows the bound INDEXER_KV / INDEXER_STATE block tables to its
        own slice and calls back into the bsz==1 path."""
        bsz = int(x.size(0))
        max_S = int(x.size(1))
        device = x.device

        if sequence_lengths is None:
            seq_t = torch.full((bsz,), max_S, device=device, dtype=torch.long)
        else:
            seq_t = sequence_lengths.to(device=device, dtype=torch.long)
        sp_t = start_pos.to(device=device, dtype=torch.long)

        end_pos_t = sp_t + seq_t
        k_per_row = torch.minimum(
            torch.full_like(end_pos_t, self.index_topk),
            end_pos_t // self.compress_ratio,
        )
        K_max = int(k_per_row.max().item()) if bsz > 0 else 0
        if K_max <= 0:
            return torch.full(
                (bsz, max_S, 0), -1, dtype=torch.long, device=device
            )
        out = torch.full(
            (bsz, max_S, K_max), -1, dtype=torch.long, device=device
        )

        saved_kv_bt = self._kv_block_table
        saved_state_bt = self._state_block_table
        try:
            for b in range(bsz):
                seq_b = int(seq_t[b].item())
                if seq_b == 0:
                    continue
                sp_b = int(sp_t[b].item())
                x_b = x[b : b + 1, :seq_b]
                qr_b = qr[b : b + 1, :seq_b]
                self._kv_block_table = (
                    saved_kv_bt[b : b + 1] if saved_kv_bt is not None else None
                )
                self._state_block_table = (
                    saved_state_bt[b : b + 1]
                    if saved_state_bt is not None
                    else None
                )
                topk_b = self.forward(x_b, qr_b, sp_b, offset)  # [1, seq_b, K_b]
                k_b = int(topk_b.size(-1))
                if k_b > 0:
                    out[b : b + 1, :seq_b, :k_b] = topk_b
        finally:
            self._kv_block_table = saved_kv_bt
            self._state_block_table = saved_state_bt

        return out

    # ------------------------------------------------------------------
    # Optional debug-prefix attribute used by ``attention.py`` when the
    # _record_tensor gate is on. We don't actually emit per-tensor records
    # here; just absorb the attribute so callers can set/clear it without
    # crashing.
    # ------------------------------------------------------------------
    _dbg_prefix: Optional[str] = None

    # ------------------------------------------------------------------
    # Decode (vectorized over B)
    # ------------------------------------------------------------------
    def forward_decode_vectorized(
        self,
        x: torch.Tensor,        # [B, 1, dim] bf16
        qr: torch.Tensor,       # [B, 1, q_lora_rank] bf16
        start_pos: torch.Tensor,  # [B] int32
        out_topk_buffer: torch.Tensor,  # [B, 1, K]
    ) -> torch.Tensor:
        assert x.shape[1] == 1, "decode-only: q_len must be 1"

        bsz = x.size(0)
        ratio = self.compress_ratio
        K = self.index_topk

        self._propagate_pool_to_nested()
        try:
            # Nested compressor decode: reads its state from INDEXER_STATE,
            # writes the new compressed token into the BF16 INDEXER_KV pool.
            self.compressor.forward_decode_vectorized(x, start_pos)

            freqs_per_b = self.freqs_cis[start_pos.long()]
            q = self._compute_indexer_q(qr, freqs_per_b, batched_rope=True)
            # ``softmax_scale * n_heads^-0.5`` is pre-folded into weights_proj.
            weights = F.linear(x, self.weights_proj)

            # Fresh pool read picks up the slot the nested compressor
            # just scattered. Gather full ``T_max = self._kv_cache_t``;
            # mask invalid positions via ``compressed_len`` in TopK.
            T_max = (
                self._kv_block_table.shape[1] * self._kv_eb
                if self._kv_block_table is not None and self._kv_eb > 0
                else self._kv_cache_t
            )
            kv_cache = self._gather_compressed_k(bsz, T_max, x.device)

            # BF16 fused score, no causal mask (decode → q_pos=None).
            score = v4_indexer_score(
                q.contiguous(),
                kv_cache,
                weights,
                q_pos=None,
                compress_ratio=1,
            )

            compressed_len = ((start_pos + 1) // ratio).to(torch.int64).view(bsz, 1, 1)

            # TopK — direct ``persistent_topk`` call mirroring the FP8
            # source's ``dsv4_persistent_topk`` block (only the op name
            # differs; algorithm + signature are identical).
            K_eff = min(K, T_max)
            if (
                K_eff > 0
                and K in (512, 1024, 2048)
                and _persistent_topk_enabled()
            ):
                rtp_llm_ops.persistent_topk(
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
                    k_arange = torch.arange(
                        K, device=out_topk_buffer.device
                    ).view(1, 1, K)
                    out_topk_buffer.masked_fill_(k_arange >= compressed_len, -1)
            return out_topk_buffer
        finally:
            self._clear_nested_pool()
