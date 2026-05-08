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
from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import PoolBackedModule
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


# Persistent radix-select TopK — vendored CUDA kernel binding. Same gate
# as the BF16 class so a single env knob disables both paths.
_PERSISTENT_TOPK_OK = hasattr(rtp_llm_ops, "dsv4_persistent_topk")
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


class _IndexerFP8PrefillMeta(NamedTuple):
    """Per-call FP8 prefill metadata for :class:`IndexerFP8`.

    Built once by :meth:`IndexerFP8.prepare` from the caller's
    per-step state (``bsz``, ``seqlen``, ``sp_int``, ``device``) and the
    bound INDEXER_KV pool. Replaces the in-forward ``torch.arange`` /
    ``_compute_pool_slots`` / ``where`` / ``zeros`` chain so
    :meth:`IndexerFP8.forward` is kernel-only.

    All ``torch.Tensor`` fields are device-side, contiguous, and
    immediately consumable by the gather / score / topk kernels.

    Coordinate system: ``ks`` / ``ke`` index the compressed K pool
    starting at 0 — no global offset baked in. The caller is responsible
    for any downstream coordinate transforms (``+offset`` / dtype cast
    for cat-with-SWA / etc.).
    """

    # ── geometry (Python scalars; cheap) ──
    bsz: int
    seqlen: int
    M: int  # = bsz * seqlen  (= T_total for flat 2D)
    sp_int: int  # absolute prefix length
    end_pos: int  # = sp_int + seqlen
    is_fresh_prefill: bool  # = (sp_int == 0)
    T: int  # compressed K count = end_pos // ratio

    # ── Q-side ──
    freqs_cis_slice: torch.Tensor  # self.freqs_cis[sp:sp+seqlen]; pre-sliced view
    positions_d: torch.Tensor  # [M] int32 — global positions sp..sp+S-1

    # ── per-row visible-K window for fp8_mqa_logits ──
    # ks[r] = 0; ke[r] = clamp((positions_d[r]+1)//ratio, max=T)
    ks: torch.Tensor  # [M] int32
    ke: torch.Tensor  # [M] int32

    # ── K-side (paged) ──
    block_table_i32: torch.Tensor  # [B, max_blocks] int32 contig
    cu_kv_seqlens: torch.Tensor  # [B+1] int32 (compressed)


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

        from rtp_llm.models_py.modules.dsv4.attention import _v4_fp8_linear
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

    # --------------------------------------------------------------
    # Pool propagation to nested compressor
    # --------------------------------------------------------------
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
        # IndexerFP8 does not support Context-Parallel. transformer's
        # global ``_propagate_cp_ctx`` calls this on every indexer; accept
        # ``None`` (CP off) silently and refuse a real ctx loudly.
        assert cp_ctx is None, "IndexerFP8 does not support Context-Parallel"

    # --------------------------------------------------------------
    # Q-projection + RoPE helper (shared between prefill & decode)
    # --------------------------------------------------------------
    def _compute_indexer_q(
        self,
        qr: torch.Tensor,
        freqs_cis: torch.Tensor,
        batched_rope: bool = False,
    ) -> torch.Tensor:
        """qr → wq_b → unflatten → RoPE → q [B, S, H, D]."""
        # Framework FP8 linear (CudaFp8DeepGEMMLinear) requires 2D input;
        # flatten leading dims and restore.
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

    # --------------------------------------------------------------
    # Decode (vectorized over B)
    # --------------------------------------------------------------
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
            # Nested compressor writes its compressed token to the 132B pool.
            self.compressor.forward_decode_vectorized(x, start_pos)

            freqs_per_b = self.freqs_cis[start_pos.long()]
            q = self._compute_indexer_q(qr, freqs_per_b, batched_rope=True)
            # ``softmax_scale * n_heads^-0.5`` is pre-folded into weights_proj at __init__.
            weights = F.linear(x, self.weights_proj)

            compressed_len = ((start_pos + 1) // ratio).to(torch.int64).view(bsz, 1, 1)

            # Always use DeepGEMM (FP8 path).  Eager decode trims the
            # scored context length from the live max position. During CUDA
            # graph capture that D2H scalar read is illegal, so capture with
            # a static upper bound; replay updates block tables and lengths
            # in-place before the captured kernels run.
            T_cache = self._kv_block_table.shape[1] * self._kv_eb
            # Capture-aware: under cuda graph capture we can't D2H sync via
            # ``.item()`` (operation-not-permitted on captured stream), so
            # fall back to the static upper bound. Eager keeps the dynamic
            # clamp to avoid overshoot.
            if q.is_cuda and torch.cuda.is_current_stream_capturing():
                T_static = self._kv_cache_t if self._kv_cache_t > 0 else T_cache
                T_max = max(32, min(T_cache, T_static))
            else:
                sp_max = int(start_pos.max().item())
                T_live = (((sp_max + 1) // ratio) + 31) & ~31
                T_max = max(32, min(T_cache, T_live))

            q_fp8, w_fold = indexer_q_fp8_quant_fold(
                _as_bf16_contig(q), _as_bf16_contig(weights)
            )
            ctx_lens_2d = compressed_len.view(bsz, 1).to(torch.int32)
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
                w_fold.view(bsz * 1, self.n_heads),
                pool_2d,
                bt_i32,
                ctx_lens_2d,
                block_size=self._kv_eb,
                max_ctx_len=T_max,
            )  # [B, T_max] fp32
            score = logits.view(bsz, 1, T_max)

            # TopK (with optional persistent radix-select)
            K_eff = min(K, T_max)
            if K_eff > 0 and K in (512, 1024, 2048) and _persistent_topk_enabled():
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
                    out_topk_buffer.masked_fill_(k_arange >= compressed_len, -1)
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
        # Phase-1 varlen plumbing — defaults preserve the B==1 + scalar
        # ``(seqlen, sp_int)`` path bit-for-bit. Phase-3 work converts the
        # body to consume these so:
        #   * ``positions_d``    = position_ids (per-token global)
        #   * ``ke``             = ``((position_ids+1)//ratio).clamp_max(T_b)``
        #     where T_b is per-request compressed-K count; needs cu_seqlens
        #     and prefix_lengths to compute T_b and clamp per-row.
        #   * ``ks``             = ``(prefix_lengths[req] // ratio)`` per
        #     row when continuation prefill stops attending pre-prefix
        #     compressed K (default still 0 for cold-start parity).
        #   * ``cu_kv_seqlens``  = per-request cumsum of T_b across [B+1].
        #   * ``block_table_i32``= ``kv_block_table[:batch_size]``.
        # Note: ``bsz`` and ``batch_size`` are kept separate during the
        # transition — ``bsz`` is the legacy positional, ``batch_size``
        # the new canonical (== B from upper layer); enforce equality
        # once Engineer B's Phase-3 lands.
        batch_size: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
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
                kv_block_table[:bsz].to(device=device, dtype=torch.int32).contiguous()
            )
        else:
            block_table_i32 = torch.empty((bsz, 0), dtype=torch.int32, device=device)
        cu_kv_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

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
            q = self._compute_indexer_q(qr, attention_inputs.freqs_cis_slice)
            self.compressor(x, sp)
            # ``softmax_scale * n_heads^-0.5`` is pre-folded into weights_proj at __init__.
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
            k_quant_flat = torch.empty(
                (T, INDEXER_HEAD_DIM),
                dtype=torch.float8_e4m3fn,
                device=x.device,
            )
            # quant_block_size = head_dim*4/dst_scale.size(1) = 128*4/4 = 128.
            k_scale_buf = torch.empty((T, 4), dtype=torch.uint8, device=x.device)
            rtp_llm_ops.cp_gather_indexer_k_quant_cache(
                self._kv_pool_view,
                k_quant_flat,
                k_scale_buf,
                attention_inputs.block_table_i32,
                attention_inputs.cu_kv_seqlens,
            )
            # ``deep_gemm.fp8_mqa_logits`` expects k_scale as 1D fp32
            # contig — view uint8 [T, 4] → fp32 [T, 1] → squeeze [T].
            k_scale_flat = k_scale_buf.view(torch.float32).squeeze(-1)

            q_fp8, w_fold = indexer_q_fp8_quant_fold(
                _as_bf16_contig(q), _as_bf16_contig(weights)
            )

            logits = fp8_mqa_indexer_score(
                q_fp8.view(M, self.n_heads, INDEXER_HEAD_DIM),
                w_fold.view(M, self.n_heads),
                k_quant_flat,
                k_scale_flat,
                attention_inputs.ks,
                attention_inputs.ke,
                clean_logits=False,
            )  # [M, T] fp32

            # Vendored CUDA per-row TopK over [ks[r], ke[r]). Causal
            # mask is implicit via ke = (q_pos+1)//ratio clamped to T;
            # padding past per-row valid count is ``-1`` from the kernel.
            out_buf = torch.empty((M, K), dtype=torch.int32, device=logits.device)
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
