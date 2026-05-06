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
from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4._indexer_cp_gather_triton import (
    gather_indexer_k_for_prefill,
)
from rtp_llm.models_py.modules.dsv4._indexer_fp8_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
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
from rtp_llm.models_py.modules.dsv4.compressor_fp8 import CompressorFP8
from rtp_llm.models_py.modules.dsv4.cp import CPContext, cp_freqs_cis_local
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
        weights: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
    ):
        super().__init__()
        assert index_head_dim == INDEXER_HEAD_DIM, (
            f"IndexerFP8 locked to index_head_dim={INDEXER_HEAD_DIM} "
            f"(matches CompressorFP8 132B layout); got {index_head_dim}"
        )
        assert has_fp8_paged_mqa_logits(), (
            "deep_gemm.fp8_paged_mqa_logits not available — IndexerFP8 cannot "
            "operate without DeepGEMM. Use IndexerBF16 (or install deep_gemm)."
        )
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
                weights[f"{prefix}.weights_proj.weight"], requires_grad=False
            )
        else:
            self.wq_b = QuantizedLinear(
                q_lora_rank, index_n_heads * index_head_dim, storage="fp8"
            )
            self.weights_proj = nn.Linear(dim, index_n_heads, bias=False)

        # Nested compressor: 132B layout (head_dim=128).
        self.compressor = CompressorFP8(
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

    def set_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        self._cp_ctx = cp_ctx

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
        if self._factory_mode and qr.dim() > 2:
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
            weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)

            compressed_len = ((start_pos + 1) // ratio).to(torch.int64).view(bsz, 1, 1)

            # Always use DeepGEMM (FP8 path).  Eager decode trims the
            # scored context length from the live max position. During CUDA
            # graph capture that D2H scalar read is illegal, so capture with
            # a static upper bound; replay updates block tables and lengths
            # in-place before the captured kernels run.
            T_cache = self._kv_block_table.shape[1] * self._kv_eb
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
    # Prefill (bsz==1; FIFO scheduler)
    # --------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos,
        offset: int,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        cp_ctx = self._cp_ctx
        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1 and start_pos == 0

        if cp_on:
            freqs_cis = cp_freqs_cis_local(self.freqs_cis, cp_ctx)
            end_pos = cp_ctx.seq_len_full
            sp = 0
        else:
            sp = int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
            freqs_cis = self.freqs_cis[sp : sp + seqlen]
            end_pos = sp + seqlen

        is_fresh_prefill = sp == 0

        # Warmup forward (no pool bound by framework): emit empty topk
        # (matches BF16 Indexer's warmup fallback shape). Caller in
        # ``Attention._forward_prefill_fp8_csa_hca`` concats with the SWA
        # topk; an empty trailing dim is a no-op there.
        if (
            self._kv_block_table is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return torch.full((bsz, seqlen, 0), -1, dtype=torch.int64, device=x.device)

        if self.compressor.freqs_cis is None:
            self.compressor.freqs_cis = self.freqs_cis

        self._propagate_pool_to_nested()
        try:
            q = self._compute_indexer_q(qr, freqs_cis)
            self.compressor(x, start_pos)
            weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)

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

            # FP8 prefill score via DeepGEMM ``fp8_mqa_logits`` — gather
            # k_quant + k_scale from the 132B pool (per-token), then fused
            # einsum + ReLU + per-head weighted sum.
            assert (
                has_fp8_mqa_logits()
            ), "deep_gemm.fp8_mqa_logits required for IndexerFP8 prefill"
            assert (
                self._kv_block_table is not None
            ), "IndexerFP8 prefill requires bound KV pool + block_table"
            assert not cp_on, (
                "IndexerFP8 + CP path not yet supported "
                "(positions split across ranks; needs gather first)"
            )

            if T == 0:
                # Cold-start prefill before any compressed tokens — no K
                # to score against; emit empty topk buffer behavior.
                topk_idxs = torch.full(
                    (bsz, S, 0), -1, dtype=torch.int64, device=q.device
                )
            else:
                valid, safe_slot = self._compute_pool_slots(
                    bsz,
                    T,
                    self._kv_block_table,
                    self._kv_eb,
                    x.device,
                    pool_rows=int(
                        self._kv_pool_view.numel() // self._kv_pool_view.shape[-1]
                    ),
                )
                slot_mapping = torch.where(
                    valid, safe_slot, torch.full_like(safe_slot, -1)
                ).reshape(
                    -1
                )  # [bsz * T]

                # Normalize to flat 2D first (production hands us 3D
                # ``[num_blocks, eb, 132]``; tests pass flat 2D), then
                # reshape to the ``[total_slots, 1, 132]`` virtual-block
                # layout the gather kernel expects (slot indices are
                # already absolute into the flat pool).
                _flat_pool = (
                    self._kv_pool_view.flatten(0, 1)
                    if self._kv_pool_view.dim() == 3
                    else self._kv_pool_view
                )
                pool_blocks = _flat_pool.view(-1, 1, INDEXER_ENTRY_BYTES)
                k_quant_flat, k_scale_flat = gather_indexer_k_for_prefill(
                    pool_blocks,
                    slot_mapping,
                    head_dim=INDEXER_HEAD_DIM,
                )  # [bsz*T, D], [bsz*T]

                q_fp8, w_fold = indexer_q_fp8_quant_fold(
                    _as_bf16_contig(q), _as_bf16_contig(weights)
                )

                M = bsz * S
                if q_pos_kernel is not None:
                    ke = ((q_pos_kernel.to(torch.int32) + 1) // ratio).clamp_max(T)
                    cu_ke = ke.reshape(M).contiguous()
                else:
                    cu_ke = torch.full((M,), T, dtype=torch.int32, device=x.device)
                cu_ks = torch.zeros(M, dtype=torch.int32, device=x.device)

                assert bsz == 1, (
                    "IndexerFP8 prefill assumes bsz==1 (FIFO scheduler "
                    "max_context_batch_size=1); got bsz={bsz}"
                ).format(bsz=bsz)
                k_quant = k_quant_flat
                k_scale = k_scale_flat

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
                K = self.index_topk
                # Prefer persistent radix-select TopK (same gate as decode).
                # Per-row valid length comes from cu_ke (= (q_pos+1)//ratio
                # clamped to T); the kernel writes -1 past that boundary.
                use_persistent = (
                    K in (512, 1024, 2048) and _persistent_topk_enabled() and T >= K
                )
                if use_persistent:
                    out_buf = torch.empty(
                        (M, K), dtype=torch.int32, device=index_score.device
                    )
                    rtp_llm_ops.dsv4_persistent_topk(
                        index_score.view(M, T).contiguous(),
                        cu_ke,
                        out_buf,
                        _get_topk_workspace(index_score.device),
                        K,
                        T,
                    )
                    topk_idxs = out_buf.view(bsz, S, K).long()
                else:
                    topk_idxs = index_score.topk(min(K, T), dim=-1)[1]

            # Sentinel-aware post-processing: persistent_topk emits -1 past
            # the per-row valid length; ``+offset`` would otherwise corrupt
            # those into ``offset - 1``. Fold the q-position causal mask in.
            sentinel = topk_idxs < 0
            if is_fresh_prefill:
                if cp_on:
                    q_pos_1b = (cp_ctx.global_positions + 1).unsqueeze(1)
                else:
                    q_pos_1b = torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1)
                mask = sentinel | (topk_idxs >= (q_pos_1b // ratio))
                topk_idxs = torch.where(mask, -1, topk_idxs + offset)
            else:
                topk_idxs = torch.where(sentinel, -1, topk_idxs + offset)
            return topk_idxs
        finally:
            self._clear_nested_pool()
