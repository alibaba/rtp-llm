"""V4 decode-arm FP8 sparse attention op.

Wraps FlashMLA's ``flash_mla_with_kvcache(is_fp8_kvcache=True)`` for
single- and dual-pool decode. The kernel reads the packed
``fp8_model1_mla`` KV cache directly (no dequant on the read path) and
outputs bf16 attention output.

Dual-pool support uses FlashMLA's ``extra_k_cache`` +
``extra_indices_in_kvcache`` parameters to attend over a second FP8 pool
(CSA / HCA compressor pool) in a single kernel call, with in-kernel
softmax merging across both pools (mirrors vLLM
``deepseek_v4_attention.py:849-865``). Replaces the legacy "dequant both
pools -> BF16 cat -> TileLang sparse_attn" path which was
bandwidth-bound on the dequant kernels.

FlashMLA wheel is required (CUDA >= 12.9). The op asserts wheel
availability at forward — there is no slow Python reference fallback
because all dev/CI/prod boxes carry flash_mla.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

_FLASH_MLA_AVAILABLE = False
try:
    if torch.version.cuda:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        if (major, minor) >= (12, 9):
            from flash_mla import (
                flash_mla_with_kvcache,  # type: ignore[import-not-found]
            )
            from flash_mla import get_mla_metadata  # type: ignore[import-not-found]

            _FLASH_MLA_AVAILABLE = True
except (ImportError, AttributeError, ValueError) as e:
    logging.warning("[dsv4-fp8] flash_mla wheel unavailable (%s)", e)


class SparseAttnV4DecodeFp8Op:
    """FP8 sparse attention decode op (single- or dual-pool).

    Args (forward):
      q          : ``[B, q_len, n_heads, head_dim]`` bf16
      kv_cache   : ``[num_blocks, block_size, 584]`` uint8 packed FP8
        primary pool (SWA in dual-pool mode).
      attn_sink  : ``[n_heads]`` fp32 — per-head learned sink
      topk_idxs  : ``[B, q_len, topk]`` int32 — per-request global slot
        ids into the primary pool.
      cache_seqlens : optional ``[B]`` int32 — per-request cache length.
      block_table   : optional ``[B, max_blocks]`` int32 — primary pool
        block table.
      topk_length        : optional ``[B]`` int32 — per-request leftmost
        valid length on ``topk_idxs``.
      extra_k_cache      : optional secondary FP8 pool (CMP). Triggers
        FlashMLA's dual-pool path.
      extra_topk_idxs    : optional ``[B, q_len, extra_topk]`` int32 —
        global slot ids into ``extra_k_cache``.
      extra_topk_length  : optional ``[B]`` int32 — per-request leftmost
        valid length on ``extra_topk_idxs``.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        softmax_scale: float,
    ) -> None:
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale
        # Iter2: cached FlashMLASchedMeta keyed by the dual-pool flag (single
        # vs dual). The wheel populates ``sched_meta.config`` and the kernel-
        # side ``tile_scheduler_metadata`` / ``num_splits`` tensors on the
        # FIRST call where ``have_initialized=False``; subsequent calls with
        # the same shape skip the planner. Decode at fixed B/q_len/H/topk
        # across all 60 layers means one planner setup per layer-type per
        # process lifetime instead of 60 setups per step. Mirrors vLLM's
        # ``swa_metadata.tile_sched_{swaonly,c4a,c128a}`` cache.
        #
        # Keys: 0 = single-pool (no extra_k_cache); 1 = dual-pool. Different
        # extra_topk values would in principle need separate metas, but
        # within one layer-type the dual-pool topk is constant across calls.
        self._sched_meta_cache: dict = {}

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        cache_seqlens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        extra_k_cache: Optional[torch.Tensor] = None,
        extra_topk_idxs: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single- or dual-pool sparse attention.

        Dual-pool: pass ``extra_k_cache`` (e.g. CMP pool 3D
        ``[num_blocks, block_size, 584]`` uint8) + ``extra_topk_idxs``
        (3D ``[B, q_len, extra_topk]`` int32 global slot ids) to attend
        over a second FP8 KV pool in a single FlashMLA invocation. The
        kernel merges softmax across both pools natively.
        """
        assert _FLASH_MLA_AVAILABLE, (
            "flash_mla wheel is required for FP8 sparse decode "
            "(install rtp_llm with cuda12_9 / cuda13 config)"
        )
        assert q.is_cuda and kv_cache.is_cuda, "FP8 sparse decode requires CUDA tensors"
        return self._forward_flash_mla(
            q,
            kv_cache,
            attn_sink,
            topk_idxs,
            cache_seqlens,
            block_table,
            topk_length,
            extra_k_cache,
            extra_topk_idxs,
            extra_topk_length,
        )

    def _forward_flash_mla(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        cache_seqlens: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        topk_length: Optional[torch.Tensor] = None,
        extra_k_cache: Optional[torch.Tensor] = None,
        extra_topk_idxs: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from flash_mla import (  # type: ignore[import-not-found]
            flash_mla_with_kvcache,
            get_mla_metadata,
        )

        B, q_len, H, D = q.shape
        # FlashMLA expects 4D q ``(batch_size, seq_len_q, num_heads_q, head_dim)``
        # and 3D indices ``(batch_size, seq_len_q, topk)`` per the installed
        # wheel's ``flash_mla_interface.flash_mla_with_kvcache`` docstring.

        # block_table[r, 0] = r: each request r uses FP8 block r.
        if block_table is None:
            block_table = torch.arange(B, dtype=torch.int32, device=q.device).unsqueeze(
                1
            )

        if cache_seqlens is None:
            cache_seqlens = torch.full(
                (B,),
                kv_cache.shape[1],
                dtype=torch.int32,
                device=q.device,
            )

        # FlashMLA FP8 kernel requires 4D k_cache: [num_blocks, block_size, num_heads_k=1, kv_dim].
        kv_4d = kv_cache.unsqueeze(-2)
        extra_kv_4d = extra_k_cache.unsqueeze(-2) if extra_k_cache is not None else None

        # topk_idxs: [B, q_len, topk] preferred; collapse a stray num_heads_k axis if present.
        if topk_idxs.dim() == 4:
            topk_3d = topk_idxs.squeeze(2).contiguous()
        else:
            topk_3d = topk_idxs.contiguous()
        topk = topk_3d.shape[-1]

        if extra_topk_idxs is not None:
            extra_topk_3d = (
                extra_topk_idxs.squeeze(2).contiguous()
                if extra_topk_idxs.dim() == 4
                else extra_topk_idxs.contiguous()
            )
        else:
            extra_topk_3d = None

        # Iter2 (revised iter3.0a): cache + reuse FlashMLASchedMeta across
        # decode layers of the same shape bucket. Cache key must include
        # batch size (FlashMLA bakes ``config.b`` into sched_meta — reusing
        # across different B triggers the wheel's ``sched_meta.config.b ==
        # batch_size`` assert, which CUDA graph capture surfaces because it
        # enumerates BS ∈ {capture sizes}). q_len is always 1 in decode and
        # H/topk are per-op-instance constants, so (B, single/dual) is the
        # minimal unique key; including q_len/H/topk for safety.
        cache_key = (B, q_len, H, topk, 1 if extra_k_cache is not None else 0)
        sched_meta = self._sched_meta_cache.get(cache_key)
        if sched_meta is None:
            sched_meta, _ = get_mla_metadata(
                cache_seqlens=None,
                num_q_tokens_per_head_k=B * q_len * H,
                topk=topk,
                num_heads_q=H,
                num_heads_k=1,
                is_fp8_kvcache=True,
            )
            self._sched_meta_cache[cache_key] = sched_meta

        # DSv4 attn_sink is per-head fp32, loaded from ckpt (layers.*.attn.attn_sink
        # shape [n_heads], non-zero ~0.3..0.6 mean). FlashMLA kernel applies
        # output *= exp(lse) / (exp(lse) + exp(attn_sink)). Mirrors vLLM
        # ``deepseek_v4_attention.py:860`` (both single- and dual-pool calls).
        attn_out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_4d,
            block_table=block_table,
            head_dim_v=self.head_dim,
            cache_seqlens=cache_seqlens,
            tile_scheduler_metadata=sched_meta,
            num_splits=None,
            is_fp8_kvcache=True,
            indices=topk_3d,
            softmax_scale=self.softmax_scale,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_k_cache=extra_kv_4d,
            extra_indices_in_kvcache=extra_topk_3d,
            extra_topk_length=extra_topk_length,
        )

        return attn_out.view(B, q_len, H, self.head_dim).contiguous()
