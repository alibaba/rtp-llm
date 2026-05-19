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
from typing import Any, Optional

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
      cache_seqlens : unused in sparse FP8 decode; accepted to keep the
        attention helper call sites uniform.
      block_table   : unused in sparse FP8 decode; FlashMLA consumes global
        slot ids from ``topk_idxs`` directly.
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

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        sched_meta: Any,
        cache_seqlens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        extra_k_cache: Optional[torch.Tensor] = None,
        extra_topk_idxs: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single- or dual-pool sparse attention.

        ``sched_meta`` is the FlashMLA planner output (``get_mla_metadata``
        return) — owned by :class:`DSv4DecodeAttnMetadataFP8` and fetched via
        :func:`~decode_attn_metadata.get_or_build_sched_meta`. The op is a
        pure dispatcher; it does NOT cache sched_meta itself (was the
        iter2 anti-pattern that accumulated per-instance state across
        decode steps).

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
            sched_meta,
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
        sched_meta: Any,
        cache_seqlens: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        topk_length: Optional[torch.Tensor] = None,
        extra_k_cache: Optional[torch.Tensor] = None,
        extra_topk_idxs: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from flash_mla import flash_mla_with_kvcache  # type: ignore[import-not-found]

        B, q_len, H, D = q.shape
        # FlashMLA expects 4D q ``(batch_size, seq_len_q, num_heads_q, head_dim)``
        # and 3D indices ``(batch_size, seq_len_q, topk)`` per the installed
        # wheel's ``flash_mla_interface.flash_mla_with_kvcache`` docstring.

        assert topk_idxs is not None, "FP8 sparse decode requires topk_idxs"

        # FlashMLA FP8 kernel requires 4D k_cache: [num_blocks, block_size, num_heads_k=1, kv_dim].
        kv_4d = kv_cache.unsqueeze(-2)
        extra_kv_4d = extra_k_cache.unsqueeze(-2) if extra_k_cache is not None else None

        # topk_idxs: [B, q_len, topk] preferred; collapse a stray num_heads_k axis if present.
        if topk_idxs.dim() == 4:
            topk_3d = topk_idxs.squeeze(2).contiguous()
        else:
            topk_3d = topk_idxs.contiguous()

        if extra_topk_idxs is not None:
            extra_topk_3d = (
                extra_topk_idxs.squeeze(2).contiguous()
                if extra_topk_idxs.dim() == 4
                else extra_topk_idxs.contiguous()
            )
        else:
            extra_topk_3d = None

        # Sparse FlashMLA consumes global slot ids from ``indices`` directly.
        # Its sparse branch does not pass block_table/cache_seqlens to the CUDA
        # kernel, so keep dense metadata disabled here.
        block_table = None
        cache_seqlens = None

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
