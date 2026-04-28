"""Phase 4 — V4 decode-arm FP8 sparse attention op.

Wraps FlashMLA's ``flash_mla_with_kvcache(is_fp8_kvcache=True)`` for
the V4 decode arm. The kernel reads the packed ``fp8_model1_mla`` KV
cache directly (no dequant on the read path) and outputs bf16
attention output.

Mirrors the Phase 1 :class:`SparseAttnV4DecodeOp` interface so the
substitution is local to ``Attention.forward_decode_fp8``.

The reference fallback (used on dev boxes without flash_mla / without
CUDA) dequantizes the FP8 KV slots back to bf16 via
:func:`dequantize_v4_kv_slot` and runs the Phase 1 ``_sparse_attn``
Python reference. This produces an output close to but not bit-equal
to the FlashMLA path (~1-3% rel diff from FP8 quant noise); the
SM100_ARM smoke gates the production correctness.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.decode.fp8_kv_quant_decode_op import (
    ENTRY_BYTES,
    NOPE_DIM,
    ROPE_DIM,
    dequantize_v4_kv_slot,
)

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
    logging.warning(
        "[dsv4-fp8] flash_mla not available (%s); FP8 sparse attn falls back "
        "to dequant + Python reference",
        e,
    )


def _dequant_kv_view_to_bf16(
    kv_cache_packed: torch.Tensor,
) -> torch.Tensor:
    """Dequantize a full packed FP8 cache back to bf16 ``[..., 512]``.

    Slow Python loop — only used by the reference fallback path. Production
    runtime hits :func:`flash_mla_with_kvcache` which reads packed FP8
    directly.

    Args:
        kv_cache_packed: ``[num_blocks, block_size, 584]`` uint8.

    Returns:
        ``[num_blocks, block_size, 512]`` bf16.
    """
    num_blocks, block_size, _ = kv_cache_packed.shape
    head_dim = NOPE_DIM + ROPE_DIM
    out = torch.zeros(
        (num_blocks, block_size, head_dim),
        dtype=torch.bfloat16,
        device=kv_cache_packed.device,
    )
    for b in range(num_blocks):
        for s in range(block_size):
            slot_view = kv_cache_packed[b, s]
            if bool((slot_view == 0).all()):
                continue  # skip empty slots
            out[b, s] = dequantize_v4_kv_slot(slot_view)
    return out


class SparseAttnV4DecodeFp8Op:
    """Phase 4 sparse attention decode op for FP8 KV cache.

    Same call signature as :class:`SparseAttnV4DecodeOp` so substitution
    is one-line in ``Attention.forward_decode_fp8``.

    Args (forward):
      q          : ``[B, q_len, n_heads, head_dim]`` bf16
      kv_cache   : ``[num_blocks, block_size, 584]`` uint8 packed FP8
      attn_sink  : ``[n_heads]`` fp32 — per-head learned sink
      topk_idxs  : ``[B, q_len, topk]`` int32 — per-request global slot idxs
      softmax_scale : float
      block_table   : optional ``[B, max_blocks]`` int32 — block table for
        true block-paged layout. If None and ``num_blocks==1``, treat the
        cache as a single contiguous block (Phase 4 default).
      cache_seqlens : optional ``[B]`` int32 — per-request cache length.
        Required by FlashMLA's sparse path.
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
        cache_seqlens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if _FLASH_MLA_AVAILABLE and q.is_cuda and kv_cache.is_cuda:
            return self._forward_flash_mla(
                q,
                kv_cache,
                attn_sink,
                topk_idxs,
                cache_seqlens,
                block_table,
            )
        return self._forward_reference(
            q,
            kv_cache,
            attn_sink,
            topk_idxs,
        )

    def _forward_flash_mla(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        cache_seqlens: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
    ) -> torch.Tensor:
        from flash_mla import (  # type: ignore[import-not-found]
            flash_mla_with_kvcache,
            get_mla_metadata,
        )

        B, q_len, H, D = q.shape
        # Flatten (B, q_len) → [T, H, D] for FlashMLA (per-request layout).
        T = B * q_len
        q_batched = q.reshape(T, H, D)

        # block_table[r, 0] = r: each request r uses FP8 block r (one block per request).
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
        kv_4d = kv_cache.unsqueeze(-2)  # [B, block_size, 1, 584]

        # topk_idxs: [B, q_len, topk] or [B, q_len, 1, topk] → reshape to [T, 1, topk].
        if topk_idxs.dim() == 4:
            topk_flat = topk_idxs.reshape(T, 1, -1)
        else:
            topk_flat = topk_idxs.reshape(T, 1, -1)

        # get_mla_metadata is a no-op stub that returns empty scheduler structures;
        # calling it inline is graph-safe (no stream sync, no alloc).
        topk = topk_flat.shape[-1]
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens=None,
            num_q_tokens_per_head_k=T * H,
            topk=topk,
            num_heads_q=H,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )

        attn_out, _ = flash_mla_with_kvcache(
            q=q_batched,
            k_cache=kv_4d,
            block_table=block_table,
            head_dim_v=self.head_dim,
            cache_seqlens=cache_seqlens,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            is_fp8_kvcache=True,
            indices=topk_flat,
            softmax_scale=self.softmax_scale,
        )
        # V4-Flash attn_sink is zero; non-zero case is unsupported by FlashMLA native path.
        if attn_sink is not None and attn_sink.abs().max().item() > 0:
            logging.warning(
                "[dsv4-fp8] non-zero attn_sink with FlashMLA path: sink correction deferred"
            )

        return attn_out.view(B, q_len, H, self.head_dim).contiguous()

    def _forward_reference(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """CPU / no-flash_mla fallback: dequant KV → BF16, run Phase 1 ref."""
        from rtp_llm.models_py.modules.dsv4.attention import _sparse_attn

        B, q_len, H, D = q.shape
        # Dequantize the packed FP8 cache to a bf16 view.
        # For per-request (single-block) layout, kv_cache is [1, B*T, 584];
        # but our wrapper interface mirrors Phase 1's [B, T, head_dim] view.
        # Reshape: [num_blocks, block_size, 584] → [num_blocks*block_size, 584]
        # → dequant → [num_blocks*block_size, 512] → reshape to [B, T, 512].
        flat_packed = kv_cache.reshape(-1, ENTRY_BYTES)
        T_total = flat_packed.shape[0]
        kv_bf16 = torch.zeros(
            (T_total, NOPE_DIM + ROPE_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        )
        for t in range(T_total):
            slot = flat_packed[t]
            if bool((slot == 0).all()):
                continue
            kv_bf16[t] = dequantize_v4_kv_slot(slot)

        # Reshape to [B, T_per_req, head_dim] — assumes single-block layout
        # with block_size = T_per_req per request (Phase 4 default).
        T_per_req = T_total // B
        kv_view = kv_bf16.view(B, T_per_req, self.head_dim)

        return _sparse_attn(q, kv_view, attn_sink, topk_idxs, self.softmax_scale)
