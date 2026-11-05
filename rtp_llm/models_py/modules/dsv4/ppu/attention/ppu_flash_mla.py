"""Deterministic MODEL1 sparse Decode scheduling for the PPU FlashMLA ABI."""

import torch
from rtp_llm.models_py.kernels.ppu.attention.ppu_sparse_attention import (
    MODEL1_NATIVE_PAGES,
    _schedule,
)

from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class PpuSparseAttnV4DecodeFp8Op(SparseAttnV4DecodeFp8Op):
    def _forward_flash_mla(
        self, q, kv_cache, attn_sink, topk_idxs, sched_meta,
        cache_seqlens=None, block_table=None, topk_length=None,
        extra_k_cache=None, extra_topk_idxs=None, extra_topk_length=None,
    ):
        if (
            q.shape[-2:] == (64, 512)
            and self.softmax_scale == 512 ** -0.5
            and kv_cache.shape[1] in MODEL1_NATIVE_PAGES
            and (extra_k_cache is None or extra_k_cache.shape[1] in MODEL1_NATIVE_PAGES)
        ):
            from rtp_llm.models_py.kernels.ppu.attention.ppu_sparse_attention import forward

            return forward(
                q, kv_cache, attn_sink, topk_idxs, sched_meta,
                cache_seqlens, block_table, topk_length,
                extra_k_cache, extra_topk_idxs, extra_topk_length,
            )
        if not sched_meta.have_initialized:
            batch, queries, heads, _ = q.shape
            width = topk_idxs.shape[-1]
            extra_width = extra_topk_idxs.shape[-1] if extra_topk_idxs is not None else 0
            sched_meta.config = type(sched_meta).Config(
                batch, queries, heads, kv_cache.shape[1], 1,
                False, True, width,
                extra_k_cache.shape[1] if extra_k_cache is not None else None,
                extra_width if extra_k_cache is not None else None,
            )
            sched_meta.tile_scheduler_metadata = torch.empty(
                (batch, 8), dtype=torch.int32, device=q.device
            )
            sched_meta.num_splits = torch.empty(
                batch + 1, dtype=torch.int32, device=q.device
            )
            # Split-request combine can vary by a BF16 ulp in this wheel;
            # subsequent FP8 projection can amplify that into a token change.
            # The cache is rebuilt on graph-capture entry, capturing this
            # launch so dynamic lengths refresh on every graph replay.
            _schedule[(batch,)](
                sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
                topk_length, extra_topk_length, batch, width, extra_width,
                topk_length is not None, extra_topk_length is not None,
            )
            sched_meta.have_initialized = True
        return super()._forward_flash_mla(
            q, kv_cache, attn_sink, topk_idxs, sched_meta,
            cache_seqlens, block_table, topk_length,
            extra_k_cache, extra_topk_idxs, extra_topk_length,
        )
