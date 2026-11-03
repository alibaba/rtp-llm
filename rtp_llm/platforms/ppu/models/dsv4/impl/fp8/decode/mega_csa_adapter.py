"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_adapter import (
    MegaCSAPoolContext,
    __all__,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional
import torch
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_runtime import (
    MegaCSARuntime,
    MegaCSASlotMappings,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
    COMPRESS_RATIO,
    GEOMETRY_BY_DIM,
    HC,
    HEAD_DIM,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    MAX_BATCH,
    O_LORA_RANK,
    ROPE_DIM,
    CSAGeometry,
    MegaCSAWeights,
)

if TYPE_CHECKING:
    from rtp_llm.platforms.ppu.models.dsv4.impl.block import Block

from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_adapter import MegaCSAAdapter as _BaseMegaCSAAdapter

class MegaCSAAdapter(_BaseMegaCSAAdapter):
    def _forward_bound(
        self,
        block: "Block",
        hidden: torch.Tensor,
        metadata: Any,
        pools: MegaCSAPoolContext,
        dsv4_mega: Any,
    ) -> torch.Tensor:
        g = self._geometry
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.decode.decode_attn_metadata import get_or_build_sched_meta
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.indexer import _get_topk_workspace
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import CSA_KV, SWA_KV
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        attn = block.attn
        batch_size, q_len = int(hidden.shape[0]), int(hidden.shape[1])
        token_count = batch_size * q_len
        positions_i32 = metadata.position_ids[:token_count]
        positions_i64 = metadata.position_ids_long[:token_count]
        if positions_i32.dtype != torch.int32 or positions_i64.dtype != torch.int64:
            raise TypeError("DSV4 mega positions must provide int32 and int64 tensors")
        rope_cos, rope_sin = self.runtime.rope_tables(attn.freqs_cis)
        num_split = self.runtime.num_hc_splits(token_count, hidden.device, g.dim)
        workspace = self.runtime.layer_workspace(
            token_count, num_split, hidden.device, g
        )
        hidden_rows = hidden.view(token_count, HC, g.dim)

        compressed_lengths_2d = metadata.compressed_lens_per_token.get(COMPRESS_RATIO)
        if (
            compressed_lengths_2d is None
            or compressed_lengths_2d.dim() != 2
            or int(compressed_lengths_2d.shape[0]) < batch_size
            or int(compressed_lengths_2d.shape[1]) < q_len
        ):
            raise RuntimeError(
                "DSV4 mega metadata is missing per-token CSA compressed lengths"
            )
        compressed_lengths_2d = compressed_lengths_2d[:batch_size, :q_len].contiguous()
        compressed_lengths = compressed_lengths_2d.view(token_count)
        if (
            compressed_lengths.dtype != torch.int32
            or not compressed_lengths.is_cuda
            or not compressed_lengths.is_contiguous()
        ):
            raise TypeError(
                "DSV4 mega compressed lengths must be contiguous CUDA int32"
            )

        topk = int(attn.indexer.index_topk)
        topk_buffer = metadata.topk_buffer_compressed
        if (
            topk_buffer.dim() != 3
            or int(topk_buffer.shape[0]) < batch_size
            or int(topk_buffer.shape[1]) < q_len
            or int(topk_buffer.shape[2]) < topk
            or topk_buffer.dtype != torch.int32
            or not topk_buffer.is_cuda
            or not topk_buffer.is_contiguous()
        ):
            raise TypeError("DSV4 mega TopK output must be contiguous CUDA int32")
        topk_output = topk_buffer[:batch_size, :q_len, :topk].reshape(token_count, topk)

        for attn_type, name in ((SWA_KV, "SWA_KV"), (CSA_KV, "CSA_KV")):
            pool = attn._pool_view_3d_fp8(attn_type)
            block_table = metadata.pool_block_tables.get(attn_type)
            if pool is None or block_table is None or block_table.numel() == 0:
                raise RuntimeError(f"DSV4 mega FlashMLA input {name} is unavailable")
        if (
            metadata.swa_abs_idx is None
            or metadata.req_id_per_token is None
            or metadata.swa_global_slots is None
            or CSA_KV not in metadata.paged_pool_tokens_per_block
        ):
            raise RuntimeError("DSV4 mega metadata is incomplete for native FlashMLA")
        attn._get_fp8_decode_op()
        get_or_build_sched_meta(
            metadata,
            batch_size=batch_size,
            q_len=q_len,
            num_heads=attn.n_heads,
            topk=attn.window_size,
            extra_attn_type=CSA_KV,
            extra_index_width=topk,
        )

        score_capacity = int(pools.indexer_block_table.shape[1]) * pools.indexer_entries
        logits = self.runtime.logits(token_count, score_capacity, hidden.device)
        schedule = self.runtime.mqa_schedule(
            compressed_lengths_2d, pools.indexer_entries
        )
        topk_workspace = _get_topk_workspace(hidden.device)

        tf32_hc_prenorm_gemm(
            hidden_rows.view(token_count, HC * g.dim),
            self.weights.hc_fn,
            workspace.hc_partial,
            workspace.hc_sum_sq,
            num_split,
        )
        # The lite reduction writes mix; front_mixed_gemm_csa's fused HC tail
        # produces post/comb before attn_hc.post consumes them.
        dsv4_mega.hc_reduce_fuse_out(
            hidden_rows,
            workspace.hc_partial,
            workspace.hc_sum_sq,
            self.weights.hc_base,
            self.weights.hc_scale,
            block.attn_hc.hc_eps,
            block.attn_hc.norm_eps,
            workspace.collapsed,
            workspace.pre,
            workspace.post,
            workspace.comb,
            with_post_comb=False,
            attn_norm_w=self.weights.attn_norm,
            attn_norm_eps=block.attn_hc.norm_eps,
            mix_out=workspace.mix,
            xq_out=workspace.hidden_fp8.view(torch.uint8),
            xsf_out=workspace.hidden_sf,
            pdl=False,
        )
        dsv4_mega.front_mixed_gemm_csa(
            workspace.collapsed,
            workspace.hidden_fp8,
            workspace.hidden_sf,
            self.weights.front_bf16,
            self.weights.front_fp8,
            self.weights.front_sf,
            workspace.mix,
            self.weights.hc_base,
            self.weights.hc_scale,
            workspace.post,
            workspace.comb.view(token_count, HC * HC),
            out=workspace.front_out,
            hc_eps=block.attn_hc.hc_eps,
            main_state=pools.main_state,
            main_ape=self.weights.main_ape,
            main_state_row=pools.slots.main_state_rows,
            ape_phase=positions_i32,
            idx_state=pools.indexer_state,
            idx_state_row=pools.slots.indexer_state_rows,
            idx_ape=self.weights.indexer_ape,
            win_y2=workspace.window_y,
            w64=workspace.indexer_weights,
            pdl=True,
        )
        wq_outputs = dsv4_mega.wq_b_proj_gemm_merged_csa(
            workspace.q_lora_fp8,
            workspace.q_lora_sf,
            self.weights.wq_b_fp8,
            self.weights.wq_b_sf,
            positions_i32,
            rope_cos,
            rope_sin,
            mock_post=False,
            cmp_pos=positions_i64,
            idx_norm=self.weights.indexer_norm,
            cos_tab=rope_cos,
            sin_tab=rope_sin,
            idx_state=pools.indexer_state,
            idx_state_row=pools.slots.indexer_state_rows,
            state_ring_entries=pools.indexer_state_entries,
            idx_state_block_table=pools.indexer_state_block_table,
            idx_token_to_req=metadata.req_id_per_token[:token_count],
            idx_state_tokens_per_block=pools.indexer_state_tokens_per_block,
            win_y2=workspace.window_y,
            win_norm=self.weights.window_norm,
            q_y=workspace.front_out[:, : g.q_lora_rank],
            q_norm_w=self.weights.q_norm,
            q_eps=attn.eps,
            indexer_fp8=True,
            iq_weights=workspace.indexer_weights,
            idx_cache=pools.indexer_cache,
            idx_dst=pools.slots.indexer_destinations,
            idx_entries_per_block=pools.indexer_entries,
            idx_block_stride_bytes=pools.indexer_stride_bytes,
            swa_cache=pools.swa_cache,
            swa_dst=pools.slots.swa_destinations,
            swa_entries_per_block=pools.swa_entries,
            swa_block_stride_bytes=pools.swa_stride_bytes,
            iq_dst=workspace.indexer_q,
            iq_dst_sf=workspace.indexer_folded_weights,
            pdl=True,
        )
        q_raw, indexer_q, folded_weights = wq_outputs[:3]

        dsv4_mega.mqa_logits_fp8_decode_out(
            indexer_q,
            pools.indexer_cache,
            folded_weights,
            compressed_lengths_2d,
            pools.indexer_block_table[:batch_size],
            schedule,
            logits,
            kv_entries_per_block=pools.indexer_entries,
            kv_block_stride_bytes=pools.indexer_stride_bytes,
            cmp_pos=positions_i64,
            comp_norm=self.weights.main_compressor_norm,
            cos_tab=rope_cos,
            sin_tab=rope_sin,
            comp_state=pools.main_state,
            comp_state_row=pools.slots.main_state_rows,
            comp_state_ring_entries=pools.main_state_entries,
            comp_state_block_table=pools.main_state_block_table,
            comp_token_to_req=metadata.req_id_per_token[:token_count],
            comp_state_tokens_per_block=pools.main_state_tokens_per_block,
            cmp_cache=pools.main_cache,
            cmp_dst=pools.slots.main_destinations,
            cmp_entries_per_block=pools.main_entries,
            cmp_block_stride_bytes=pools.main_stride_bytes,
            query_x=q_raw,
            query_positions=positions_i64,
            query_cos=rope_cos,
            query_sin=rope_sin,
            query_out=q_raw,
            query_eps=attn.eps,
            pdl=True,
        )
        # The fused MQA tail normalizes and rotates q_raw in place.
        q_ready = q_raw.view(batch_size, q_len, g.main_heads, HEAD_DIM)

        rtp_llm_ops.dsv4_persistent_topk(
            logits,
            compressed_lengths,
            topk_output,
            topk_workspace,
            topk,
            int(logits.shape[1]),
        )
        attention = attn._forward_decode_compressed(
            q_ready,
            topk_output.view(batch_size, q_len, topk),
            batch_size,
            q_len,
            metadata,
            cmp_attn_type=CSA_KV,
        )
        dsv4_mega.mla_o_inv_rope_quant(
            attention.view(token_count, g.main_heads, HEAD_DIM),
            positions_i64,
            rope_cos,
            rope_sin,
            workspace.o_proj_fp8,
            workspace.o_proj_scale,
        )
        o_lora = attn._wo_a_einsum_from_fp8(
            workspace.o_proj_fp8,
            workspace.o_proj_scale,
            batch_size,
            q_len,
        )
        projected = attn._lin(attn.wo_b, o_lora.flatten(2))
        return block.attn_hc.post(
            projected,
            hidden,
            workspace.post.view(batch_size, q_len, HC, 1),
            workspace.comb.view(batch_size, q_len, HC, HC),
        )
