"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_adapter import (
    MegaHCAPoolContext,
    __all__,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict
import torch
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_runtime import (
    MegaCSARuntime,
    MegaHCASlotMappings,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
    GEOMETRY_BY_DIM,
    HC,
    HEAD_DIM,
    MAX_BATCH,
    O_LORA_RANK,
    ROPE_DIM,
    CSAGeometry,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_weights import (
    HCA_COMPRESS_RATIO,
    HCA_STATE_WIDTH,
    MegaHCAWeights,
)

if TYPE_CHECKING:
    from rtp_llm.platforms.ppu.models.dsv4.impl.block import Block

from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_adapter import MegaHCAAdapter as _BaseMegaHCAAdapter

class MegaHCAAdapter(_BaseMegaHCAAdapter):
    def _forward_bound(
        self,
        block: "Block",
        hidden: torch.Tensor,
        metadata: Any,
        pools: MegaHCAPoolContext,
        dsv4_mega: Any,
    ) -> torch.Tensor:
        g = self._geometry
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.decode.decode_attn_metadata import get_or_build_sched_meta
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import HCA_KV, SWA_KV

        attn = block.attn
        batch_size, q_len = int(hidden.shape[0]), int(hidden.shape[1])
        token_count = batch_size * q_len
        positions_i64 = metadata.position_ids_long[:token_count]
        if positions_i64.dtype != torch.int64:
            raise TypeError("DSV4 mega positions must provide an int64 tensor")
        rope_cos, rope_sin = self.runtime.rope_tables(attn.freqs_cis)
        num_split = self.runtime.num_hc_splits(token_count, hidden.device, g.dim)
        workspace = self.runtime.hca_layer_workspace(
            token_count, num_split, hidden.device, g
        )
        hidden_rows = hidden.view(token_count, HC, g.dim)

        # Dense compressed index — built once per step for every HCA layer.
        window = int(attn.window_size)
        dense_total = metadata.topk_total_by_ratio.get(HCA_COMPRESS_RATIO)
        if (
            dense_total is None
            or dense_total.dim() != 3
            or int(dense_total.shape[0]) < batch_size
            or int(dense_total.shape[1]) < q_len
        ):
            raise RuntimeError(
                "DSV4 mega metadata is missing the HCA dense compressed index"
            )
        cmp_local_raw = dense_total[:batch_size, :q_len, window:]

        for attn_type, name in ((SWA_KV, "SWA_KV"), (HCA_KV, "HCA_KV")):
            pool = attn._pool_view_3d_fp8(attn_type)
            block_table = metadata.pool_block_tables.get(attn_type)
            if pool is None or block_table is None or block_table.numel() == 0:
                raise RuntimeError(f"DSV4 mega FlashMLA input {name} is unavailable")
        if (
            metadata.swa_abs_idx is None
            or metadata.req_id_per_token is None
            or metadata.swa_global_slots is None
            or HCA_KV not in metadata.paged_pool_tokens_per_block
        ):
            raise RuntimeError("DSV4 mega metadata is incomplete for native FlashMLA")
        attn._get_fp8_decode_op()
        get_or_build_sched_meta(
            metadata,
            batch_size=batch_size,
            q_len=q_len,
            num_heads=attn.n_heads,
            topk=window,
            extra_attn_type=HCA_KV,
            extra_index_width=int(cmp_local_raw.shape[-1]),
        )

        tf32_hc_prenorm_gemm(
            hidden_rows.view(token_count, HC * g.dim),
            self.weights.hc_fn,
            workspace.hc_partial,
            workspace.hc_sum_sq,
            num_split,
        )
        # The lite reduction writes mix; front_mixed_gemm_hca's fused HC tail
        # produces post/comb before attn_hc.post consumes them.
        dsv4_mega.hc_reduce_fuse_out(
            hidden_rows,
            workspace.hc_partial,
            workspace.hc_sum_sq,
            self.weights.hc_base,
            self.weights.hc_scale,
            block.attn_hc.hc_eps,
            block.attn_hc.norm_eps,
            workspace.collapsed[:token_count],
            workspace.pre,
            workspace.post,
            workspace.comb,
            with_post_comb=False,
            attn_norm_w=self.weights.attn_norm,
            attn_norm_eps=block.attn_hc.norm_eps,
            mix_out=workspace.mix,
            xq_out=workspace.hidden_fp8[:token_count].view(torch.uint8),
            xsf_out=workspace.hidden_sf[:token_count],
            pdl=False,
        )
        # For physical batches there is no padding memset between the fused
        # reduce producer and front, so front can overlap its launch prologue.
        # Sub-16 batches retain stream ordering for the padded-tail writes.
        front_pdl = token_count >= 16
        dsv4_mega.front_mixed_gemm_hca(
            workspace.collapsed,
            workspace.hidden_fp8,
            workspace.hidden_sf,
            self.weights.front_bf16,
            self.weights.front_fp8,
            self.weights.front_sf,
            workspace.front_out,
            workspace.mix,
            self.weights.hc_base,
            self.weights.hc_scale,
            workspace.post,
            workspace.comb.view(token_count, HC * HC),
            positions_i64,
            pools.slots.state_rows,
            self.weights.compressor_ape,
            pools.state_kv,
            pools.state_gate,
            token_count,
            hc_eps=block.attn_hc.hc_eps,
            pdl=front_pdl,
        )
        q_raw = dsv4_mega.wq_b_proj_gemm_merged_hca(
            workspace.front_out,
            self.weights.wq_b_fp8,
            self.weights.wq_b_sf,
            self.weights.q_norm,
            self.weights.window_norm,
            self.weights.compressor_norm,
            positions_i64,
            rope_cos,
            rope_sin,
            pools.state_kv,
            pools.state_gate,
            pools.slots.state_rows,
            pools.swa_cache,
            pools.slots.window_destinations,
            pools.compressed_cache,
            pools.slots.compressed_destinations,
            workspace.q_raw,
            q_norm_eps=attn.eps,
            window_page_tokens=pools.swa_entries,
            compressed_page_tokens=pools.compressed_entries,
            state_ring_entries=pools.state_ring_entries,
        )
        # Index the shared RoPE table inside the normalization kernel so every
        # HCA layer avoids materializing a gathered frequency tensor.
        q_ready = dsv4_mega.q_rmsnorm_rope_cuda_(
            q_raw.view(batch_size, q_len, g.main_heads, HEAD_DIM),
            attn.freqs_cis,
            positions_i64,
            eps=attn.eps,
        )
        attention = attn._forward_decode_compressed(
            q_ready,
            cmp_local_raw,
            batch_size,
            q_len,
            metadata,
            cmp_attn_type=HCA_KV,
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
