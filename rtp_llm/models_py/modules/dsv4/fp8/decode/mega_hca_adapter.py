"""Thin TP1 DSV4 HCA attention-sublayer megakernel adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import torch

from .mega_csa_runtime import MegaCSARuntime, MegaHCASlotMappings
from .mega_csa_weights import (
    GEOMETRY_BY_DIM,
    HC,
    HEAD_DIM,
    MAX_BATCH,
    O_LORA_RANK,
    ROPE_DIM,
    CSAGeometry,
)
from .mega_hca_weights import HCA_COMPRESS_RATIO, HCA_STATE_WIDTH, MegaHCAWeights

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.dsv4.block import Block


@dataclass(frozen=True)
class MegaHCAPoolContext:
    state_kv: torch.Tensor
    state_gate: torch.Tensor
    compressed_cache: torch.Tensor
    swa_cache: torch.Tensor
    compressed_entries: int
    swa_entries: int
    state_ring_entries: int
    slots: MegaHCASlotMappings


class MegaHCAAdapter:
    """Own per-layer packed weights; orchestration and scratch stay shared.

    HCA has no indexer, no TopK and no MQA stage: opA (front) and opB (q_b)
    carry the whole fused pipeline, the dense compressed index comes from the
    per-step metadata, and the decode-specialized CUDA RMSNorm+RoPE consumes
    the raw query projection published by opB.
    """

    def __init__(
        self,
        block: "Block",
        layer_weights: Dict[str, torch.Tensor],
        runtime: MegaCSARuntime,
    ) -> None:
        self._geometry = self._validate_geometry(block)
        self.weights = MegaHCAWeights.from_layer_weights(layer_weights, self._geometry)
        self.runtime = runtime
        self._runtime_checked = False

    @staticmethod
    def supports_decode_shape(hidden: torch.Tensor, metadata: Any) -> bool:
        """Return whether this request can enter the fixed TP1 kernel geometry."""
        if hidden.dim() != 4:
            return False
        batch_size, q_len = int(hidden.shape[0]), int(hidden.shape[1])
        return (
            batch_size >= 1
            and q_len >= 1
            and batch_size * q_len <= MAX_BATCH
            and int(getattr(metadata, "batch_size", 0)) == batch_size
            and int(getattr(metadata, "q_len_per_req", 0)) == q_len
        )

    @staticmethod
    def _validate_geometry(block: "Block") -> CSAGeometry:
        attn = block.attn
        geometry = GEOMETRY_BY_DIM.get(int(attn.dim))
        if geometry is None:
            raise ValueError(
                f"DSV4 HCA mega geometry mismatch: dim={attn.dim} "
                f"(compiled: {sorted(GEOMETRY_BY_DIM)})"
            )
        expected = (
            ("tp_size", attn.tp_size, 1),
            ("tp_rank", attn.tp_rank, 0),
            ("compress_ratio", attn.compress_ratio, HCA_COMPRESS_RATIO),
            ("q_lora_rank", attn.q_lora_rank, geometry.q_lora_rank),
            ("n_heads", attn.n_heads, geometry.main_heads),
            ("head_dim", attn.head_dim, HEAD_DIM),
            ("rope_head_dim", attn.rope_head_dim, ROPE_DIM),
            ("o_groups", attn.n_groups, geometry.o_groups),
            ("o_lora_rank", attn.o_lora_rank, O_LORA_RANK),
        )
        problems = [
            f"{name}={actual} (expected {wanted})"
            for name, actual, wanted in expected
            if int(actual) != wanted
        ]
        if getattr(attn, "indexer", None) is not None:
            problems.append("HCA layer must not have an indexer")
        if problems:
            raise ValueError("DSV4 HCA mega geometry mismatch: " + "; ".join(problems))
        return geometry

    def _require_runtime(self, device: torch.device) -> Any:
        if self._runtime_checked:
            from rtp_kernel import dsv4_mega

            return dsv4_mega
        from .mega_support import require_mega_runtime

        dsv4_mega = require_mega_runtime(device, ("hca",))
        self._runtime_checked = True
        return dsv4_mega

    def _bind_pools(
        self,
        block: "Block",
        metadata: Any,
        token_count: int,
    ) -> MegaHCAPoolContext:
        from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV, HCA_STATE, SWA_KV

        attn = block.attn
        state = attn._pool_view(HCA_STATE)
        compressed_cache = attn._pool_raw_u8(HCA_KV)
        swa_cache = attn._pool_raw_u8(SWA_KV)
        pools = {
            "HCA_STATE": state,
            "HCA_KV": compressed_cache,
            "SWA_KV": swa_cache,
        }
        missing = [name for name, value in pools.items() if value is None]
        if missing:
            raise RuntimeError("DSV4 mega pools are unavailable: " + ", ".join(missing))
        assert state is not None
        assert compressed_cache is not None
        assert swa_cache is not None

        # HCA_STATE rows interleave ``kv(512) | gate(512)`` fp32. The two
        # kernel arguments are stride-1024 views into the same storage, which
        # is one of the op's two accepted forms.
        if int(state.shape[-1]) != 2 * HCA_STATE_WIDTH:
            raise ValueError(
                "DSV4 mega HCA_STATE rows must interleave kv|gate "
                f"({2 * HCA_STATE_WIDTH} floats), got {int(state.shape[-1])}"
            )
        if state.dtype != torch.float32 or not state.is_contiguous():
            raise TypeError("DSV4 mega HCA_STATE pool must be contiguous fp32")
        state_rows = state.view(-1, 2 * HCA_STATE_WIDTH)
        state_kv = state_rows.narrow(1, 0, HCA_STATE_WIDTH)
        state_gate = state_rows.narrow(1, HCA_STATE_WIDTH, HCA_STATE_WIDTH)

        compressed_entries = attn._pool_entries_per_block(HCA_KV)
        swa_entries = attn._pool_entries_per_block(SWA_KV)
        state_ring_entries = attn._pool_entries_per_block(HCA_STATE)
        if min(compressed_entries, swa_entries) <= 0:
            raise ValueError("DSV4 mega pool geometry contains an empty region")
        # The opB compressor folds the last RATIO logical rows through the
        # runtime-sized ring, so the ring cannot be narrower than RATIO.
        if state_ring_entries < HCA_COMPRESS_RATIO:
            raise ValueError(
                "DSV4 mega HCA state ring must hold at least "
                f"{HCA_COMPRESS_RATIO} rows, got {state_ring_entries}"
            )

        slots = self.runtime.hca_slot_mappings(metadata, token_count)
        return MegaHCAPoolContext(
            state_kv=state_kv,
            state_gate=state_gate,
            compressed_cache=compressed_cache,
            swa_cache=swa_cache,
            compressed_entries=compressed_entries,
            swa_entries=swa_entries,
            state_ring_entries=state_ring_entries,
            slots=slots,
        )

    def forward_attention_sublayer(
        self,
        block: "Block",
        hidden: torch.Tensor,
        metadata: Any,
        *,
        kv_cache: Any = None,
    ) -> torch.Tensor:
        """Run the complete HCA attention sublayer; never falls back after entry."""
        g = self._geometry
        if hidden.dim() != 4 or tuple(hidden.shape[2:]) != (HC, g.dim):
            raise ValueError(
                f"DSV4 mega hidden must be [B,S,{HC},{g.dim}], "
                f"got {tuple(hidden.shape)}"
            )
        batch_size, q_len = int(hidden.shape[0]), int(hidden.shape[1])
        token_count = batch_size * q_len
        if batch_size < 1 or q_len < 1 or token_count > MAX_BATCH:
            raise ValueError(
                f"DSV4 mega requires B>=1, S>=1, and B*S<={MAX_BATCH}; "
                f"got B={batch_size}, S={q_len}"
            )
        if (
            hidden.dtype != torch.bfloat16
            or not hidden.is_cuda
            or not hidden.is_contiguous()
        ):
            raise TypeError("DSV4 mega hidden must be contiguous CUDA bfloat16")
        if metadata.batch_size != batch_size or metadata.q_len_per_req != q_len:
            raise ValueError("DSV4 mega hidden and metadata geometry disagree")
        if metadata.position_ids_long is None:
            raise RuntimeError("DSV4 mega metadata is missing int64 positions")
        dsv4_mega = self._require_runtime(hidden.device)

        attn = block.attn
        previous_kv = attn._kv_cache
        previous_tables = attn._block_tables_by_type
        if kv_cache is not None:
            attn._kv_cache = kv_cache
        attn._block_tables_by_type = metadata.pool_block_tables
        try:
            attn._ensure_freqs_cis_bound()
            pools = self._bind_pools(block, metadata, token_count)
            return self._forward_bound(block, hidden, metadata, pools, dsv4_mega)
        finally:
            attn._kv_cache = previous_kv
            attn._block_tables_by_type = previous_tables

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
        from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV, SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
            get_or_build_sched_meta,
        )

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
        )

        tf32_hc_prenorm_gemm(
            hidden_rows.view(token_count, HC * g.dim),
            self.weights.hc_fn,
            workspace.hc_partial,
            workspace.hc_sum_sq,
            num_split,
        )
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


__all__ = ["MegaHCAAdapter"]
