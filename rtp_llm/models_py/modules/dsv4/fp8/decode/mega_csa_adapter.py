"""Thin TP1 DSV4-Pro CSA attention-sublayer megakernel adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from .mega_csa_runtime import MegaCSARuntime, MegaCSASlotMappings
from .mega_csa_weights import (
    COMPRESS_RATIO,
    GEOMETRY_BY_DIM,
    HC,
    HEAD_DIM,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    MAX_BATCH,
    O_LORA_RANK,
    PRO_GEOMETRY,
    ROPE_DIM,
    CSAGeometry,
    MegaCSAWeights,
)

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.dsv4.block import Block


@dataclass(frozen=True)
class MegaCSAPoolContext:
    main_state: torch.Tensor
    indexer_state: torch.Tensor
    main_cache: torch.Tensor
    indexer_cache: torch.Tensor
    swa_cache: torch.Tensor
    indexer_block_table: torch.Tensor
    main_entries: int
    indexer_entries: int
    swa_entries: int
    main_state_entries: int
    indexer_state_entries: int
    main_stride_bytes: int
    indexer_stride_bytes: int
    swa_stride_bytes: int
    slots: MegaCSASlotMappings


class MegaCSAAdapter:
    """Own per-layer packed weights; orchestration and scratch stay shared."""

    def __init__(
        self,
        block: "Block",
        layer_weights: Dict[str, torch.Tensor],
        runtime: MegaCSARuntime,
    ) -> None:
        self._geometry = self._validate_geometry(block)
        self.weights = MegaCSAWeights.from_layer_weights(layer_weights, self._geometry)
        self.runtime = runtime
        self._runtime_checked = False

    @staticmethod
    def supports_decode_shape(hidden: torch.Tensor, metadata: Any) -> bool:
        """Return whether this request can enter the fixed TP1 kernel geometry."""
        return (
            int(getattr(metadata, "q_len_per_req", 0)) == 1
            and hidden.dim() == 4
            and 1 <= int(hidden.shape[0]) <= MAX_BATCH
        )

    @staticmethod
    def _validate_geometry(block: "Block") -> CSAGeometry:
        attn = block.attn
        geometry = GEOMETRY_BY_DIM.get(int(attn.dim))
        if geometry is None:
            raise ValueError(
                f"DSV4 CSA mega geometry mismatch: dim={attn.dim} "
                f"(compiled: {sorted(GEOMETRY_BY_DIM)})"
            )
        expected = (
            ("tp_size", attn.tp_size, 1),
            ("tp_rank", attn.tp_rank, 0),
            ("compress_ratio", attn.compress_ratio, COMPRESS_RATIO),
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
        indexer = getattr(attn, "indexer", None)
        if indexer is None:
            problems.append("CSA indexer is missing")
        else:
            if int(indexer.n_heads) != INDEX_HEADS:
                problems.append(
                    f"index_n_heads={indexer.n_heads} (expected {INDEX_HEADS})"
                )
            if int(indexer.head_dim) != INDEX_HEAD_DIM:
                problems.append(
                    f"index_head_dim={indexer.head_dim} (expected {INDEX_HEAD_DIM})"
                )
        if problems:
            raise ValueError("DSV4 CSA mega geometry mismatch: " + "; ".join(problems))
        return geometry

    def _require_runtime(self, device: torch.device) -> Any:
        if self._runtime_checked:
            from rtp_kernel import dsv4_mega

            return dsv4_mega

        capability = torch.cuda.get_device_capability(device)
        if capability not in ((10, 0), (10, 3)):
            raise RuntimeError(
                "DSV4 Mega CSA requires sm_100a or sm_103a, "
                f"got sm_{capability[0]}{capability[1]}"
            )

        from rtp_kernel import dsv4_mega

        required = (
            "geometry_csa",
            "hc_reduce_fuse_out",
            "front_mixed_gemm_csa",
            "wq_b_proj_gemm_merged_csa",
            "mqa_logits_fp8_decode_out",
            "mla_o_inv_rope_quant",
        )
        missing = [name for name in required if not hasattr(dsv4_mega, name)]
        if missing:
            raise RuntimeError(
                "rtp-kernel does not provide the DSV4 TP1 CSA ABI: "
                + ", ".join(missing)
            )
        import inspect

        required_parameters = {
            "front_mixed_gemm_csa": ("hc_mix", "main_state", "idx_state", "pdl"),
            "wq_b_proj_gemm_merged_csa": (
                "indexer_fp8",
                "idx_cache",
                "swa_cache",
                "iq_dst",
                "pdl",
            ),
            "mqa_logits_fp8_decode_out": (
                "schedule_meta",
                "comp_state",
                "cmp_cache",
                "query_x",
                "query_out",
                "pdl",
            ),
            "mla_o_inv_rope_quant": (
                "input",
                "positions",
                "rope_cos",
                "rope_sin",
                "output_fp8",
                "output_scale",
            ),
        }
        incompatible = []
        for function_name, parameters in required_parameters.items():
            signature = inspect.signature(getattr(dsv4_mega, function_name))
            absent = [name for name in parameters if name not in signature.parameters]
            if absent:
                incompatible.append(f"{function_name} missing {','.join(absent)}")
        if incompatible:
            raise RuntimeError(
                "rtp-kernel DSV4 TP1 CSA ABI is incompatible: "
                + "; ".join(incompatible)
            )
        geometry = dsv4_mega.geometry_csa()
        g = self._geometry
        suffix = "" if g is PRO_GEOMETRY else "_flash"
        expected = {
            f"n_main{suffix}": g.n_main,
            "n_index": INDEX_HEADS * INDEX_HEAD_DIM,
            f"n_merged{suffix}": g.n_merged,
            f"num_main_heads{suffix}": g.main_heads,
            "num_index_heads": INDEX_HEADS,
            "slot_dtype_bits": 64,
        }
        mismatched = {
            key: geometry.get(key)
            for key, value in expected.items()
            if geometry.get(key) != value
        }
        if mismatched:
            raise RuntimeError(
                f"rtp-kernel DSV4 geometry mismatch: got {mismatched}, "
                f"expected {expected}"
            )
        import deep_gemm

        for name in (
            "tf32_hc_prenorm_gemm",
            "get_paged_mqa_logits_metadata",
            "get_num_sms",
        ):
            if not hasattr(deep_gemm, name):
                raise RuntimeError(f"DeepGEMM is missing required DSV4 API {name}")
        self._runtime_checked = True
        return dsv4_mega

    @staticmethod
    def _pool_stride_bytes(raw: torch.Tensor) -> int:
        return int(raw.stride(0) * raw.element_size())

    def _bind_pools(
        self,
        block: "Block",
        metadata: Any,
        token_count: int,
    ) -> MegaCSAPoolContext:
        from rtp_llm.models_py.modules.dsv4.attn_type import (
            CSA_KV,
            CSA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
            SWA_KV,
        )

        attn = block.attn
        main_state = attn._pool_view(CSA_STATE)
        indexer_state = attn._pool_view(INDEXER_STATE)
        main_cache = attn._pool_raw_u8(CSA_KV)
        indexer_cache = attn._pool_raw_u8(INDEXER_KV)
        swa_cache = attn._pool_raw_u8(SWA_KV)
        pools = {
            "CSA_STATE": main_state,
            "INDEXER_STATE": indexer_state,
            "CSA_KV": main_cache,
            "INDEXER_KV": indexer_cache,
            "SWA_KV": swa_cache,
        }
        missing = [name for name, value in pools.items() if value is None]
        if missing:
            raise RuntimeError("DSV4 mega pools are unavailable: " + ", ".join(missing))
        assert main_state is not None
        assert indexer_state is not None
        assert main_cache is not None
        assert indexer_cache is not None
        assert swa_cache is not None

        indexer_block_table = metadata.pool_block_tables.get(INDEXER_KV)
        if (
            indexer_block_table is None
            or int(indexer_block_table.shape[0]) < token_count
        ):
            raise RuntimeError("DSV4 mega metadata is missing INDEXER_KV block table")
        indexer_block_table = indexer_block_table[:token_count]
        if (
            indexer_block_table.dtype != torch.int32
            or not indexer_block_table.is_contiguous()
        ):
            raise TypeError("DSV4 mega INDEXER_KV block table must be contiguous int32")

        main_entries = attn._pool_entries_per_block(CSA_KV)
        indexer_entries = attn._pool_entries_per_block(INDEXER_KV)
        swa_entries = attn._pool_entries_per_block(SWA_KV)
        main_state_entries = attn._pool_entries_per_block(CSA_STATE)
        indexer_state_entries = attn._pool_entries_per_block(INDEXER_STATE)
        if indexer_entries not in (32, 64, 128):
            raise ValueError(
                "DSV4 mega INDEXER_KV entries must be 32, 64, or 128, "
                f"got {indexer_entries}"
            )
        if (
            min(
                main_entries,
                swa_entries,
                main_state_entries,
                indexer_state_entries,
            )
            <= 0
        ):
            raise ValueError("DSV4 mega pool geometry contains an empty region")

        slots = self.runtime.slot_mappings(metadata, token_count)
        return MegaCSAPoolContext(
            main_state=main_state,
            indexer_state=indexer_state,
            main_cache=main_cache,
            indexer_cache=indexer_cache,
            swa_cache=swa_cache,
            indexer_block_table=indexer_block_table,
            main_entries=main_entries,
            indexer_entries=indexer_entries,
            swa_entries=swa_entries,
            main_state_entries=main_state_entries,
            indexer_state_entries=indexer_state_entries,
            main_stride_bytes=self._pool_stride_bytes(main_cache),
            indexer_stride_bytes=self._pool_stride_bytes(indexer_cache),
            swa_stride_bytes=self._pool_stride_bytes(swa_cache),
            slots=slots,
        )

    def forward_attention_sublayer(
        self,
        block: "Block",
        hidden: torch.Tensor,
        metadata: Any,
        *,
        kv_cache: Optional[Any] = None,
    ) -> torch.Tensor:
        """Run the complete CSA attention sublayer; never falls back after entry."""
        g = self._geometry
        if hidden.dim() != 4 or tuple(hidden.shape[2:]) != (HC, g.dim):
            raise ValueError(
                f"DSV4 mega hidden must be [B,1,{HC},{g.dim}], "
                f"got {tuple(hidden.shape)}"
            )
        m, q_len = int(hidden.shape[0]), int(hidden.shape[1])
        if q_len != 1 or not 1 <= m <= MAX_BATCH:
            raise ValueError(f"DSV4 mega requires q_len=1 and batch in [1,{MAX_BATCH}]")
        if (
            hidden.dtype != torch.bfloat16
            or not hidden.is_cuda
            or not hidden.is_contiguous()
        ):
            raise TypeError("DSV4 mega hidden must be contiguous CUDA bfloat16")
        if metadata.batch_size != m or metadata.q_len_per_req != 1:
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
            pools = self._bind_pools(block, metadata, m)
            return self._forward_bound(block, hidden, metadata, pools, dsv4_mega)
        finally:
            attn._kv_cache = previous_kv
            attn._block_tables_by_type = previous_tables

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
        from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
            get_or_build_sched_meta,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.indexer import _get_topk_workspace
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        attn = block.attn
        m = int(hidden.shape[0])
        positions_i32 = metadata.position_ids[:m]
        positions_i64 = metadata.position_ids_long[:m]
        if positions_i32.dtype != torch.int32 or positions_i64.dtype != torch.int64:
            raise TypeError("DSV4 mega positions must provide int32 and int64 tensors")
        rope_cos, rope_sin = self.runtime.rope_tables(attn.freqs_cis)
        num_split = self.runtime.num_hc_splits(m, hidden.device, g.dim)
        workspace = self.runtime.layer_workspace(m, num_split, hidden.device, g)
        hidden_rows = hidden.view(m, HC, g.dim)

        compressed_lengths = metadata.compressed_lens.get(COMPRESS_RATIO)
        if compressed_lengths is None or int(compressed_lengths.numel()) < m:
            raise RuntimeError("DSV4 mega metadata is missing CSA compressed lengths")
        compressed_lengths = compressed_lengths[:m]
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
            topk_buffer.dtype != torch.int32
            or not topk_buffer.is_cuda
            or not topk_buffer.is_contiguous()
            or int(topk_buffer.numel()) < m * topk
        ):
            raise TypeError("DSV4 mega TopK output must be contiguous CUDA int32")
        topk_output = topk_buffer[:m].reshape(m, topk)

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
            batch_size=m,
            q_len=1,
            num_heads=attn.n_heads,
            topk=attn.window_size,
            extra_attn_type=CSA_KV,
        )

        score_capacity = int(pools.indexer_block_table.shape[1]) * pools.indexer_entries
        logits = self.runtime.logits(m, score_capacity, hidden.device)
        schedule = self.runtime.mqa_schedule(compressed_lengths, pools.indexer_entries)
        topk_workspace = _get_topk_workspace(hidden.device)

        tf32_hc_prenorm_gemm(
            hidden_rows.view(m, HC * g.dim),
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
            workspace.comb.view(m, HC * HC),
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
            head_ssq=None,
            mock_post=False,
            enable_ssq=False,
            cmp_pos=positions_i64,
            idx_norm=self.weights.indexer_norm,
            cos_tab=rope_cos,
            sin_tab=rope_sin,
            idx_state=pools.indexer_state,
            idx_state_row=pools.slots.indexer_state_rows,
            state_ring_entries=pools.indexer_state_entries,
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
        q_ready = q_raw.view(m, 1, g.main_heads, HEAD_DIM)

        dsv4_mega.mqa_logits_fp8_decode_out(
            indexer_q,
            pools.indexer_cache,
            folded_weights,
            compressed_lengths.view(m, 1),
            pools.indexer_block_table,
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
            topk_output.view(m, 1, topk),
            m,
            1,
            metadata,
            cmp_attn_type=CSA_KV,
        )
        dsv4_mega.mla_o_inv_rope_quant(
            attention.view(m, g.main_heads, HEAD_DIM),
            positions_i64,
            rope_cos,
            rope_sin,
            workspace.o_proj_fp8,
            workspace.o_proj_scale,
        )
        o_lora = attn._wo_a_einsum_from_fp8(
            workspace.o_proj_fp8, workspace.o_proj_scale, m, 1
        )
        projected = attn._lin(attn.wo_b, o_lora.flatten(2))
        return block.attn_hc.post(
            projected,
            hidden,
            workspace.post.view(m, 1, HC, 1),
            workspace.comb.view(m, 1, HC, HC),
        )


__all__ = ["MegaCSAAdapter"]
