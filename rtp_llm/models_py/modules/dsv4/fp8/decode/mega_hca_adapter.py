"""Thin TP1 DSV4-Pro HCA attention-sublayer megakernel adapter."""

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
    PRO_GEOMETRY,
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
    per-step metadata, and query RMSNorm+RoPE stays on the framework Triton
    pass because opB publishes the raw projection on purpose.
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

        capability = torch.cuda.get_device_capability(device)
        if capability not in ((10, 0), (10, 3)):
            raise RuntimeError(
                "DSV4 Mega HCA requires sm_100a or sm_103a, "
                f"got sm_{capability[0]}{capability[1]}"
            )

        from rtp_kernel import dsv4_mega

        required = (
            "geometry_hca",
            "hc_reduce_fuse_out",
            "front_mixed_gemm_hca",
            "wq_b_proj_gemm_merged_hca",
            "mla_o_inv_rope_quant",
        )
        missing = [name for name in required if not hasattr(dsv4_mega, name)]
        if missing:
            raise RuntimeError(
                "rtp-kernel does not provide the DSV4 TP1 HCA ABI: "
                + ", ".join(missing)
            )
        import inspect

        required_parameters = {
            "front_mixed_gemm_hca": (
                "normalized_mix",
                "state_slot_mapping",
                "state_kv",
                "state_gate",
                "pdl",
            ),
            "wq_b_proj_gemm_merged_hca": (
                "state_ring_entries",
                "window_cache",
                "compressed_cache",
                "window_page_tokens",
                "compressed_page_tokens",
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
                "rtp-kernel DSV4 TP1 HCA ABI is incompatible: "
                + "; ".join(incompatible)
            )
        geometry = dsv4_mega.geometry_hca()
        g = self._geometry
        suffix = "_pro" if g is PRO_GEOMETRY else "_flash"
        expected = {
            f"n_q{suffix}": g.n_main,
            f"front_n_fp8{suffix}": g.front_fp8_rows,
            "compress_ratio": HCA_COMPRESS_RATIO,
            "state_width": HCA_STATE_WIDTH,
            "slot_dtype_bits": 64,
        }
        mismatched = {
            key: (geometry.get(key), want)
            for key, want in expected.items()
            if geometry.get(key) != want
        }
        if mismatched:
            raise RuntimeError(f"rtp-kernel DSV4 HCA geometry mismatch: {mismatched}")
        import deep_gemm

        if not hasattr(deep_gemm, "tf32_hc_prenorm_gemm"):
            raise RuntimeError(
                "DeepGEMM is missing required DSV4 API tf32_hc_prenorm_gemm"
            )
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
        pools: MegaHCAPoolContext,
        dsv4_mega: Any,
    ) -> torch.Tensor:
        g = self._geometry
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm
        from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
            fused_rmsnorm_rope,
        )
        from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV, SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
            get_or_build_sched_meta,
        )

        attn = block.attn
        m = int(hidden.shape[0])
        positions_i64 = metadata.position_ids_long[:m]
        if positions_i64.dtype != torch.int64:
            raise TypeError("DSV4 mega positions must provide an int64 tensor")
        rope_cos, rope_sin = self.runtime.rope_tables(attn.freqs_cis)
        num_split = self.runtime.num_hc_splits(m, hidden.device, g.dim)
        workspace = self.runtime.hca_layer_workspace(m, num_split, hidden.device, g)
        hidden_rows = hidden.view(m, HC, g.dim)

        # Dense compressed index — built once per step for every HCA layer.
        window = int(attn.window_size)
        dense_total = metadata.topk_total_by_ratio.get(HCA_COMPRESS_RATIO)
        if dense_total is None or int(dense_total.shape[0]) < m:
            raise RuntimeError(
                "DSV4 mega metadata is missing the HCA dense compressed index"
            )
        cmp_local_raw = dense_total[:m, :, window:]

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
            batch_size=m,
            q_len=1,
            num_heads=attn.n_heads,
            topk=window,
            extra_attn_type=HCA_KV,
        )

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
            workspace.collapsed[:m],
            workspace.pre,
            workspace.post,
            workspace.comb,
            with_post_comb=False,
            attn_norm_w=self.weights.attn_norm,
            attn_norm_eps=block.attn_hc.norm_eps,
            mix_out=workspace.mix,
            xq_out=workspace.hidden_fp8[:m].view(torch.uint8),
            xsf_out=workspace.hidden_sf[:m],
            pdl=False,
        )
        # PDL stays off by design: the op's bf16 feature tasks read
        # ``collapsed`` with no predecessor wait, and ``hc_reduce_fuse_out``
        # writes it (see the op docstring).
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
            workspace.comb.view(m, HC * HC),
            positions_i64,
            pools.slots.state_rows,
            self.weights.compressor_ape,
            pools.state_kv,
            pools.state_gate,
            m,
            hc_eps=block.attn_hc.hc_eps,
            pdl=False,
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
        # opB publishes the raw projection; per-head RMSNorm + partial RoPE
        # stay on the same framework Triton pass the original path uses.
        freqs_cis = attn.freqs_cis.index_select(0, positions_i64).contiguous()
        q_ready = fused_rmsnorm_rope(
            q_raw.view(m, 1, g.main_heads, HEAD_DIM),
            None,
            freqs_cis,
            ROPE_DIM,
            eps=attn.eps,
            inplace=True,
        )
        attention = attn._forward_decode_compressed(
            q_ready,
            cmp_local_raw,
            m,
            1,
            metadata,
            cmp_attn_type=HCA_KV,
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


__all__ = ["MegaHCAAdapter"]
