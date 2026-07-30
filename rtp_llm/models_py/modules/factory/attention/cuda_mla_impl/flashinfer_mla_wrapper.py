import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MlaImplBase
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, rtp_llm_ops

from .flashinfer_mla import (
    MlaFlashInferDecodeOp,
    MlaFlashInferPrefillOp,
    check_attention_inputs,
    warmup_flashinfer_python,
)
from .rope_emb_new import NewMlaRotaryEmbeddingOp


class MlaFlashInferImplBase(MlaImplBase):

    def __init__(
        self,
        fmha_impl: Any,
        rope_impl: Any,
        kv_cache_write_op: MlaKVCacheWriteOp,
        attn_inputs: PyAttentionInputs,
        seq_size_per_block: int,
        attn_configs: AttentionConfigs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        super().__init__(
            attn_configs,
            attn_inputs,
            weights,
            cos_sin_cache,
            fmha_config,
            use_trt_fmha=use_trt_fmha,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
        )
        warmup_flashinfer_python()
        self.seq_size_per_block = seq_size_per_block
        self.fmha_impl: Any = fmha_impl
        if (
            self.fmha_impl is not None
            and os.environ.get("KIMI_K3_USE_HOST_METADATA", "0") == "1"
        ):
            input_host = getattr(attn_inputs, "input_lengths_host", None)
            prefix_host = getattr(attn_inputs, "prefix_lengths_host", None)
            if input_host is not None and input_host.numel():
                input_values = [int(value) for value in input_host.tolist()]
                prefix_values = (
                    [int(value) for value in prefix_host.tolist()]
                    if prefix_host is not None and prefix_host.numel()
                    else [0] * len(input_values)
                )
                self.fmha_impl.total_kv_lens_hint = sum(input_values) + sum(
                    prefix_values
                )
        self.fmha_params = None
        self.rope_params = None
        self.rope_impl = rope_impl
        self.kv_cache_write_op = kv_cache_write_op
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        self.create_params(attn_inputs)

    def create_params(self, attn_inputs: PyAttentionInputs):
        if self.fmha_impl is not None:
            self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
            self.rope_params = self.fmha_params
            self.prepare(attn_inputs)

    def prepare(self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False):
        """Update fmha_params for prepare or CUDA Graph replay.

        Note: fmha_params is initialized in __init__, this method only updates it.
        forbid_realloc: True only when called from prepare_cuda_graph (replay); forbids buffer realloc.
        """
        assert self.fmha_impl is not None
        assert (
            self.fmha_params is not None
        ), "fmha_params should be initialized in __init__"
        # HybridCache switches the explicit PyAttentionInputs view per group.
        # Keep cache-store consumers on the same view as the planner.
        self.attn_inputs = attn_inputs
        check_attention_inputs(attn_inputs)
        use_host_metadata = os.environ.get("KIMI_K3_USE_HOST_METADATA", "0") == "1"
        prefix_lengths = (
            getattr(attn_inputs, "prefix_lengths_host", None)
            if use_host_metadata
            else None
        )
        sequence_lengths = (
            getattr(attn_inputs, "sequence_lengths_host", None)
            if use_host_metadata
            else None
        )
        input_lengths = (
            getattr(attn_inputs, "input_lengths_host", None)
            if use_host_metadata
            else None
        )
        prefix_lengths = (
            prefix_lengths
            if prefix_lengths is not None and prefix_lengths.numel()
            else attn_inputs.prefix_lengths
        )
        sequence_lengths = (
            sequence_lengths
            if sequence_lengths is not None and sequence_lengths.numel()
            else attn_inputs.sequence_lengths
        )
        input_lengths = (
            input_lengths
            if input_lengths is not None and input_lengths.numel()
            else attn_inputs.input_lengths
        )
        self.fmha_params.fill_params(
            prefix_lengths,
            sequence_lengths,
            input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            self.seq_size_per_block,
            forbid_realloc,
        )
        self.fmha_impl.plan(self.fmha_params)

    def _device_decode_slot_mapping(self) -> Optional[torch.Tensor]:
        """Build the current HybridCache group's decode write locations.

        The regular host planner populates ``fmha_params.slot_mapping``.  The
        graph-safe group refresh intentionally skips that planner, so derive
        the same mapping from its device metadata and the currently selected
        group block table.  The tensor operations are captured and therefore
        read live sequence lengths and block IDs on every replay.
        """

        assert self.fmha_params is not None
        slot_mapping = getattr(self.fmha_params, "slot_mapping", None)
        if slot_mapping is not None:
            return None

        # The pybind surface exposes the stable device buffers, not the
        # transient C++ aliases (positions/batch_indice).
        positions = getattr(self.fmha_params, "positions_d", None)
        batch_indices = getattr(self.fmha_params, "batch_indice_d", None)
        block_table = getattr(
            self.attn_inputs, "kv_cache_kernel_block_id_device", None
        )
        if (
            positions is None
            or batch_indices is None
            or block_table is None
            or positions.numel() == 0
            or batch_indices.numel() == 0
            or block_table.numel() == 0
        ):
            raise RuntimeError(
                "CUDA Graph MLA cache write requires device positions, "
                "batch indices, and the selected group block table"
            )

        positions_i64 = positions.to(torch.int64)
        batch_indices_i64 = batch_indices.to(torch.int64)
        block_indices = torch.div(
            positions_i64,
            self.seq_size_per_block,
            rounding_mode="floor",
        )
        block_numbers = block_table[batch_indices_i64, block_indices].to(torch.int64)
        return (
            block_numbers * self.seq_size_per_block
            + torch.remainder(positions_i64, self.seq_size_per_block)
        )

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (
            topk_indices is None
        ), "topk_indices should be None for MlaFlashInferImplBase"
        assert self.rope_impl is not None and self.fmha_params is not None

        def profile(stage: str, shape: tuple[int, ...]):
            if os.environ.get("KIMI_K3_PERF_MODE", "0").strip() != "1":
                return nullcontext()
            shape_text = "x".join(str(dim) for dim in shape)
            return torch.autograd.profiler.record_function(
                f"layer.{layer_id}.mla.{stage}[shape={shape_text}]"
            )

        q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]

        # Apply RoPE to Q and K
        with profile("nope_rope_adapter", tuple(q_pe.shape)):
            self.rope_impl.forward(q_pe, k_pe, self.rope_params)

        # Write compressed KV and position-encoded K to cache
        with profile(
            "kv_cache_update_normalized_latent_plus_suffix",
            tuple(compressed_kv.shape),
        ):
            self.kv_cache_write_op.forward(
                compressed_kv,
                k_pe,
                kv_cache,
                self.rope_params,
                slot_mapping_override=self._device_decode_slot_mapping(),
            )

        with profile("cache_store_publish_noop_for_standalone_prefill", ()):
            common.apply_write_cache_store(
                self.write_cache_store_impl, self.attn_inputs, kv_cache
            )

        # Split query for FMHA
        q_nope, q_pe = torch.split(
            q,
            [self.fmha_impl.qk_nope_head_dim, self.fmha_impl.qk_rope_head_dim],
            dim=-1,
        )
        assert self.fmha_impl is not None
        with profile("flashinfer_causal_attention_context", tuple(q.shape)):
            res = self.fmha_impl.forward(q_nope, q_pe, kv_cache, layer_id)
        return res


class MlaFlashInferPrefillImpl(MlaFlashInferImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        super().__init__(
            MlaFlashInferPrefillOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.v_head_dim,
                attn_configs.kernel_tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                weights,
                quant_config,
                attn_configs.kv_cache_dtype,
            ),
            NewMlaRotaryEmbeddingOp(
                cos_sin_cache=cos_sin_cache,
                is_neox_style=attn_configs.rope_config.is_neox_style,
            ),
            MlaKVCacheWriteOp(
                kv_cache_dtype=attn_configs.kv_cache_dtype,
            ),
            attn_inputs,
            attn_configs.kernel_tokens_per_block,
            attn_configs,
            weights,
            cos_sin_cache,
            fmha_config,
            use_trt_fmha,
            quant_config,
            max_seq_len,
            is_cuda_graph,
            parallelism_config,
        )
        self.has_reuse_cache = False
        use_host_metadata = os.environ.get("KIMI_K3_USE_HOST_METADATA", "0") == "1"
        prefix_host = (
            getattr(attn_inputs, "prefix_lengths_host", None)
            if use_host_metadata
            else None
        )
        if prefix_host is not None and prefix_host.numel():
            self.has_reuse_cache = max(int(v) for v in prefix_host.tolist()) > 0
        elif attn_inputs.prefix_lengths is not None:
            # Compatibility fallback for older bindings.
            self.has_reuse_cache = bool(attn_inputs.prefix_lengths.max().item() > 0)

        self.absorb_opt_len = (
            fmha_config.absorb_opt_len if fmha_config is not None else 1024
        )
        input_host = (
            getattr(attn_inputs, "input_lengths_host", None)
            if use_host_metadata
            else None
        )
        q_len = (
            sum(int(v) for v in input_host.tolist())
            if input_host is not None and input_host.numel()
            else attn_inputs.input_lengths.sum().item()
        )
        self.absorb_fmha: Optional[MlaFlashInferDecodeOp] = None
        if (
            q_len < self.absorb_opt_len
            and self.has_reuse_cache
            and attn_configs.kv_cache_dtype == KvCacheDataType.BASE
        ):
            self.absorb_fmha = MlaFlashInferDecodeOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.kernel_tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                attn_configs.is_sparse,
                weights,
            )
            self.absorb_fmha.plan(self.fmha_params)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return attn_configs.use_mla and attn_inputs.is_prefill

    def _handle_long_sequence(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ):
        """Handle long sequences using cache reuse operation."""
        # Handle cache reuse for longer sequences
        return self.fmha_impl.forward(q, compressed_kv, k_pe, kv_cache, layer_id)

    def _handle_short_sequence(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache], layer_id: int
    ) -> torch.Tensor:
        """Handle short sequences using absorb operation."""
        # Split query into nope and pe components
        assert self.absorb_fmha is not None, "absorb_fmha is not initialized"
        q_nope, q_pe = torch.split(
            q,
            [self.absorb_fmha.qk_nope_head_dim, self.absorb_fmha.qk_rope_head_dim],
            dim=-1,
        )

        return self.absorb_fmha.forward(q_nope, q_pe, kv_cache, layer_id)

    def compute_prefill_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ):
        """Compute prefill context with optimized cache reuse logic."""

        if self.absorb_fmha is not None:
            return self._handle_short_sequence(q, kv_cache, layer_id)
        else:
            return self._handle_long_sequence(
                q, compressed_kv, k_pe, kv_cache, layer_id
            )

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (
            topk_indices is None
        ), "topk_indices should be None for MlaFlashInferPrefillImpl"
        assert self.rope_impl is not None and self.rope_params is not None
        q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]

        # Apply RoPE to Q and K
        self.rope_impl.forward(q_pe, k_pe, self.rope_params)

        # Write compressed KV and position-encoded K to cache
        self.kv_cache_write_op.forward(compressed_kv, k_pe, kv_cache, self.rope_params)

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        assert self.fmha_impl is not None
        return self.compute_prefill_context(q, compressed_kv, k_pe, kv_cache, layer_id)


class MlaFlashInferDecodeImpl(MlaFlashInferImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        use_host_metadata = os.environ.get("KIMI_K3_USE_HOST_METADATA", "0") == "1"
        sequence_lengths_host = (
            getattr(attn_inputs, "sequence_lengths_host", None)
            if use_host_metadata
            else None
        )
        if sequence_lengths_host is not None and sequence_lengths_host.numel() > 0:
            sequence_values = [int(value) for value in sequence_lengths_host.tolist()]
            max_bs = len(sequence_values)
            num_tokens = sum(sequence_values)
        else:
            max_bs = attn_inputs.sequence_lengths.size(0)
            num_tokens = int(attn_inputs.sequence_lengths.sum().item())
        super().__init__(
            MlaFlashInferDecodeOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.kernel_tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                attn_configs.is_sparse,
                weights,
                max_bs=max_bs,
                max_context_len=max_seq_len,
                num_tokens=num_tokens,
                is_cuda_graph=is_cuda_graph,
            ),
            NewMlaRotaryEmbeddingOp(
                cos_sin_cache=cos_sin_cache,
                is_neox_style=attn_configs.rope_config.is_neox_style,
            ),
            MlaKVCacheWriteOp(
                kv_cache_dtype=attn_configs.kv_cache_dtype,
            ),
            attn_inputs,
            attn_configs.kernel_tokens_per_block,
            attn_configs,
            weights,
            cos_sin_cache,
            fmha_config,
            use_trt_fmha=False,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
        )

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return (
            attn_configs.use_mla
            and not attn_inputs.is_prefill
            and not attn_configs.is_sparse
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.prepare(attn_inputs, forbid_realloc=True)

    def prepare_cuda_graph_group(self, attn_inputs: PyAttentionInputs) -> None:
        """Refresh one HybridCache FULL group inside graph capture.

        ``prepare_cuda_graph`` runs before replay and may call FlashInfer's
        host-side planner.  Kimi K3 additionally switches the selected FULL
        cache group between MLA layers *inside* the captured forward.  Calling
        the regular ``prepare`` there would materialize CUDA lengths and block
        tables on the host, which CUDA Graph capture forbids.

        The batch shape and sequence lengths are identical for every FULL
        group, so the plan prepared before capture/replay remains valid.  Only
        regenerate the compact page indices on device and copy them into
        FlashInfer's stable CUDA Graph indices buffer before the group runs.
        Both operations are then recorded in the graph and use the live
        per-group block table on every replay.
        """

        assert self.fmha_impl is not None
        assert self.fmha_params is not None
        self.attn_inputs = attn_inputs
        sequence_lengths_plus_1 = getattr(
            attn_inputs, "sequence_lengths_plus_1_d", None
        )
        block_table = getattr(
            attn_inputs, "kv_cache_kernel_block_id_device", None
        )
        if (
            sequence_lengths_plus_1 is None
            or sequence_lengths_plus_1.numel() == 0
        ):
            raise RuntimeError(
                "K3 CUDA Graph MLA group refresh requires "
                "sequence_lengths_plus_1_d"
            )
        if block_table is None or block_table.numel() == 0:
            raise RuntimeError(
                "K3 CUDA Graph MLA group refresh requires a device block table"
            )

        self.fmha_params.fill_decode_cuda_graph_params(
            sequence_lengths_plus_1,
            block_table,
            self.seq_size_per_block,
        )
        page_indices = self.fmha_params.page_indice_d
        graph_indices = self.fmha_impl.kv_indices_d
        if page_indices.numel() < graph_indices.numel():
            raise RuntimeError(
                "K3 CUDA Graph MLA compact page-index source is too small: "
                f"source={page_indices.numel()} target={graph_indices.numel()}"
            )
        # CudaGraphRunner rounds each request's block-table width up to a full
        # KV-cache page.  ``page_indices`` therefore has padded tail capacity,
        # while FlashInfer reserves only ceil(max_context/kernel_page) entries
        # per request.  The device preparation kernel compacts every live page
        # at the front, so copying the stable graph-buffer capacity preserves
        # all usable indices and deliberately drops only that padded tail.
        graph_indices.copy_(page_indices[: graph_indices.numel()])
