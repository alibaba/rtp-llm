import os
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


def decode_query_length(attn_inputs: PyAttentionInputs) -> int:
    """Return the rectangular per-request decode query width without a device sync."""
    is_target_verify = bool(getattr(attn_inputs, "is_target_verify", False))
    is_mtp_draft_update = bool(getattr(attn_inputs, "is_mtp_draft_update", False))
    if not is_target_verify and not is_mtp_draft_update:
        if getattr(attn_inputs, "is_prefill", False):
            raise RuntimeError(
                "paged MLA decode query length was requested for a prefill batch"
            )
        # Normal decode and MTP draft decode advance one token per forward.
        return 1

    input_lengths = getattr(attn_inputs, "input_lengths", None)
    batch_size = int(input_lengths.numel()) if input_lengths is not None else 0
    total_tokens = int(getattr(attn_inputs, "total_tokens", 0))
    if batch_size <= 0:
        raise RuntimeError(
            "MLA target verify requires a positive batch size, "
            f"got batch_size={batch_size}"
        )

    input_lengths_host = getattr(attn_inputs, "input_lengths_host", None)
    if input_lengths_host is not None and input_lengths_host.numel():
        if input_lengths_host.is_cuda:
            raise RuntimeError("MLA target verify input_lengths_host must be on CPU")
        values = [int(value) for value in input_lengths_host.tolist()]
        if (
            len(values) != batch_size
            or values[0] <= 0
            or any(value != values[0] for value in values[1:])
        ):
            raise RuntimeError(
                "MLA target verify requires uniform host query lengths for "
                f"batch_size={batch_size}, got {values}"
            )
        if total_tokens > 0 and values[0] * batch_size != total_tokens:
            raise RuntimeError(
                "MLA target verify host query lengths do not match the packed "
                f"query: q_len={values[0]}, batch_size={batch_size}, "
                f"total_tokens={total_tokens}"
            )
        return values[0]

    # Async target-verify preparation deliberately keeps lengths on device.
    if total_tokens <= 0 or total_tokens % batch_size:
        raise RuntimeError(
            "MLA target verify requires a positive rectangular query shape, "
            f"got total_tokens={total_tokens}, batch_size={batch_size}"
        )
    return total_tokens // batch_size


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
        warmup_flashinfer: bool = True,
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
        if warmup_flashinfer:
            warmup_flashinfer_python()
        self.seq_size_per_block = seq_size_per_block
        self.fmha_impl: Any = fmha_impl
        if self.fmha_impl is not None:
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
        prefix_lengths = getattr(attn_inputs, "prefix_lengths_host", None)
        sequence_lengths = getattr(attn_inputs, "sequence_lengths_host", None)
        input_lengths = getattr(attn_inputs, "input_lengths_host", None)
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
        if input_lengths is not None and input_lengths.numel():
            input_values = [int(value) for value in input_lengths.tolist()]
            prefix_values = (
                [int(value) for value in prefix_lengths.tolist()]
                if prefix_lengths is not None and prefix_lengths.numel()
                else [0] * len(input_values)
            )
            # Whole-model Prefill replaces the request metadata for every
            # segment.  Refresh this host-side hint together with the planner;
            # retaining the initial full-request length makes the first 64K
            # segment gather and expand the entire 1M cache capacity.
            self.fmha_impl.total_kv_lens_hint = sum(input_values) + sum(prefix_values)
        self.fmha_params.fill_params(
            prefix_lengths,
            sequence_lengths,
            input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            self.seq_size_per_block,
            forbid_realloc,
        )
        self.fmha_impl.plan(self.fmha_params)

    def _device_slot_mapping(self) -> Optional[torch.Tensor]:
        """Map direct-plan positions through the live HybridCache group."""

        assert self.fmha_params is not None
        slot_mapping = getattr(self.fmha_params, "slot_mapping", None)
        if slot_mapping is not None:
            # The legacy FlashInfer planner already produced the mapping.
            return None

        positions = getattr(self.fmha_params, "positions_d", None)
        batch_indices = getattr(self.fmha_params, "batch_indice_d", None)
        block_table = getattr(self.attn_inputs, "kv_cache_kernel_block_id_device", None)
        if (
            positions is None
            or batch_indices is None
            or block_table is None
            or positions.numel() == 0
            or batch_indices.numel() == 0
            or block_table.numel() == 0
        ):
            raise RuntimeError(
                "direct CUDA MLA cache write requires positions, batch indices, "
                "and the current HybridCache group block table"
            )
        if positions.numel() != batch_indices.numel():
            raise RuntimeError(
                "direct CUDA MLA position/batch metadata size mismatch: "
                f"positions={positions.numel()} batch={batch_indices.numel()}"
            )

        positions_i64 = positions.to(torch.int64)
        batch_indices_i64 = batch_indices.to(torch.int64)
        block_indices = torch.div(
            positions_i64,
            self.seq_size_per_block,
            rounding_mode="floor",
        )
        block_numbers = block_table[batch_indices_i64, block_indices].to(torch.int64)
        slot_mapping = block_numbers * self.seq_size_per_block + torch.remainder(
            positions_i64, self.seq_size_per_block
        )
        slot_mapping.record_stream(torch.cuda.current_stream(slot_mapping.device))
        return slot_mapping

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

        q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]

        # Apply RoPE to Q and K
        self.rope_impl.forward(q_pe, k_pe, self.rope_params)

        # Write compressed KV and position-encoded K to cache
        slot_mapping_override = (
            self._device_slot_mapping() if kv_cache is not None else None
        )
        self.kv_cache_write_op.forward(
            compressed_kv,
            k_pe,
            kv_cache,
            self.rope_params,
            slot_mapping_override=slot_mapping_override,
        )

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
        return self.fmha_impl.forward(q_nope, q_pe, kv_cache, layer_id)


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
        prefix_host = getattr(attn_inputs, "prefix_lengths_host", None)
        if prefix_host is not None and prefix_host.numel():
            self.has_reuse_cache = max(int(v) for v in prefix_host.tolist()) > 0
        elif attn_inputs.prefix_lengths is not None:
            # Compatibility fallback for older bindings.
            self.has_reuse_cache = bool(attn_inputs.prefix_lengths.max().item() > 0)

        self.absorb_opt_len = (
            fmha_config.absorb_opt_len if fmha_config is not None else 1024
        )
        input_host = getattr(attn_inputs, "input_lengths_host", None)
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
        # K3 的 Prefill 恒走 FlashMLA(见 MlaFlashMLAPrefillImpl),所以这条
        # K3 Prefill 不选择 FlashInfer 实现。
        return False

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
        slot_mapping_override = (
            self._device_slot_mapping() if kv_cache is not None else None
        )
        self.kv_cache_write_op.forward(
            compressed_kv,
            k_pe,
            kv_cache,
            self.rope_params,
            slot_mapping_override=slot_mapping_override,
        )

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        assert self.fmha_impl is not None
        return self.compute_prefill_context(q, compressed_kv, k_pe, kv_cache, layer_id)


class MlaFlashMLAPrefillImpl(MlaFlashInferPrefillImpl):
    """Dense FlashMLA variant of RTP's shared MLA Prefill pipeline."""

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
        from .flashmla_dense_prefill import MlaFlashMLAPrefillOp

        MlaFlashInferImplBase.__init__(
            self,
            MlaFlashMLAPrefillOp(
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
                kv_cache_dtype=attn_configs.kv_cache_dtype,
                prefix_chunk_tokens=attn_configs.mla_prefill_kv_chunk_tokens,
            ),
            NewMlaRotaryEmbeddingOp(
                cos_sin_cache=cos_sin_cache,
                is_neox_style=attn_configs.rope_config.is_neox_style,
            ),
            MlaKVCacheWriteOp(kv_cache_dtype=attn_configs.kv_cache_dtype),
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
            warmup_flashinfer=False,
        )
        self.has_reuse_cache = False
        self.absorb_opt_len = 0
        self.absorb_fmha = None

    def create_params(self, attn_inputs: PyAttentionInputs):
        if self.fmha_impl is not None:
            self.prepare(attn_inputs)

    def prepare(self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False):
        """Plan dense FlashMLA directly from CUDA metadata.

        A fresh parameter object is installed for every model invocation.  The
        small model's following decode may replace this wrapper, but it cannot
        mutate or recycle a pinned FlashInfer host plan because none is created.
        """

        if forbid_realloc:
            raise RuntimeError("dense FlashMLA Prefill does not support graph replay")
        assert self.fmha_impl is not None
        check_attention_inputs(attn_inputs)
        from .flashmla_dense_prefill import build_flashmla_device_params

        params = build_flashmla_device_params(attn_inputs, self.seq_size_per_block)
        self.attn_inputs = attn_inputs
        self.fmha_params = params
        self.rope_params = params
        self.fmha_impl.plan(params)
        return params

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        # Prefill 按 PD 角色固定走 FlashMLA。
        return (
            attn_configs.use_mla
            and not attn_configs.is_sparse
            and attn_inputs.is_prefill
        )


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
        query_length = decode_query_length(attn_inputs)
        is_target_verify = bool(getattr(attn_inputs, "is_target_verify", False))
        is_mtp_draft_update = bool(getattr(attn_inputs, "is_mtp_draft_update", False))
        sequence_lengths_host = getattr(attn_inputs, "sequence_lengths_host", None)
        if is_target_verify or is_mtp_draft_update:
            max_bs = int(attn_inputs.input_lengths.size(0))
            num_tokens = max_bs * query_length
        elif sequence_lengths_host is not None and sequence_lengths_host.numel() > 0:
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
        is_target_verify = bool(getattr(attn_inputs, "is_target_verify", False))
        is_mtp_draft_update = bool(getattr(attn_inputs, "is_mtp_draft_update", False))
        return (
            attn_configs.use_mla
            and (
                not attn_inputs.is_prefill
                or is_target_verify
                or is_mtp_draft_update
            )
            and (
                not attn_configs.is_sparse
                or is_target_verify
                or is_mtp_draft_update
            )
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        is_target_verify = bool(getattr(attn_inputs, "is_target_verify", False))
        is_mtp_draft_update = bool(
            getattr(attn_inputs, "is_mtp_draft_update", False)
        )
        sequence_lengths_d = getattr(
            attn_inputs, "sequence_lengths_plus_1_d", None
        )
        sequence_lengths_host = getattr(
            attn_inputs, "sequence_lengths_host", None
        )
        block_table_d = getattr(
            attn_inputs, "kv_cache_kernel_block_id_device", None
        )

        # Normal and MTP draft decode are q=1. Build their bulk metadata with
        # the existing CUDA replay kernel, while retaining only the tiny CPU
        # arrays that FlashInfer's length-dependent scheduler still requires.
        if (
            not is_target_verify
            and not is_mtp_draft_update
            and sequence_lengths_d is not None
            and sequence_lengths_d.numel() > 0
            and sequence_lengths_host is not None
            and sequence_lengths_host.numel() > 0
            and block_table_d is not None
            and block_table_d.numel() > 0
        ):
            assert self.fmha_params is not None
            self.attn_inputs = attn_inputs
            check_attention_inputs(attn_inputs)
            self.fmha_params.fill_decode_cuda_graph_params(
                sequence_lengths_d,
                block_table_d,
                self.seq_size_per_block,
            )
            self.fmha_params.fill_decode_cuda_graph_plan_host_params(
                sequence_lengths_host,
                block_table_d,
                self.seq_size_per_block,
            )
            self.fmha_impl.plan(self.fmha_params)
            return

        # Target verification and the Prefill-shaped MTP draft update have
        # q>1 and must use the generic planner.
        self.prepare(attn_inputs, forbid_realloc=True)
