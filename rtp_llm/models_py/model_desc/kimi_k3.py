"""Framework-facing Kimi K3 model orchestration.

This module contains the ``GptModelBase`` integration, decoder-layer
composition and the top-level packed-token/cache-group loop. Operators live under
``rtp_llm.models_py.modules.kimi_k3``:

* ``kda`` / ``cache``: linear attention and its paged recurrent state;
* ``mla``: gated MLA integration over the framework attention backends;
* ``moe``: the K3 latent-expert feed-forward layer;
* shared distributed modules: reusable execution infrastructure.

Like ``KimiLinearModel``, this module composes those components but does not
own their operator implementations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather_trim,
    get_process_group,
)
from rtp_llm.models_py.distributed.sequence_parallel import (
    sequence_parallel_layout_from_attention_inputs,
    shard_physical_tokens,
)
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.base.common.embedding import Embedding
from rtp_llm.models_py.modules.base.common.kvcache_store import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalEmbeddingInjector,
)
from rtp_llm.models_py.modules.hybrid.dense_mlp import (
    DenseMLP,
    DenseMLPParallelExecutor,
)
from rtp_llm.models_py.modules.kimi_k3.all_gather_gemm import configure_all_gather_gemm
from rtp_llm.models_py.modules.kimi_k3.chunk_prefill import (
    KimiK3ChunkPublishContext,
    KimiK3ChunkRound,
    kda_materialized_block_maps,
    run_kimi_k3_chunk_prefill,
)
from rtp_llm.models_py.modules.kimi_k3.gemm_reduce_scatter import (
    configure_gemm_reduce_scatter,
)
from rtp_llm.models_py.modules.kimi_k3.kda import KDAExecutionMode, KimiK3KDA
from rtp_llm.models_py.modules.kimi_k3.kda.prefill import (
    KimiKDACurrentStateRegistry,
    KimiKDAPrefillMetadata,
    prepare_kimi_kda_prefill_metadata,
)
from rtp_llm.models_py.triton_kernels.common.activation import SituAndMul
from rtp_llm.ops import HybridAttentionType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W

if TYPE_CHECKING:
    from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig

from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MLA
from rtp_llm.models_py.modules.kimi_k3.moe import (
    KimiK3LatentMoE,
    resolve_kimi_k3_moe_strategy,
)
from rtp_llm.models_py.modules.kimi_k3.moe_se import KimiK3LatentMoESE
from rtp_llm.models_py.modules.kimi_k3.mtp import KimiK3MTPTargetMixin
from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
from rtp_llm.models_py.modules.kimi_k3.utils import (
    collective_gemm_workspace_global_tokens,
    mask_multimodal_token_ids,
    prefill_chunk_tokens,
    resolve_cu_seqlens,
)


class KimiK3FinalNorm(nn.Module):
    """Finish K3's block residual and apply the decoder output RMSNorm."""

    def __init__(
        self,
        attn_res_norm: torch.Tensor,
        attn_res_proj: torch.Tensor,
        final_norm: torch.Tensor,
        eps: float,
    ) -> None:
        super().__init__()
        self.attention_residual = KimiK3AttentionResidual(
            attn_res_norm,
            attn_res_proj,
            eps,
        )
        self.final_norm_weight = final_norm
        self.final_norm_eps = float(eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> torch.Tensor:
        return self.attention_residual(
            hidden_states,
            block_residual,
            output_norm_weight=self.final_norm_weight,
            output_norm_eps=self.final_norm_eps,
        )


class KimiK3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        hw_kernel_config: Optional[Any] = None,
        moe_strategy: str = "mega_moe",
    ) -> None:
        super().__init__()
        eps = float(config.layernorm_eps)
        block_size = int(config.k3_runtime_config.attn_res_block_size)
        self.is_kda = (
            config.hybrid_attention_config.hybrid_attention_types[layer_idx]
            == HybridAttentionType.LINEAR
        )
        self._previous_residual_blocks = (layer_idx + block_size - 1) // block_size
        self._writes_residual_block = layer_idx % block_size == 0
        self._residual_block_write_idx = (
            self._previous_residual_blocks if self._writes_residual_block else -1
        )
        self._active_residual_blocks = self._previous_residual_blocks + int(
            self._writes_residual_block
        )
        self.self_attention_residual = KimiK3AttentionResidual(
            weights[K3W.SELF_ATTN_RES_NORM],
            weights[K3W.SELF_ATTN_RES_PROJ],
            eps,
        )
        self.mlp_residual = KimiK3AttentionResidual(
            weights[K3W.MLP_RES_NORM],
            weights[K3W.MLP_RES_PROJ],
            eps,
        )
        self.attention_norm_weight = weights[W.pre_ln_gamma]
        self.mlp_norm_weight = weights[W.post_ln_gamma]
        self.norm_eps = eps
        self.self_attn: nn.Module = (
            KimiK3KDA(config, parallelism_config, weights, layer_idx)
            if self.is_kda
            else KimiK3MLA(config, parallelism_config, weights, layer_idx)
        )
        self.mlp: nn.Module
        if layer_idx in config.moe_layer_index:
            moe_class = {
                "mega_moe": KimiK3LatentMoE,
                "mega_moe_se": KimiK3LatentMoESE,
            }[moe_strategy]
            self.mlp = moe_class(config, parallelism_config, weights, layer_idx)
        else:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                config.quant_config,
                hw_kernel_config=hw_kernel_config,
                gated_activation=SituAndMul(
                    config.k3_runtime_config.activation_situ_beta,
                    config.k3_runtime_config.activation_situ_linear_beta,
                ),
                merge_gate_up=False,
                parallel_executor=DenseMLPParallelExecutor(
                    weights,
                    parallelism_config,
                    gated=True,
                ),
            )

    def prepare_kda_cache_store(self, kv_cache: LayerKVCache) -> None:
        kv_cache.cache_store_segment_sizes = list(
            self.self_attn.cache_store_segment_sizes
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        mode: KDAExecutionMode,
        kda_prefill_metadata: Optional[KimiKDAPrefillMetadata] = None,
        kda_current_state_registry: Optional[KimiKDACurrentStateRegistry] = None,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        fmha_impl: Any = None,
    ) -> torch.Tensor:
        prefix_sum: Optional[torch.Tensor] = hidden_states
        attention_input = self.self_attention_residual(
            prefix_sum,
            block_residual,
            output_norm_weight=self.attention_norm_weight,
            output_norm_eps=self.norm_eps,
            num_blocks=self._previous_residual_blocks,
            block_write_idx=self._residual_block_write_idx,
        )
        if self._writes_residual_block:
            prefix_sum = None
        active_block_residual = block_residual[:, : self._active_residual_blocks]
        if self.is_kda:
            attention_output = self.self_attn(
                attention_input,
                cu_seqlens,
                mode=mode,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                prefill_metadata=kda_prefill_metadata,
                current_state_registry=kda_current_state_registry,
            )
        else:
            attention_output = self.self_attn(
                attention_input,
                fmha_impl,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
            )
        attention_delta: Optional[torch.Tensor] = None
        if prefix_sum is None:
            prefix_sum = attention_output
        else:
            attention_delta = attention_output
        normalized_mlp_input = self.mlp_residual(
            prefix_sum,
            active_block_residual,
            output_norm_weight=self.mlp_norm_weight,
            output_norm_eps=self.norm_eps,
            delta=attention_delta,
            num_blocks=self._active_residual_blocks,
        )
        mlp_output = self.mlp(normalized_mlp_input)
        output = prefix_sum + mlp_output
        return output


class KimiK3Model(KimiK3MTPTargetMixin, GptModelBase):
    """Text decoder body consumed by RTP's Python model executor."""

    requires_sequence_parallel_padding = True

    def __init__(
        self,
        model_config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
        moe_config=None,
    ) -> None:
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embed_tokens = Embedding(
            model_config,
            parallelism_config,
            weights.get_global_weight(W.embedding),
        )
        self.embedding_weight = self.embed_tokens.weight
        attn_res_block_size = int(model_config.k3_runtime_config.attn_res_block_size)
        self.num_attn_res_blocks = (
            self.layer_num + attn_res_block_size - 1
        ) // attn_res_block_size
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        moe_strategy = resolve_kimi_k3_moe_strategy(moe_config)
        logging.info("Kimi K3 MoE strategy: %s", moe_strategy)
        self.layers = nn.ModuleList(
            [
                KimiK3DecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[layer_idx],
                    layer_idx,
                    hw_kernel_config=py_hw_kernel_config,
                    moe_strategy=moe_strategy,
                )
                for layer_idx in range(self.layer_num)
            ]
        )
        self._kda_layer_indices = tuple(
            layer_idx
            for layer_idx, layer in enumerate(self.layers)
            if layer.is_kda
        )
        first_kda = self.layers[self._kda_layer_indices[0]].self_attn
        self._kda_local_heads = int(first_kda.local_heads)
        self._kda_head_dim = int(first_kda.head_dim)
        self.norm = KimiK3FinalNorm(
            weights.get_global_weight(K3W.OUTPUT_ATTN_RES_NORM),
            weights.get_global_weight(K3W.OUTPUT_ATTN_RES_PROJ),
            weights.get_global_weight(W.final_ln_gamma),
            model_config.layernorm_eps,
        )
        self._layer_group_ids: Optional[tuple[int, ...]] = None
        self._max_generate_batch_size = int(max_generate_batch_size)
        self._prefill_chunk_tokens = prefill_chunk_tokens()
        self._is_decode_role = False
        self._initialize_mtp_state()
        self._prefill_static_attn_res_bank: Optional[torch.Tensor] = None

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        """Bind runtime resources and reserve Prefill collective workspaces."""

        super().initialize(init_resource)
        self._is_decode_role = bool(init_resource.is_decode_role)
        self._initialize_mtp_runtime(init_resource)
        tp_size = int(self.parallelism_config.get_attn_tp_size())
        if init_resource.is_decode_role:
            tokens_per_batch = max(int(self.config.gen_num_per_cycle) + 1, 1)
            max_global_tokens = (
                max(
                    self._max_generate_batch_size,
                    int(init_resource.max_decode_graph_batch_size),
                )
                * tokens_per_batch
            )
        else:
            max_global_tokens = collective_gemm_workspace_global_tokens(
                int(self.config.max_seq_len),
                int(init_resource.max_context_batch_size),
                self._prefill_chunk_tokens,
            )
        max_local_tokens = (max_global_tokens + tp_size - 1) // tp_size
        max_physical_tokens = max_local_tokens * tp_size
        collective_gemm_enabled = (
            tp_size > 1
            and self.embedding_weight.is_cuda
            and self.embedding_weight.dtype == torch.bfloat16
        )
        tp_group = get_process_group(Group.TP)
        configure_all_gather_gemm(
            tp_group,
            self.embedding_weight.device,
            enabled=collective_gemm_enabled,
            max_m=max_physical_tokens,
            k=int(self.config.hidden_size),
            dtype=self.embedding_weight.dtype,
        )
        configure_gemm_reduce_scatter(
            tp_group,
            self.embedding_weight.device,
            enabled=collective_gemm_enabled,
            max_m=max_physical_tokens,
            n=int(self.config.hidden_size),
        )
        return True

    def _ensure_prefill_static_attn_res_bank(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Return a model-owned AttnRes bank reused across Prefill chunks."""

        tp_size = int(self.parallelism_config.get_attn_tp_size())
        chunk_tokens = self._prefill_chunk_tokens
        chunk_local_rows = (
            (chunk_tokens + tp_size - 1) // tp_size if chunk_tokens > 0 else 0
        )
        required_rows = int(hidden_states.shape[0])
        capacity_rows = max(required_rows, chunk_local_rows)
        required_shape = (
            capacity_rows,
            int(self.num_attn_res_blocks),
            int(hidden_states.shape[1]),
        )
        bank = self._prefill_static_attn_res_bank
        if bank is None or bank.shape[0] < capacity_rows:
            bank = hidden_states.new_empty(required_shape)
            self._prefill_static_attn_res_bank = bank
            logging.info(
                "[K3_PREFILL_ATTN_RES_BANK] allocated shape=%s bytes=%.3fGiB",
                tuple(bank.shape),
                bank.numel() * bank.element_size() / float(1 << 30),
            )
        return bank.narrow(0, 0, required_rows)

    # ``prepare_fmha_impl`` is inherited from ``GptModelBase``: it builds the
    # framework MLA impl via ``AttnImplFactory.get_fmha_impl`` (identical to the
    # generic MoE path).  K3's MLA layers consume that impl through
    # ``KimiK3MLA`` (an ``MlaAttention`` subclass); K3's KDA layers ignore it.

    def _embed(self, input_ids: torch.Tensor, multimodal_inputs: Any) -> torch.Tensor:
        multimodal_features = multimodal_inputs.multimodal_features
        mm_features_locs = multimodal_inputs.mm_features_locs_host
        if multimodal_features:
            input_ids = mask_multimodal_token_ids(
                input_ids, multimodal_features, mm_features_locs
            )

        hidden_states = self.embed_tokens(input_ids)
        # Vision features use full hidden space, so inject after the TP all-gather
        # in Embedding rather than into its rank-local vocabulary projection.
        return self.multimodal_embedding_injector(
            hidden_states, multimodal_features, mm_features_locs
        )

    def chunk_prefill_token_budget(self) -> int:
        """Opt Kimi K3 into the generic executor chunk-Prefill protocol."""

        return self._prefill_chunk_tokens

    def forward(
        self,
        inputs: PyModelInputs,
        fmha_impl: Any = None,
        chunk_prefill_round_hook: Any = None,
    ) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        input_ids = inputs.input_ids.reshape(-1)
        chunk_tokens = self._prefill_chunk_tokens
        if (
            chunk_tokens > 0
            and attention_inputs.is_prefill
            and input_ids.numel() > chunk_tokens
        ):
            return run_kimi_k3_chunk_prefill(
                self,
                inputs,
                fmha_impl,
                chunk_tokens,
                chunk_prefill_round_hook,
            )
        return self._forward_impl_one(inputs, fmha_impl)

    def _forward_impl_one(
        self,
        inputs: PyModelInputs,
        fmha_impl: Any = None,
        *,
        kda_current_state_registry: Optional[KimiKDACurrentStateRegistry] = None,
        round_plan: Optional[KimiK3ChunkRound] = None,
        chunk_publish_context: Optional[KimiK3ChunkPublishContext] = None,
    ) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        input_ids = inputs.input_ids.reshape(-1)
        tp_size = int(self.parallelism_config.get_attn_tp_size())
        tp_rank = int(self.parallelism_config.get_attn_tp_rank())
        token_layout = sequence_parallel_layout_from_attention_inputs(
            attention_inputs,
            physical_tokens=int(input_ids.numel()),
            world_size=tp_size,
            rank=tp_rank,
        )
        cu_seqlens = resolve_cu_seqlens(attention_inputs, input_ids)

        # MTP target verification is represented as a packed multi-token
        # attention batch, so generic attention metadata may classify it as
        # prefill-shaped. KDA must nevertheless replay it through the paged
        # Decode path and update the Decode-owned recurrent state.
        mode: KDAExecutionMode = (
            "prefill" if token_layout.mode == "prefill" else "decode"
        )
        hidden_states = shard_physical_tokens(
            self._embed(input_ids, inputs.multimodal_inputs),
            token_layout,
        )
        block_residual = (
            self._ensure_prefill_static_attn_res_bank(hidden_states)
            if mode == "prefill"
            else hidden_states.new_empty(
                hidden_states.shape[0],
                self.num_attn_res_blocks,
                hidden_states.shape[1],
            )
        )
        if self._layer_group_ids is None:
            self._layer_group_ids = tuple(
                int(value)
                for value in attention_inputs.kv_cache_layer_to_group_host.tolist()
            )
        layer_group_ids = self._layer_group_ids
        kda_prefill_metadata: Optional[KimiKDAPrefillMetadata] = None
        if mode == "prefill":
            cu_host = attention_inputs.cu_seqlens_host
            lengths_host = attention_inputs.input_lengths_host
            prefixes_host = attention_inputs.prefix_lengths_host
            page_size = int(self.kv_cache.seq_size_per_block)
            materialized_maps = kda_materialized_block_maps(
                attention_inputs,
                layer_group_ids=layer_group_ids,
                kda_layer_indices=self._kda_layer_indices,
            )
            active_indices = None
            continuation_mask = None
            if round_plan is not None:
                active_indices = [
                    item.original_batch_idx for item in round_plan.slices
                ]
                continuation_mask = [
                    item.processed_length > 0 for item in round_plan.slices
                ]
            kda_prefill_metadata = prepare_kimi_kda_prefill_metadata(
                cu_host,
                lengths_host,
                prefixes_host,
                page_size=page_size,
                local_heads=self._kda_local_heads,
                head_dim=self._kda_head_dim,
                device=input_ids.device,
                active_original_batch_indices=active_indices,
                continuation_mask=continuation_mask,
                materialized_block_maps_host=materialized_maps,
            )
        write_cache_store_impl = create_write_cache_store_impl(
            attention_inputs, self.kv_cache
        )
        # As in vLLM/SGLang, the target keeps one decoder forward. Optional
        # speculative decoding only asks that forward to expose selected
        # intermediate states; ownership and publication live in the MTP mixin.
        aux_layer_ids = self._mtp_aux_layer_ids(inputs)
        aux_layer_set = frozenset(aux_layer_ids)
        aux_hidden_states: dict[int, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.layers):
            static_group_id = layer_group_ids[layer_idx]
            select_block_map_for_layer(attention_inputs, layer_idx, static_group_id)
            layer_cache = self.kv_cache.get_layer_cache(layer_idx)
            hidden_states = layer(
                hidden_states,
                block_residual,
                cu_seqlens=cu_seqlens,
                mode=mode,
                kda_prefill_metadata=kda_prefill_metadata,
                kda_current_state_registry=kda_current_state_registry,
                kv_cache=layer_cache,
                attention_inputs=attention_inputs,
                fmha_impl=fmha_impl,
            )
            if layer_idx in aux_layer_set:
                aux_hidden_states[layer_idx] = hidden_states
            if chunk_publish_context is not None:
                chunk_publish_context.publish_layer(layer_idx, layer, layer_cache)
            # Loop-level cache-store is only for KDA layers. MLA publishes
            # from its wrapper immediately after concat_and_cache_mla.
            elif layer.is_kda and write_cache_store_impl is not None:
                # The shared writer selects pinned-host length mirrors prepared
                # by PyWrappedModel.  Passing the CUDA length tensors directly
                # is unsafe because PD cache-store consumes them on a CPU
                # background thread. Its physical block table remains 3-D;
                # the C++ writer maps this layer to the KDA cache group.
                layer.prepare_kda_cache_store(layer_cache)
                write_cache_store_impl(layer_cache)
        self._publish_mtp_aux_hidden_states(
            aux_hidden_states,
            aux_layer_ids,
            is_prefill=mode == "prefill",
            token_layout=token_layout,
            attention_inputs=attention_inputs,
        )
        hidden_states = self.norm(hidden_states, block_residual)
        hidden_states = all_gather_trim(
            hidden_states,
            token_layout.logical_tokens,
            group=Group.TP,
        )
        return PyModelOutputs(hidden_states)


__all__ = [
    "KimiK3DecoderLayer",
    "KimiK3FinalNorm",
    "KimiK3Model",
]
