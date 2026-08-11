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
import os
from dataclasses import dataclass
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
from rtp_llm.models_py.distributed.symm_mem import (
    reserve_fused_all_gather_matmul_workspace,
)
from rtp_llm.models_py.distributed.sequence_parallel import (
    TokenShardLayout,
    shard_tokens,
    shard_tokens_with_padding,
    token_shard_layout,
)
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.base import RMSNorm
from rtp_llm.models_py.modules.base.common.embedding import Embedding
from rtp_llm.models_py.modules.base.common.kvcache_store import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.modules.hybrid.dense_mlp import (
    DenseMLP,
    DenseMLPParallelExecutor,
)
from rtp_llm.models_py.modules.factory.linear.parallel import (
    should_use_fused_all_gather_matmul,
)
from rtp_llm.models_py.modules.kimi_k3.kda.state import KDAExecutionMode
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

from rtp_llm.models_py.modules.kimi_k3.kda import KimiK3KDA
from rtp_llm.models_py.modules.kimi_k3.mla import (
    KimiK3MLA,
    prepare_mla_fmha_for_group,
)
from rtp_llm.models_py.modules.kimi_k3.moe import KimiK3LatentMoE
from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
from rtp_llm.models_py.modules.kimi_k3.sequence import sequence_offsets


@dataclass(frozen=True)
class KimiK3DecoderMetadata:
    """Request-scoped execution state shared by every decoder layer."""

    cu_seqlens: torch.Tensor
    mode: KDAExecutionMode
    sequence_parallel: bool
    prefill_sp_layout: Optional[TokenShardLayout] = None


@dataclass(frozen=True)
class KimiK3DecoderOutput:
    """State carried from one K3 decoder layer to the next."""

    hidden_states: torch.Tensor
    block_residual: torch.Tensor


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
        self.final_norm = RMSNorm(final_norm, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> torch.Tensor:
        return self.attention_residual(
            hidden_states,
            block_residual,
            output_norm_weight=self.final_norm.weight,
            output_norm_eps=self.final_norm.variance_epsilon,
        )


class KimiK3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        hw_kernel_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.layer_idx = int(layer_idx)
        self.eps = float(config.layernorm_eps)
        self.attn_res_block_size = config.k3_runtime_config.attn_res_block_size
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]
        self.self_attention_residual = KimiK3AttentionResidual(
            weights[K3W.SELF_ATTN_RES_NORM],
            weights[K3W.SELF_ATTN_RES_PROJ],
            self.eps,
        )
        self.mlp_residual = KimiK3AttentionResidual(
            weights[K3W.MLP_RES_NORM],
            weights[K3W.MLP_RES_PROJ],
            self.eps,
        )
        self.attention_norm = RMSNorm(weights[W.pre_ln_gamma], self.eps)
        self.mlp_norm = RMSNorm(weights[W.post_ln_gamma], self.eps)
        self.self_attn: nn.Module = (
            KimiK3KDA(config, parallelism_config, weights, layer_idx)
            if self.is_kda
            else KimiK3MLA(config, parallelism_config, weights, layer_idx)
        )
        self.mlp: nn.Module = (
            KimiK3LatentMoE(config, parallelism_config, weights, layer_idx)
            if layer_idx in set(config.moe_layer_index)
            else DenseMLP(
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
        )

    @property
    def is_kda(self) -> bool:
        return self.layer_type == HybridAttentionType.LINEAR

    def prepare_mla_fmha_for_group(
        self,
        fmha_impl: Any,
        attention_inputs: PyAttentionInputs,
        group_id: int,
        prepared_group_id: Optional[int],
    ) -> Optional[int]:
        if self.is_kda or fmha_impl is None:
            return prepared_group_id
        return prepare_mla_fmha_for_group(
            fmha_impl,
            attention_inputs,
            group_id,
            prepared_group_id,
        )

    def prepare_kda_cache_store(self, kv_cache: LayerKVCache) -> None:
        if not self.is_kda:
            raise RuntimeError("only KDA layers publish recurrent cache segments")
        kv_cache.cache_store_segment_sizes = list(
            self.self_attn.cache_store_segment_sizes
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        *,
        attn_meta: KimiK3DecoderMetadata,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        fmha_impl: Any = None,
    ) -> KimiK3DecoderOutput:
        cu_seqlens = attn_meta.cu_seqlens
        mode = attn_meta.mode
        sequence_parallel = attn_meta.sequence_parallel
        prefill_sp_layout = attn_meta.prefill_sp_layout
        decode_sp = sequence_parallel and mode == "decode"
        logical_tokens = int(hidden_states.shape[0])
        tp_size = int(self.self_attn.parallelism_config.get_attn_tp_size())
        tp_rank = int(self.self_attn.parallelism_config.get_attn_tp_rank())
        local_valid_tokens: Optional[int] = None
        if (
            prefill_sp_layout is not None
            and prefill_sp_layout.local_valid_tokens < prefill_sp_layout.local_tokens
        ):
            local_valid_tokens = prefill_sp_layout.local_valid_tokens
        prefix_sum: Optional[torch.Tensor] = hidden_states
        expected_previous_blocks = (
            self.layer_idx + self.attn_res_block_size - 1
        ) // self.attn_res_block_size
        previous_blocks = min(expected_previous_blocks, block_residual.shape[1])
        writes_block = self.layer_idx % self.attn_res_block_size == 0
        block_write_idx = (
            previous_blocks
            if writes_block and block_residual.shape[1] > previous_blocks
            else -1
        )
        if previous_blocks > 0 or block_write_idx >= 0:
            attention_input = self.self_attention_residual(
                prefix_sum,
                block_residual,
                output_norm_weight=self.attention_norm.weight,
                output_norm_eps=self.attention_norm.variance_epsilon,
                num_blocks=previous_blocks,
                block_write_idx=block_write_idx,
            )
        else:
            attention_input = self.attention_norm(hidden_states.contiguous())
        if writes_block:
            if block_write_idx < 0:
                raise RuntimeError("K3 AttnRes residual bank has no writable block")
            prefix_sum = None
        active_blocks = previous_blocks + int(writes_block)
        active_block_residual = block_residual[:, :active_blocks]
        if self.is_kda:
            attention_output, _ = self.self_attn(
                attention_input,
                cu_seqlens,
                mode=mode,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                sequence_parallel=sequence_parallel,
                prefill_sp_layout=prefill_sp_layout,
            )
        else:
            attention_output = self.self_attn(
                attention_input,
                fmha_impl,
                kv_cache,
                attention_inputs=attention_inputs,
                sequence_parallel=sequence_parallel,
                prefill_sp_layout=prefill_sp_layout,
            )
        if decode_sp:
            if prefix_sum is not None:
                prefix_sum, local_valid_tokens = shard_tokens_with_padding(
                    prefix_sum,
                    logical_tokens,
                    tp_size,
                    tp_rank,
                )
            active_block_residual, block_valid_tokens = shard_tokens_with_padding(
                active_block_residual,
                logical_tokens,
                tp_size,
                tp_rank,
            )
            if local_valid_tokens is None:
                local_valid_tokens = block_valid_tokens
            elif local_valid_tokens != block_valid_tokens:
                raise RuntimeError(
                    "K3 Decode token-SP residual shards disagree on valid rows"
                )
        attention_delta: Optional[torch.Tensor] = None
        if prefix_sum is None:
            prefix_sum = attention_output
        elif prefix_sum.is_cuda:
            attention_delta = attention_output
        else:
            prefix_sum = prefix_sum + attention_output
        normalized_mlp_input = self.mlp_residual(
            prefix_sum,
            active_block_residual,
            output_norm_weight=self.mlp_norm.weight,
            output_norm_eps=self.mlp_norm.variance_epsilon,
            delta=attention_delta,
            num_blocks=active_blocks,
        )
        mlp_output = self.mlp(
            normalized_mlp_input,
            sequence_parallel=sequence_parallel,
            valid_token_count=local_valid_tokens,
        )
        output = prefix_sum + mlp_output
        if decode_sp:
            output = all_gather_trim(output, logical_tokens, group=Group.TP)
        return KimiK3DecoderOutput(output, block_residual)


class KimiK3Model(GptModelBase):
    """Text decoder body consumed by RTP's Python model executor."""

    def __init__(
        self,
        model_config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
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
        self.attn_res_block_size = int(
            model_config.k3_runtime_config.attn_res_block_size
        )
        self.num_attn_res_blocks = (
            self.layer_num + self.attn_res_block_size - 1
        ) // self.attn_res_block_size
        self.layers = nn.ModuleList(
            [
                KimiK3DecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[layer_idx],
                    layer_idx,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for layer_idx in range(self.layer_num)
            ]
        )
        self.norm = KimiK3FinalNorm(
            weights.get_global_weight(K3W.OUTPUT_ATTN_RES_NORM),
            weights.get_global_weight(K3W.OUTPUT_ATTN_RES_PROJ),
            weights.get_global_weight(W.final_ln_gamma),
            model_config.layernorm_eps,
        )
        self._layer_group_ids: Optional[tuple[int, ...]] = None
        self._fused_ag_gemm_workspace_ready = False
        self._mtp_hidden_buffer: Optional[torch.Tensor] = None
        self._mtp_hidden_valid_tokens = 0

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        """Bind runtime resources and reserve the largest Prefill AG workspace."""

        super().initialize(init_resource)
        if self._fused_ag_gemm_workspace_ready:
            return True

        tp_size = int(self.parallelism_config.get_attn_tp_size())
        max_global_tokens = int(self.config.max_seq_len) * int(
            init_resource.max_context_batch_size
        )
        max_local_tokens = (max_global_tokens + tp_size - 1) // tp_size
        max_physical_tokens = max_local_tokens * tp_size
        if (
            init_resource.is_decode_role
            or tp_size <= 1
            or not should_use_fused_all_gather_matmul(max_physical_tokens)
        ):
            return True
        workspace_bytes = (
            max_local_tokens
            * int(self.config.hidden_size)
            * self.embedding_weight.element_size()
        )
        reserve_fused_all_gather_matmul_workspace(
            get_process_group(Group.TP),
            workspace_bytes,
        )
        self._fused_ag_gemm_workspace_ready = True
        logging.info(
            "[K3_FUSED_AG_GEMM] reserved %.3f GiB symmetric workspace "
            "for %d global Prefill tokens (TP%d)",
            workspace_bytes / (1 << 30),
            max_global_tokens,
            tp_size,
        )
        return True

    def get_mtp_target_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        if self._mtp_hidden_buffer is None:
            return None
        rows = (
            self._mtp_hidden_valid_tokens if int(num_tokens) < 0 else int(num_tokens)
        )
        if rows < 0 or rows > self._mtp_hidden_buffer.size(0):
            raise ValueError(
                f"Kimi K3 EAGLE hidden rows {rows} exceed buffered "
                f"rows {self._mtp_hidden_buffer.size(0)}"
            )
        return self._mtp_hidden_buffer.narrow(0, 0, rows)

    # ``prepare_fmha_impl`` is inherited from ``GptModelBase``: it builds the
    # framework MLA impl via ``AttnImplFactory.get_fmha_impl`` (identical to the
    # generic MoE path).  K3's MLA layers consume that impl through
    # ``KimiK3MLA`` (an ``MlaAttention`` subclass); K3's KDA layers ignore it.

    @staticmethod
    def _cu_seqlens(
        attention_inputs: PyAttentionInputs, input_ids: torch.Tensor
    ) -> torch.Tensor:
        cu_seqlens = (
            attention_inputs.cu_seqlens
            if attention_inputs.is_prefill
            else attention_inputs.decode_cu_seqlens_d
        )
        if cu_seqlens is None or cu_seqlens.numel() == 0:
            cu_seqlens = (
                torch.tensor(
                    [0, input_ids.numel()],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
                if attention_inputs.is_prefill
                else torch.arange(
                    input_ids.numel() + 1,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
        graph_decode = not attention_inputs.is_prefill and (
            bool(getattr(attention_inputs, "is_cuda_graph", False))
            or (input_ids.is_cuda and torch.cuda.is_current_stream_capturing())
        )
        if graph_decode:
            # Decode has exactly one packed token per request.  Inspecting the
            # CUDA prefix sums on the host would make capture illegal and would
            # freeze replay metadata; shape validation is sufficient here.
            if cu_seqlens.numel() != input_ids.numel() + 1:
                raise ValueError(
                    "K3 CUDA Graph decode requires one cu_seqlens interval per token"
                )
        else:
            sequence_offsets(
                cu_seqlens,
                input_ids.numel(),
                cu_seqlens_host=getattr(attention_inputs, "cu_seqlens_host", None),
            )
        return cu_seqlens

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        if attention_inputs is None:
            raise ValueError("Kimi K3 requires PyAttentionInputs")
        if not attention_inputs.is_prefill and self.kv_cache is None:
            raise RuntimeError("Kimi K3 decode requires an initialized hybrid cache")
        input_ids = inputs.input_ids.reshape(-1)
        tp_size = int(self.parallelism_config.get_attn_tp_size())
        # SP MoE 是 K3 modeling 唯一的流程,不再由开关决定 —— Decode TP8/EP8
        # 不走 SP 就在启动时 die,Prefill 侧生产配置同样一直是 SP。
        tp_rank = int(self.parallelism_config.get_attn_tp_rank())
        sp_requested = tp_size > 1
        is_target_verify = bool(
            getattr(attention_inputs, "is_target_verify", False)
        )
        # The engine represents the multi-token target verification pass with
        # Prefill-shaped metadata, but the verify kernels replay every draft
        # position on every TP rank.  It must therefore stay replicated and
        # must not enter either token-SP path.
        prefill_sp = (
            sp_requested and attention_inputs.is_prefill and not is_target_verify
        )
        prefill_sp_layout = (
            token_shard_layout(int(input_ids.numel()), tp_size, tp_rank)
            if prefill_sp
            else None
        )
        # Target verify replays multiple speculative positions on every TP
        # rank and its KDA projection performs an ordinary TP all-reduce.  Its
        # token rows are therefore replicated, unlike normal single-token
        # Decode.  Applying Decode token-SP here shards only the residual side
        # and produces incompatible full-token/sharded-token shapes.
        decode_sp = (
            sp_requested
            and not attention_inputs.is_prefill
            and not is_target_verify
        )
        sp_active = prefill_sp or decode_sp
        if not attention_inputs.is_prefill and not getattr(
            self, "_decode_sp_startup_logged", False
        ):
            logging.info(
                "[K3_DECODE_SP] rank=%d requested=%s active=%s "
                "tokens=%d tp=%d ep=%d",
                int(self.parallelism_config.get_attn_tp_rank()),
                sp_requested,
                decode_sp,
                input_ids.numel(),
                tp_size,
                int(self.parallelism_config.ep_size),
            )
            self._decode_sp_startup_logged = True
        cu_seqlens = self._cu_seqlens(attention_inputs, input_ids)
        if sp_active:
            ep_size = int(self.parallelism_config.ep_size)
            if ep_size != tp_size:
                raise RuntimeError(
                    "Kimi K3 Sequence Parallel currently requires TP == EP; "
                    f"got TP={tp_size}, EP={ep_size}"
                )
        hidden_states = self.embed_tokens(input_ids)
        if prefill_sp:
            assert prefill_sp_layout is not None
            hidden_states = shard_tokens(
                hidden_states,
                prefill_sp_layout,
            )
        block_residual = hidden_states.new_empty(
            hidden_states.shape[0],
            self.num_attn_res_blocks,
            hidden_states.shape[1],
        )
        mode: KDAExecutionMode = "prefill" if attention_inputs.is_prefill else "decode"
        attn_meta = KimiK3DecoderMetadata(
            cu_seqlens=cu_seqlens,
            mode=mode,
            sequence_parallel=sp_active,
            prefill_sp_layout=prefill_sp_layout,
        )
        write_cache_store_impl = create_write_cache_store_impl(
            attention_inputs, self.kv_cache
        )
        prepared_mla_group_id: Optional[int] = None
        if self._layer_group_ids is None:
            layer_map_host = getattr(
                attention_inputs, "kv_cache_layer_to_group_host", None
            )
            if layer_map_host is not None and layer_map_host.numel():
                self._layer_group_ids = tuple(
                    int(value) for value in layer_map_host.tolist()
                )
        eagle3_hidden_states = []
        eagle3_enabled = os.environ.get("SP_TYPE", "").lower() == "eagle3"
        if eagle3_enabled:
            raw_aux_layers = os.environ.get("KIMI_K3_EAGLE3_AUX_LAYER_IDS")
            if raw_aux_layers:
                aux_layers = [int(value) for value in raw_aux_layers.split(",")]
            else:
                aux_layers = [0, max(0, self.layer_num // 2), self.layer_num - 1]
            if len(aux_layers) != 3 or any(
                layer_id < 0 or layer_id >= self.layer_num for layer_id in aux_layers
            ):
                raise ValueError(
                    "KIMI_K3_EAGLE3_AUX_LAYER_IDS must contain three valid "
                    f"zero-based layer ids for {self.layer_num} target layers"
                )
            aux_layer_set = set(aux_layers)
        else:
            aux_layers = []
            aux_layer_set = set()
        for layer_idx, layer in enumerate(self.layers):
            static_group_id = (
                self._layer_group_ids[layer_idx]
                if self._layer_group_ids is not None
                and layer_idx < len(self._layer_group_ids)
                else None
            )
            selected_group_id = select_block_map_for_layer(
                attention_inputs, layer_idx, static_group_id
            )
            if selected_group_id is None:
                selected_group_id = 0
            layer_cache = (
                self.kv_cache.get_layer_cache(layer_idx)
                if self.kv_cache is not None
                else None
            )
            if not layer.is_kda and fmha_impl is not None:
                prepared_mla_group_id = layer.prepare_mla_fmha_for_group(
                    fmha_impl,
                    attention_inputs,
                    selected_group_id,
                    prepared_mla_group_id,
                )
            layer_output = layer(
                hidden_states,
                block_residual,
                attn_meta=attn_meta,
                kv_cache=layer_cache,
                attention_inputs=attention_inputs,
                fmha_impl=fmha_impl,
            )
            hidden_states = layer_output.hidden_states
            block_residual = layer_output.block_residual
            if layer_idx in aux_layer_set:
                eagle3_hidden_states.append((layer_idx, hidden_states))
            # Loop-level cache-store is only for KDA layers. MLA publishes
            # from its wrapper immediately after concat_and_cache_mla.
            if (
                layer.is_kda
                and write_cache_store_impl is not None
                and layer_cache is not None
            ):
                # The shared writer selects pinned-host length mirrors prepared
                # by PyWrappedModel.  Passing the CUDA length tensors directly
                # is unsafe because PD cache-store consumes them on a CPU
                # background thread. Its physical block table remains 3-D;
                # the C++ writer maps this layer to the KDA cache group.
                layer.prepare_kda_cache_store(layer_cache)
                write_cache_store_impl(layer_cache)
        if eagle3_enabled:
            by_layer = dict(eagle3_hidden_states)
            mtp_hidden_buffer = torch.cat(
                [by_layer[layer_id] for layer_id in aux_layers], dim=-1
            ).contiguous()
            if prefill_sp:
                assert prefill_sp_layout is not None
                # Auxiliary hidden states are captured while Prefill token
                # sequence parallelism is active.  Eagle3 consumes them next
                # to the replicated full-prompt embedding, so restore the
                # framework's global token layout just like final_hidden below.
                mtp_hidden_buffer = all_gather_trim(
                    mtp_hidden_buffer,
                    prefill_sp_layout.logical_tokens,
                    group=Group.TP,
                )
            self._mtp_hidden_buffer = mtp_hidden_buffer
            self._mtp_hidden_valid_tokens = self._mtp_hidden_buffer.size(0)
        hidden_states = self.norm(hidden_states, block_residual)
        if prefill_sp:
            assert prefill_sp_layout is not None
            hidden_states = all_gather_trim(
                hidden_states,
                prefill_sp_layout.logical_tokens,
                group=Group.TP,
            )
        fmha_params = getattr(fmha_impl, "fmha_params", None)
        return (
            PyModelOutputs(hidden_states, fmha_params)
            if fmha_params is not None
            else PyModelOutputs(hidden_states)
        )

__all__ = [
    "KimiK3DecoderLayer",
    "KimiK3DecoderMetadata",
    "KimiK3DecoderOutput",
    "KimiK3FinalNorm",
    "KimiK3Model",
]
