import logging
import math
import traceback
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    AttnImplFactory,
    CausalAttention,
    Embedding,
    FMHAImplBase,
    FusedMoeFactory,
    LinearFactory,
    RMSNorm,
    SelectTopk,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.afd_data_router import (
    AfdDataRouterAttn,
)
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class Qwen3MoeAfdMlpLayer(nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        rank: int,
        world_size: int,
    ):
        afd_config = config.gpt_init_params.ffn_disaggregate_config
        super().__init__()
        self.is_ffn_rank = afd_config.is_ffn_service()
        num_attn_rank = afd_config.attention_dp_size * afd_config.attention_tp_size

        assert self.is_ffn_rank == (rank >= num_attn_rank)

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_inter_padding_size
        self.num_experts = config.expert_num
        self.top_k = config.moe_k
        num_max_dispatch_tokens_per_rank = (
            (config.max_generate_batch_size) + config.tp_size - 1
        ) // config.tp_size

        assert self.is_ffn_rank
        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)

        assert (
            self.w1 is not None and self.w2 is not None
        ), "Weights w1 and w2 must be provided"

        self.fused_moe = FusedMoeFactory().create_fused_moe(config, weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="SiGLU",
        )


class Qwen3MoeAfdDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        rank: int,
        world_size: int,
        data_router: AfdDataRouterAttn,
        is_last_layer: bool,
        is_first_layer: bool,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = CausalAttention(config, weights)

        self.top_k = config.moe_k
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, config
        )
        self.select_topk = SelectTopk(config)

        self.num_experts = config.expert_num

        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

        self.is_last_layer = is_last_layer
        self.is_first_layer = is_first_layer

        self.data_router = data_router

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        if not self.is_first_layer:
            output = self.data_router.finalize()
            hidden_states = output + hidden_states

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        num_tokens, _ = hidden_states.shape
        router_logits = self.gate(hidden_states)

        router_logits_fp32 = router_logits.float()
        topk_weights = torch.zeros(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        topk_ids = torch.zeros(
            (num_tokens, self.top_k),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        self.select_topk.forward(router_logits_fp32, topk_ids, topk_weights)

        topk_ids = topk_ids.to(torch.int64)

        self.data_router.prepare(
            a1=hidden_states,
            a1_scale=None,
            a2_scale=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_config=None,
        )

        return residual


class Qwen3MoeAttnModel(GptModelBase):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: ModelWeights,
        rank: int,
        world_size: int,
    ):
        super().__init__(config, weights)

        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))

        self.data_router = AfdDataRouterAttn(
            config,
            use_fp8_dispatch=False,
            zero_copy=False,
            async_finish=True,
            return_recv_hook=False,
        )

        self.layers = nn.ModuleList(
            [
                Qwen3MoeAfdDecoderLayer(
                    config,
                    weights.weights[idx],
                    idx,
                    rank,
                    world_size,
                    self.data_router,
                    idx == self.layer_num - 1,
                    idx == 0,
                )
                for idx in range(self.layer_num)
            ]
        )

        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def forward_micro_batch(
        self, mirco_batch_inputs: List[PyModelInputs]
    ) -> List[PyModelOutputs]:
        hidden_states_list: List[torch.Tensor] = []
        next_hidden_states_list: List[torch.Tensor] = []
        fmha_impl_list: List[FMHAImplBase] = []

        for input in mirco_batch_inputs:
            hidden_states_list.append(self.embed_tokens(input.input_ids))
            fmha_impl_list.append(
                AttnImplFactory.get_fmha_impl(
                    self.config, self.weight, input.attention_inputs
                )
            )

        for i, layer in enumerate(self.layers):
            next_hidden_states_list = []

            for idx, micro_batch_input in enumerate(mirco_batch_inputs):
                # fmha_impl = self.get_fmha_impl(micro_batch_input.attention_inputs)
                hidden_states = hidden_states_list[idx]
                hidden_states = layer(
                    hidden_states,
                    fmha_impl_list[idx],
                    kv_cache=(
                        self.kv_cache.get_layer_cache(i) if self.kv_cache else None
                    ),
                )

                next_hidden_states_list.append(hidden_states)

            hidden_states_list = next_hidden_states_list

        outputs: List[PyModelOutputs] = []

        for idx, hidden_states in enumerate(hidden_states_list):
            output = self.data_router.finalize()
            hidden_states = output + hidden_states
            hidden_states = self.norm(hidden_states)
            outputs.append(PyModelOutputs(hidden_states))

        return outputs


class Qwen3MoeFfnModel(GptModelBase):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: ModelWeights,
        rank: int,
        world_size: int,
    ):
        super().__init__(config, weights)

        self.layers = nn.ModuleList(
            [
                Qwen3MoeAfdMlpLayer(config, weights.weights[idx], rank, world_size)
                for idx in range(self.layer_num)
            ]
        )

        self.hidden_dim = config.hidden_size
        self.topk = config.moe_k

    def forward_micro_batch(
        self, mirco_batch_inputs: List[PyModelInputs]
    ) -> List[PyModelOutputs]:
        for i, layer in enumerate(self.layers):
            for idx in range(self.micro_batch_size):
                hidden_states = torch.empty(0, self.hidden_dim)
                topk_weights = torch.empty(0, self.topk)
                topk_ids = torch.empty(0, self.topk, dtype=torch.int64)
                layer(hidden_states, topk_weights, topk_ids)

        return []


class Qwen3MoeAfdModel(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.is_ffn_model = (
            config.gpt_init_params.ffn_disaggregate_config.is_ffn_service()
        )
        world_size = config.world_size
        rank = config.gpt_init_params.parallelism_distributed_config.world_rank

        if self.is_ffn_model:
            self.model = Qwen3MoeFfnModel(config, weights, rank, world_size)
        else:
            self.model = Qwen3MoeAttnModel(config, weights, rank, world_size)

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        super().initialize(init_resource)
        self.model.initialize(init_resource)
        return True

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        raise NotImplementedError()

    def forward_micro_batch(self, inputs: List[PyModelInputs]) -> List[PyModelOutputs]:
        try:
            return self.model.forward_micro_batch(inputs)
        except Exception as e:
            print("\n" + "=" * 80)
            print("PYTHON EXCEPTION CAUGHT in forward_micro_batch")
            print("=" * 80)
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Details: {e}")

            print("\n--- Python Traceback ---")
            traceback.print_exc()
            raise
