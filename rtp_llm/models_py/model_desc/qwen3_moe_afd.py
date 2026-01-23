import logging
import math
import traceback
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn

from rtp_llm.config.model_config import ModelConfig
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
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.afd_data_router import (
    AfdDataRouterAttn,
)
from rtp_llm.ops import (
    FfnDisAggregateConfig,
    HWKernelConfig,
    MoeConfig,
    ParallelismConfig,
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
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()

        rank = parallelism_config.world_rank
        world_size = parallelism_config.world_size
        afd_config = parallelism_config.ffn_disaggregate_config

        self.is_ffn_rank = afd_config.is_ffn_service()
        num_attn_rank = afd_config.attention_dp_size * afd_config.attention_tp_size
        assert self.is_ffn_rank == (rank >= num_attn_rank)

        self.hidden_dim = config.hidden_size
        self.num_experts = config.expert_num
        self.top_k = config.moe_k

        assert self.is_ffn_rank
        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)

        assert (
            self.w1 is not None and self.w2 is not None
        ), "Weights w1 and w2 must be provided"

        config_adapter = MoEConfigAdapter(
            model_config=config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            max_generate_batch_size=max_generate_batch_size,
            quant_config=config.quant_config,
            enable_cuda_graph=enable_cuda_graph,
        )
        self.fused_moe = FusedMoeFactory().create_fused_moe(config_adapter, weights)

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
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        data_router: AfdDataRouterAttn,
        is_last_layer: bool,
        is_first_layer: bool,
        moe_config: MoeConfig,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        attn_configs = config.getAttentionConfigs(parallelism_config.tp_size)
        self.self_attn = CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            config.quant_config,
            hw_kernel_config,
        )

        self.top_k = config.moe_k
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, config
        )
        self.select_topk = SelectTopk(
            config, moe_config.fake_balance_expert, parallelism_config.dp_rank
        )

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
        )

        return residual


class Qwen3MoeAttnModel(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )

        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        config_adapter = MoEConfigAdapter(
            model_config=config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            max_generate_batch_size=max_generate_batch_size,
            quant_config=config.quant_config,
            enable_cuda_graph=enable_cuda_graph,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        self.data_router = AfdDataRouterAttn(
            config_adapter,
            quant_config=quant_config,
        )

        self.layers = nn.ModuleList(
            [
                Qwen3MoeAfdDecoderLayer(
                    config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    self.data_router,
                    idx == self.layer_num - 1,
                    idx == 0,
                    moe_config=moe_config,
                    hw_kernel_config=py_hw_kernel_config,
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
                    self.config,
                    self.parallelism_config,
                    self.weight,
                    input.attention_inputs,
                    self.fmha_config,
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
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )

        self.layers = nn.ModuleList(
            [
                Qwen3MoeAfdMlpLayer(
                    config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
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
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        is_ffn_model = parallelism_config.ffn_disaggregate_config.is_ffn_service()

        if is_ffn_model:
            self.model = Qwen3MoeFfnModel(
                model_config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size=max_generate_batch_size,
                fmha_config=fmha_config,
                py_hw_kernel_config=py_hw_kernel_config,
                device_resource_config=device_resource_config,
            )
        else:
            self.model = Qwen3MoeAttnModel(
                model_config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size=max_generate_batch_size,
                fmha_config=fmha_config,
                py_hw_kernel_config=py_hw_kernel_config,
                device_resource_config=device_resource_config,
            )

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
