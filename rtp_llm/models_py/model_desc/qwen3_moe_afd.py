import logging
import math
import traceback
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn

import rtp_llm.models_py.modules.moe.fused_moe as mm
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.kernels.activation import time_waster
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Attention
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.models_py.modules.moe import BatchedTritonExperts, FusedMoe
from rtp_llm.models_py.modules.moe.afd_data_router import (
    AfdDataRouterAttn,
    AfdDataRouterFfn,
    FakeExpert,
)
from rtp_llm.models_py.modules.norm import RMSNorm
from rtp_llm.ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import check_with_info

try:
    from libth_transformer.rtp_llm_ops import SelectTopkOp
except ImportError:
    logging.info("SelectTopkOp not available")


class Qwen3MoeRouter(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.top_k = config.moe_k
        self.gate = Linear(weights[W.moe_gate], None)
        self.select_topk_op = SelectTopkOp(config)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        self.select_topk_op.forward(router_logits_fp32, topk_ids, topk_weights)

        return topk_weights, topk_ids


class Qwen3MoeAfdMlpLayer(nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        rank: int,
        world_size: int,
        data_router: AfdDataRouterFfn,
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

        experts = BatchedTritonExperts(
            max_num_tokens=num_max_dispatch_tokens_per_rank * world_size,
            num_dispatchers=1,
            w1=self.w1,
            w2=self.w2,
        )

        self.fused_moe = FusedMoe(data_router, experts)

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
            global_num_experts=self.num_experts,
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
        self.self_attn = Qwen3Attention(config, weights)
        self.router = Qwen3MoeRouter(config, weights)
        # self.mlp = Qwen3MoeAfdMlpLayer(config, weights, rank, world_size, group)

        self.num_experts = config.expert_num

        self.fake_expert = FakeExpert()

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

        topk_weights, topk_ids = self.router(hidden_states)
        topk_ids = topk_ids.to(torch.int64)

        self.data_router.prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            self.num_experts,
            self.fake_expert.quant_config,
        )

        return residual


class Qwen3MoeAttnModel(GptModelBase):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: ModelWeights,
        rank: int,
        world_size: int,
        data_router: AfdDataRouterAttn,
    ):
        super().__init__(config, weights)

        self.embed_tokens = Embedding(weights.get_global_weight(W.embedding))

        self.layers = nn.ModuleList(
            [
                Qwen3MoeAfdDecoderLayer(
                    config,
                    weights.weights[idx],
                    idx,
                    rank,
                    world_size,
                    data_router,
                    idx == self.layer_num - 1,
                    idx == 0,
                )
                for idx in range(self.layer_num)
            ]
        )

        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )
        self.data_router: AfdDataRouterAttn = data_router

    def forward_micro_batch(
        self, mirco_batch_inputs: List[PyModelInputs]
    ) -> List[PyModelOutputs]:
        _ = time_waster(1024 * 1024 * 10, 50000, mirco_batch_inputs[0].input_ids.device)

        hidden_states_list: List[torch.Tensor] = []
        next_hidden_states_list: List[torch.Tensor] = []

        for input in mirco_batch_inputs:
            hidden_states_list.append(self.embed_tokens(input.input_ids))
        for i, layer in enumerate(self.layers):
            next_hidden_states_list = []

            for idx, micro_batch_input in enumerate(mirco_batch_inputs):
                fmha_impl = self.get_fmha_impl(micro_batch_input.attention_inputs)
                hidden_states = hidden_states_list[idx]
                hidden_states = layer(
                    hidden_states,
                    fmha_impl,
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
        data_router: AfdDataRouterFfn,
    ):
        super().__init__(config, weights)

        self.layers = nn.ModuleList(
            [
                Qwen3MoeAfdMlpLayer(
                    config, weights.weights[idx], rank, world_size, data_router
                )
                for idx in range(self.layer_num)
            ]
        )

        self.hidden_dim = config.hidden_size
        self.topk = config.moe_k
        self.data_router: AfdDataRouterFfn = data_router

        self.count = 0

    def forward_micro_batch(
        self, mirco_batch_inputs: List[PyModelInputs]
    ) -> List[PyModelOutputs]:
        for i, layer in enumerate(self.layers):
            for idx in range(self.micro_batch_size):
                hidden_states = torch.empty(0, self.hidden_dim)
                topk_weights = torch.empty(0, self.topk)
                topk_ids = torch.empty(0, self.topk, dtype=torch.int64)
                layer(hidden_states, topk_weights, topk_ids)

        self.count += 1
        return []


class Qwen3MoeAfdModel(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.is_ffn_model = (
            config.gpt_init_params.ffn_disaggregate_config.is_ffn_service()
        )
        import os

        os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "1"
        os.environ["ACCL_TOPO_FIX"] = "1"
        os.environ["ACCL_LOAD_BALANCE"] = "1"
        # os.environ["NCCL_TOPO_DUMP_FILE"] = "/tmp/nccl_topo.xml"
        os.environ["USE_DEEPEP_LOW_LATENCY"] = "1"
        afd_master_port = "41333"
        afd_master_addr = "127.0.0.1"

        world_size = config.world_size
        rank = config.dp_rank
        self.rank = rank

        logging.info(
            f"Initializing distributed process group - rank: {rank}, world_size: {world_size}"
        )

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{afd_master_addr}:{afd_master_port}",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )
        torch.set_default_device("cuda")
        torch.cuda.set_device(rank)
        group = dist.new_group(list(range(world_size)))
        self.group = group

        assert isinstance(group, dist.ProcessGroup)
        logging.info(f"Distributed process group setup complete - device: cuda:{rank}")

        afd_config = config.gpt_init_params.ffn_disaggregate_config
        num_attn_rank = afd_config.attention_dp_size * afd_config.attention_tp_size
        num_max_dispatch_tokens_per_rank = (
            (config.max_generate_batch_size) + config.tp_size - 1
        ) // config.tp_size

        device_properties = torch.cuda.get_device_properties(0)
        accl_num_warp_groups: int = math.ceil(
            config.expert_num
            * world_size
            / (world_size - num_attn_rank)
            / device_properties.multi_processor_count
        )  # assume tp_size = 1
        os.environ["ACCL_DISPATCH_NUM_WARP_GROUPS"] = str(accl_num_warp_groups)
        os.environ["ACCL_COMBINE_NUM_WARP_GROUPS"] = str(accl_num_warp_groups)

        if self.is_ffn_model:
            data_router = AfdDataRouterFfn(
                rank=rank,
                world_size=world_size,
                num_attn_ranks=num_attn_rank,
                num_experts=config.expert_num,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                hidden_dim=config.hidden_size,
                group=group,
            )
            self.model = Qwen3MoeFfnModel(
                config, weights, rank, world_size, data_router
            )
        else:
            data_router = AfdDataRouterAttn(
                rank=rank,
                world_size=world_size,
                num_attn_ranks=num_attn_rank,
                num_experts=config.expert_num,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                hidden_dim=config.hidden_size,
                group=group,
            )
            self.model = Qwen3MoeAttnModel(
                config, weights, rank, world_size, data_router
            )

    def destroy(self):
        if self.group is not None:
            dist.destroy_process_group(self.group)
            self.group = None

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
