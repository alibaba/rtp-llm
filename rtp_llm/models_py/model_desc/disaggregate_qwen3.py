from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group, recv, send
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.attention_pure import CausalAttentionPure
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.models_py.modules.mlp import FusedSiluActDenseMLP
from rtp_llm.models_py.modules.norm import FusedQKRMSNorm, RMSNorm
from rtp_llm.ops.compute_ops import PyModelInitResources, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import check_with_info


@dataclass
class BatchSplitInfo:
    total_mirco_batch_size: List[int]
    mirco_batch_sizes_list: List[List[int]]

    def __post_init__(self):
        check_with_info(
            len(self.total_mirco_batch_size) == len(self.mirco_batch_sizes_list),
            "total_mirco_batch_size must be equal to len of mirco_batch_sizes_list",
        )


class DisaggregateModelBase(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        check_with_info(
            self.config.gpt_init_params.ffn_disaggregate_config.attention_tp_size == 1,
            "attention_tp_size must be 1",
        )
        check_with_info(
            self.config.gpt_init_params.ffn_disaggregate_config.ffn_tp_size == 1,
            "ffn_tp_size must be 1",
        )
        check_with_info(
            self.config.gpt_init_params.ffn_disaggregate_config.ffn_dp_size == 1,
            "ffn_dp_size must be 1",
        )
        self.attn_dp_rank: List[int] = [
            i
            for i in range(
                self.config.gpt_init_params.ffn_disaggregate_config.attention_dp_size
            )
        ]
        self.attn_world_size = (
            self.config.gpt_init_params.ffn_disaggregate_config.attention_dp_size
        )
        self.device = g_parallel_info.device


class Qwen3GemmLayer(nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: ModelWeights,
        layer_idx: int,
        is_last_layer: bool,
    ):
        super().__init__()
        self.config = config
        self.weights = weights
        self.layer_idx = layer_idx
        self.is_last_layer = is_last_layer

        curent_layer_weights = weights.weights[layer_idx]
        next_layer_weights = (
            None
            if layer_idx == len(weights.weights) - 1
            else weights.weights[layer_idx + 1]
        )
        self.o_proj = Linear(
            curent_layer_weights[W.attn_o_w], curent_layer_weights.get(W.attn_o_b, None)
        )
        self.post_attention_layernorm = RMSNorm(
            curent_layer_weights[W.post_ln_gamma], eps=config.layernorm_eps
        )
        self.mlp = FusedSiluActDenseMLP(config, curent_layer_weights)

        # if last layer, then all weights are setted to None
        self.qkv_proj = None
        self.q_norm = None
        self.k_norm = None
        self.input_layernorm = None
        if self.is_last_layer:
            return

        next_layer_weights = weights.weights[layer_idx + 1]
        self.input_layernorm = RMSNorm(
            next_layer_weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.qkv_proj = Linear(
            next_layer_weights[W.attn_qkv_w],
            next_layer_weights.get(W.attn_qkv_b, None),
        )
        check_with_info(W.q_ln_gamma in next_layer_weights, "q_ln_gamma not found")
        check_with_info(W.k_ln_gamma in next_layer_weights, "k_ln_gamma not found")

        self.qk_fuse_norm = FusedQKRMSNorm(
            next_layer_weights[W.q_ln_gamma],
            next_layer_weights[W.k_ln_gamma],
            config.head_num,
            config.head_num_kv,
            config.size_per_head,
            config.layernorm_eps,
        )

    def forward(
        self, residual: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.o_proj(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if self.is_last_layer:
            return hidden_states, hidden_states

        next_hidden_states = self.input_layernorm(hidden_states)
        next_attn_input = self.qkv_proj(next_hidden_states)
        if hasattr(self, "qk_fuse_norm"):
            next_attn_input = self.qk_fuse_norm(next_attn_input)
        return next_attn_input, hidden_states


class Qwen3GemmPreLayer(nn.Module):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))
        self.input_layernorm = RMSNorm(
            weights.weights[0][W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.qkv_proj = Linear(
            weights.weights[0][W.attn_qkv_w], weights.weights[0].get(W.attn_qkv_b, None)
        )
        self.qk_fuse_norm = FusedQKRMSNorm(
            weights.weights[0][W.q_ln_gamma],
            weights.weights[0][W.k_ln_gamma],
            config.head_num,
            config.head_num_kv,
            config.size_per_head,
            config.layernorm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.qkv_proj(hidden_states)
        if hasattr(self, "qk_fuse_norm"):
            hidden_states = self.qk_fuse_norm(hidden_states)
        return hidden_states, residual


class Qwen3GemmModel(DisaggregateModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.layers = nn.ModuleList(
            [
                Qwen3GemmLayer(config, weights, idx, idx == self.layer_num - 1)
                for idx in range(self.layer_num)
            ]
        )
        self.pre_layer = Qwen3GemmPreLayer(config, weights)
        self.dp_rank = [
            self.config.gpt_init_params.ffn_disaggregate_config.attention_tp_size * i
            for i in range(
                self.config.gpt_init_params.ffn_disaggregate_config.attention_dp_size
            )
        ]

        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )
        self.lm_head = Linear(weights.get_global_weight(W.lm_head))

    def recv_micro_batch_split_info(self) -> Tuple[List[torch.Tensor], BatchSplitInfo]:
        dp_num = len(self.attn_dp_rank)
        micro_batch_split_info = torch.empty(
            [dp_num, self.micro_batch_size], dtype=torch.int64, device=self.device
        )
        for idx, rank in enumerate(self.attn_dp_rank):
            recv(micro_batch_split_info[idx], rank, Group.DP_AND_TP)
        mirco_batch_sizes_list: List[List[int]] = (
            micro_batch_split_info.cpu().transpose(0, 1).tolist()
        )
        total_micro_batch_sizes: List[int] = [
            sum(sizes) for sizes in mirco_batch_sizes_list
        ]
        input_ids_list: List[torch.Tensor] = [
            torch.empty((size,), device=self.device, dtype=torch.int32)
            for size in total_micro_batch_sizes
        ]

        for idx in range(self.micro_batch_size):
            input_ids: torch.Tensor = input_ids_list[idx]
            offset = 0
            for rank_idx, rank in enumerate(self.attn_dp_rank):
                size = mirco_batch_sizes_list[idx][rank_idx]
                if size > 0:
                    recv(input_ids[offset : offset + size], rank, Group.DP_AND_TP)
                    offset += size
        return input_ids_list, BatchSplitInfo(
            total_micro_batch_sizes, mirco_batch_sizes_list
        )

    def send_to_attention(self, t: torch.Tensor, micro_batch_size_list: List[int]):
        offset = 0
        for idx, size in enumerate(micro_batch_size_list):
            tensor_slice = t[offset : offset + size]
            send(tensor_slice, self.attn_dp_rank[idx], Group.DP_AND_TP)
            offset += size

    def recv_from_attention(
        self, mirco_batch_size_list: List[int], total_token_num: int
    ):
        offset = 0
        # TODO: tp size is not considered here
        t = torch.empty(
            [
                total_token_num,
                self.config.gpt_init_params.head_num
                * self.config.gpt_init_params.size_per_head,
            ],
            device=self.device,
            dtype=torch.half,
        )

        for idx, size in enumerate(mirco_batch_size_list):
            recv(t[offset : offset + size], self.attn_dp_rank[idx], Group.DP_AND_TP)
            offset += size
        return t

    def forward_micro_batch(self, inputs: List[PyModelInputs]) -> List[PyModelOutputs]:
        input_ids_list, batch_split_info = self.recv_micro_batch_split_info()
        micro_batch_inputs: List[torch.Tensor] = []
        residuals: List[torch.Tensor] = []
        for batch_idx, input_ids in enumerate(input_ids_list):
            hidden_states, residual = self.pre_layer(input_ids)
            residuals.append(residual)
            micro_batch_inputs.append(hidden_states)
            self.send_to_attention(
                hidden_states, batch_split_info.mirco_batch_sizes_list[batch_idx]
            )

        for layer in self.layers:
            next_residuals = []
            for batch_idx, input in enumerate(micro_batch_inputs):
                residual = residuals[batch_idx]
                attn_out = self.recv_from_attention(
                    batch_split_info.mirco_batch_sizes_list[batch_idx],
                    batch_split_info.total_mirco_batch_size[batch_idx],
                )
                out, next_residual = layer(residual, attn_out)
                next_residuals.append(next_residual)

                self.send_to_attention(
                    out, batch_split_info.mirco_batch_sizes_list[batch_idx]
                )
                # send res to attention model
            residuals = next_residuals
        return []


class Qwen3AttnModel(DisaggregateModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.attention_layers = nn.ModuleList(
            [
                CausalAttentionPure(config, weights.weights[idx])
                for idx in range(self.layer_num)
            ]
        )
        self.ffn_service_rank = (
            config.gpt_init_params.ffn_disaggregate_config.attention_dp_size
            * config.gpt_init_params.ffn_disaggregate_config.attention_tp_size
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def send_mirco_batch_split_info(self, micro_batch_split_info: List[PyModelInputs]):
        size_list = [input.input_ids.shape[0] for input in micro_batch_split_info]
        tensor_to_send = torch.tensor(size_list, device=self.device)
        send(
            tensor_to_send,
            self.ffn_service_rank,
            Group.DP_AND_TP,
        )
        for input in micro_batch_split_info:
            send(input.input_ids, self.ffn_service_rank, Group.DP_AND_TP)

    def recv_from_ffn_service(self, token_num: int) -> torch.Tensor:
        t = torch.empty(
            [
                token_num,
                (
                    self.config.gpt_init_params.head_num
                    + self.config.gpt_init_params.head_num_kv * 2
                )
                * self.config.gpt_init_params.size_per_head,
            ],
            device=self.device,
            dtype=torch.half,
        )
        recv(t, self.ffn_service_rank, Group.DP_AND_TP)
        return t

    def recv_final_from_ffn_service(self, token_num: int) -> torch.Tensor:
        t = torch.empty(
            [
                token_num,
                self.config.gpt_init_params.hidden_size,
            ],
            device=self.device,
            dtype=torch.half,
        )
        recv(t, self.ffn_service_rank, Group.DP_AND_TP)
        return t

    def send_to_ffn_service(self, out: torch.Tensor):
        send(out, self.ffn_service_rank, Group.DP_AND_TP)

    def forward_micro_batch(
        self, mirco_batch_inputs: List[PyModelInputs]
    ) -> List[PyModelOutputs]:
        self.send_mirco_batch_split_info(mirco_batch_inputs)
        for i, layer in enumerate(self.attention_layers[: self.layer_num]):
            for idx, mirco_batch_input in enumerate(mirco_batch_inputs):
                inputs = self.recv_from_ffn_service(
                    mirco_batch_input.input_ids.shape[0]
                )
                fmha_impl = self.get_fmha_impl(mirco_batch_input.attention_inputs)
                out = layer(
                    hidden_states=inputs,
                    fmha_impl=fmha_impl,
                    kv_cache=(
                        self.kv_cache.get_layer_cache(i)
                        if self.kv_cache is not None
                        else None
                    ),
                )
                self.send_to_ffn_service(out)
        outputs: List[PyModelOutputs] = []
        for idx, mirco_batch_input in enumerate(mirco_batch_inputs):
            out = self.recv_final_from_ffn_service(mirco_batch_input.input_ids.shape[0])
            outputs.append(PyModelOutputs(out))
        return outputs


class Qwen3DisaggregateModel(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.is_ffn_model = (
            config.gpt_init_params.ffn_disaggregate_config.is_ffn_service()
        )
        if self.is_ffn_model:
            self.model = Qwen3GemmModel(config, weights)
        else:
            self.model = Qwen3AttnModel(config, weights)

        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )
        self.lm_head = Linear(weights.get_global_weight(W.lm_head))

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        super().initialize(init_resource)
        self.model.initialize(init_resource)
        return True

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        raise NotImplementedError()

    def forward_micro_batch(self, inputs: List[PyModelInputs]) -> List[PyModelOutputs]:
        return self.model.forward_micro_batch(inputs)
