import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from typing_extensions import Unpack

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group, recv, send
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.models_py.modules.attention_pure import FlashInferAttentionPure
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.models_py.modules.mlp import DenseMLP
from rtp_llm.models_py.modules.norm import RMSNorm
from rtp_llm.ops import PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import check_with_info


class BatchSplitInfo(object):
    # total_mirco_batch_size: [mirco_batch_size] total_token_num of each micro batch
    # mirco_batch_sizes_list: [mirco_batch_size, dp_num] token_num of each micro batch size]
    def __init__(
        self, total_mirco_batch_size: List[int], mirco_batch_sizes_list: List[List[int]]
    ):
        self.total_mirco_batch_size = total_mirco_batch_size
        self.mirco_batch_sizes_list = mirco_batch_sizes_list
        check_with_info(
            len(total_mirco_batch_size) == len(mirco_batch_sizes_list),
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
        self.mlp = DenseMLP(config, curent_layer_weights)

        # if last layer, then all weights are setted to None
        self.qkv_proj = None
        self.q_norm = None
        self.k_norm = None
        self.input_layernorm = None
        if self.is_last_layer:
            return

        next_layer_weights = weights.weights[layer_idx + 1]
        self.qkv_proj = Linear(
            next_layer_weights[W.attn_qkv_w],
            next_layer_weights.get(W.attn_qkv_b, None),
        )
        check_with_info(W.q_ln_gamma in next_layer_weights, "q_ln_gamma not found")
        check_with_info(W.k_ln_gamma in next_layer_weights, "k_ln_gamma not found")
        self.q_norm = RMSNorm(
            next_layer_weights[W.q_ln_gamma], eps=config.layernorm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = RMSNorm(
            next_layer_weights[W.k_ln_gamma], eps=config.layernorm_eps
        )  # thus post q_norm does not need reshape

    def forward(self, residual: torch.Tensor, hidden_states: torch.Tensor):
        hidden_states = self.o_proj(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if self.is_last_layer:
            return hidden_states

        hidden_states = self.qkv_proj(hidden_states)
        # QK Norm not implemented yet
        # hidden_states = self.q_norm(hidden_states)
        # hidden_states = self.k_norm(hidden_states)
        return hidden_states


class Qwen3GemmPreLayer(nn.Module):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(weights.get_global_weight(W.embedding))
        self.input_layernorm = RMSNorm(
            weights.weights[0][W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.qkv_proj = Linear(
            weights.weights[0][W.attn_qkv_w], weights.weights[0].get(W.attn_qkv_b, None)
        )
        self.q_norm = RMSNorm(
            weights.weights[0][W.q_ln_gamma], eps=config.layernorm_eps
        )
        self.k_norm = RMSNorm(
            weights.weights[0][W.k_ln_gamma], eps=config.layernorm_eps
        )

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.qkv_proj(hidden_states)
        hidden_states = self.q_norm(hidden_states)
        hidden_states = self.k_norm(hidden_states)
        return hidden_states


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

    def recv_micro_batch_split_info(self) -> Tuple[List[torch.Tensor], BatchSplitInfo]:
        dp_num = len(self.attn_dp_rank)
        micro_batch_split_info = torch.empty(
            [dp_num, self.micro_batch_size], device=self.device
        )
        for idx, rank in enumerate(self.attn_dp_rank):
            print(
                "bbb",
                micro_batch_split_info[idx].dtype,
                micro_batch_split_info[idx].shape,
                micro_batch_split_info[idx].device,
            )
            recv(micro_batch_split_info[idx], rank, Group.DP_AND_TP)
        mirco_batch_sizes_list: List[List[int]] = (
            micro_batch_split_info.cpu().transpose(0, 1).tolist()
        )
        total_micro_batch_sizes: List[int] = [
            sum(sizes) for sizes in mirco_batch_sizes_list
        ]
        input_ids_list: List[torch.Tensor] = [
            torch.empty([size], device=self.device) for size in total_micro_batch_sizes
        ]

        for idx in range(self.micro_batch_size):
            input_ids: torch.Tensor = input_ids_list[idx]
            offset = 0
            for idx, size in enumerate(total_micro_batch_sizes):
                dist.recv(
                    input_ids[offset, offset + size],
                    self.attn_dp_rank[idx],
                    get_total_group(),
                )
                offset += size
        return input_ids_list, BatchSplitInfo(
            total_micro_batch_sizes, mirco_batch_sizes_list
        )

    def send_to_attention(self, t: torch.Tensor, micro_batch_size_list: List[int]):
        offset = 0
        for idx, size in enumerate(micro_batch_size_list):
            send(t[offset, offset + size], self.attn_dp_rank[idx], Group.DP_AND_TP)
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
        )
        for idx, size in enumerate(mirco_batch_size_list):
            recv(t[offset, offset + size], self.attn_dp_rank[idx], Group.DP_AND_TP)
            offset += size
        return t

    def forward_micro_batch(self, inputs: List[PyModelInputs]) -> List[PyModelOutputs]:
        input_ids_list, batch_split_info = self.recv_micro_batch_split_info()
        micro_batch_inputs: List[torch.Tensor] = []
        for batch_idx, input_ids in enumerate(input_ids_list):
            hidden_states = self.pre_layer(input_ids)
            micro_batch_inputs.append(hidden_states)
            self.send_to_attention(
                hidden_states, batch_split_info.mirco_batch_sizes_list[batch_idx]
            )

        for layer in self.layers:
            for batch_idx, input in enumerate(micro_batch_inputs):
                residual = input
                attn_out = self.recv_from_attention(
                    batch_split_info.mirco_batch_sizes_list[batch_idx],
                    batch_split_info.total_mirco_batch_size[batch_idx],
                )
                out = layer(residual, attn_out)
                self.send_to_attention(
                    out, batch_split_info.mirco_batch_sizes_list[batch_idx]
                )
                # send res to attention model
        return []


class Qwen3AttnModel(DisaggregateModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.attention_layers = nn.ModuleList(
            [
                FlashInferAttentionPure(config, weights.weights[idx], idx)
                for idx in range(self.layer_num)
            ]
        )
        self.ffn_service_rank = (
            config.gpt_init_params.ffn_disaggregate_config.attention_dp_size
            * config.gpt_init_params.ffn_disaggregate_config.attention_tp_size
        )

    def send_mirco_batch_split_info(self, micro_batch_split_info: List[PyModelInputs]):
        size_list = [input.input_ids.shape[0] for input in micro_batch_split_info]
        send(
            torch.tensor(size_list, device=self.device),
            self.ffn_service_rank,
            Group.DP_AND_TP,
        )
        for input in micro_batch_split_info:
            print(
                "aaa",
                input.input_ids.dtype,
                input.input_ids.shape,
                input.input_ids.device,
            )
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
        )
        recv(t, self.ffn_service_rank, Group.DP_AND_TP)
        return t

    def send_to_ffn_service(self, out: torch.Tensor):
        send(out, self.ffn_service_rank, Group.DP_AND_TP)

    def forward_micro_batch(
        self, mirco_batch_inputs: List[PyModelInputs]
    ) -> List[PyModelOutputs]:
        self.send_mirco_batch_split_info(mirco_batch_inputs)
        for layer in self.attention_layers:
            for idx, mirco_batch_input in enumerate(mirco_batch_inputs):
                inputs = self.recv_from_ffn_service(
                    mirco_batch_input.input_ids.shape[0]
                )
                out = layer(
                    hidden_states=inputs,
                    k_cache_base=self.k_cache_base,
                    v_cache_base=self.v_cache_base,
                    attention_inputs=mirco_batch_input.attention_inputs,
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

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        raise NotImplementedError()

    def forward_mirco_batch(self, inputs: List[PyModelInputs]) -> List[PyModelOutputs]:
        return self.model.forward_micro_batch(inputs)
