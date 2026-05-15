from typing import Any

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import Embedding, LinearFactory, RMSNorm, RMSResNorm
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class GenericMoeMTPModel(GptModelBase):
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
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.pre_fc_norm_embedding = RMSNorm(
            weights.global_weights[W.multi_tokens_predict_enorm],
            eps=model_config.layernorm_eps,
        )
        self.pre_fc_norm_hidden = RMSNorm(
            weights.global_weights[W.multi_tokens_predict_hnorm],
            eps=model_config.layernorm_eps,
        )
        self.fc = LinearFactory.create_linear_from_weights(
            weights.global_weights, W.multi_tokens_predict_eh_proj
        )

        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.global_weights[W.multi_tokens_predict_final_ln_gamma],
            eps=model_config.layernorm_eps,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        last_hidden_states = inputs.input_hiddens

        e_norm = self.pre_fc_norm_embedding(inputs_embeds)
        h_norm = self.pre_fc_norm_hidden(last_hidden_states)
        cat_hidden_states = torch.cat([e_norm, h_norm], -1)
        hidden_states = self.fc(cat_hidden_states)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            hidden_states = output.hidden_states
            residual = output.residual

        hidden_states = self.norm(hidden_states, residual)

        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
