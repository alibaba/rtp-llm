from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextMetadata,
)
from rtp_llm.models_py.modules import (
    AttnImplFactory,
    Embedding,
    LinearFactory,
    RMSNorm,
    RMSResNorm,
)
from rtp_llm.ops import HybridAttentionType, ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class Qwen3NextMTPModel(GptModelBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config,
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
            weights.global_weights,
            W.multi_tokens_predict_eh_proj,
            hw_kernel_config=py_hw_kernel_config,
        )
        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                Qwen3NextDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        if self.kv_cache is None:
            return None
        tags: list[str] = []
        seen: set[str] = set()
        for layer in self.layers:
            if layer.layer_type == HybridAttentionType.LINEAR:
                continue
            binding = getattr(layer, "cache_binding", None)
            if binding is None:
                continue
            for tag in binding.group_tags:
                if tag not in seen:
                    tags.append(tag)
                    seen.add(tag)
        return tags

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
        for i, decoder_layer in enumerate(self.layers):
            binding = getattr(decoder_layer, "cache_binding", None)
            layer_attention_inputs = (
                binding.select_attention_inputs(inputs.attention_inputs)
                if binding is not None
                else inputs.attention_inputs
            )
            layer_fmha_impl = (
                None
                if decoder_layer.layer_type == HybridAttentionType.LINEAR
                else (
                    binding.select_fmha_impl(fmha_impl)
                    if binding is not None
                    else fmha_impl
                )
            )
            hidden_states, residual = decoder_layer(
                hidden_states,
                residual,
                layer_fmha_impl,
                kv_cache=binding.sole_cache() if binding is not None else None,
                attention_inputs=layer_attention_inputs,
                attn_meta=Qwen3NextMetadata(),
            )
        hidden_states, residual = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states)
