from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3DecoderLayer
from rtp_llm.models_py.modules import AttnImplFactory, Embedding, LinearFactory, RMSNorm
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class Qwen2MtpModel(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
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
        self.eh_proj = LinearFactory.create_linear_from_weights(
            weights.weights[0], W.multi_tokens_predict_eh_proj
        )
        self.e_norm = RMSNorm(
            weights.weights[0][W.multi_tokens_predict_enorm], eps=config.layernorm_eps
        )
        self.h_norm = RMSNorm(
            weights.weights[0][W.multi_tokens_predict_hnorm], eps=config.layernorm_eps
        )

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config, parallelism_config, weights.weights[idx], quant_config
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.weights[0][W.multi_tokens_predict_final_ln_gamma],
            eps=config.layernorm_eps,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        embedding_hidden_states = inputs_embeds
        last_hidden_states = inputs.input_hiddens

        e_norm = self.e_norm(embedding_hidden_states)
        h_norm = self.h_norm(last_hidden_states)
        cat_hidden_states = torch.cat([h_norm, e_norm], -1)
        hidden_states = self.eh_proj(cat_hidden_states)

        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(
                inputs
            )  # pyright: ignore[reportUnreachable]
            fmha_impl.prepare(inputs.attention_inputs)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "Qwen2MtpModel",
]
