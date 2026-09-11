"""Qwen3 DSpark using the standard model, graph and paged-cache contracts."""

from collections.abc import Mapping

import torch
from torch import nn

from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    get_group_tags_for_layers,
    select_attention_inputs_for_tag,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.models_py.modules import LinearFactory, RMSNorm
from rtp_llm.models_py.modules.factory.attention.context_kv import ContextKVWriter
from rtp_llm.ops.compute_ops import PyModelOutputs
from rtp_llm.utils.model_weight import W


class Qwen3DSparkModel(Qwen3Model):
    # The draft consumes target features; it never exports decoder features.
    _captures_aux_hidden = False

    def __init__(
        self,
        config,
        parallelism_config,
        weights,
        max_generate_batch_size,
        **kwargs,
    ):
        if kwargs.get("quant_config") is not None:
            raise NotImplementedError("Qwen3 DSpark quantization is not supported")
        cp_config = parallelism_config.prefill_cp_config
        if cp_config.is_enabled() or cp_config.kv_cache_sharded:
            raise NotImplementedError("Qwen3 DSpark requires replicated prefill")
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size,
            **kwargs,
        )
        self.attn_configs = config.getAttentionConfigs(
            parallelism_config.get_attn_tp_size()
        )
        if self.attn_configs.is_causal:
            raise ValueError("Qwen3 DSpark proposal attention must be non-causal")
        self.aux_feature_dim = weights.get_global_weight(W.dspark_fc_w).shape[0]
        self.fc = LinearFactory.create_linear_from_weights(
            weights.global_weights, W.dspark_fc_w
        )
        self.hidden_norm = RMSNorm(
            weights.get_global_weight(W.dspark_hidden_norm_gamma),
            eps=config.layernorm_eps,
        )
        # Every layer projects the same committed target features. Fuse these
        # K/V GEMMs; proposal attention retains Qwen3's normal layer weights.
        q_cols = self.attn_configs.head_num * self.attn_configs.size_per_head
        layer_weights = weights.weights[: self.layer_num]
        self.context_kv_projection = LinearFactory.create_linear(
            torch.cat([w[W.attn_qkv_w][:, q_cols:] for w in layer_weights], dim=1),
            None,
            None,
            None,
            self.py_hw_kernel_config,
        )
        self.context_k_norms = nn.ModuleList(
            RMSNorm(w[W.k_ln_gamma], eps=config.layernorm_eps) for w in layer_weights
        )

    def cuda_graph_input_hidden_size(self):
        return self.aux_feature_dim

    def prepare_forward_commit(self, inputs, is_cuda_graph=False):
        attention = get_attention_inputs_value(inputs)
        if isinstance(attention, Mapping):
            return {
                tag: ContextKVWriter(
                    self.attn_configs,
                    select_attention_inputs_for_tag(attention, tag),
                    self.fmha_config,
                )
                for tag in get_group_tags_for_layers(
                    self.kv_cache, range(self.layer_num)
                )
            }
        return ContextKVWriter(self.attn_configs, attention, self.fmha_config)

    @torch.inference_mode()
    def forward_propose(self, inputs, fmha_impl=None):
        # The engine already builds [anchor, noise, ...] and selects the
        # prediction rows for lm_head. Reuse that geometry without rebuilding it.
        if self.kv_cache is None:
            # PyWrappedModel also invokes this entrypoint during warmup before
            # the paged cache is allocated, as for the main DSv4 implementation.
            return PyModelOutputs(
                self.embed_tokens.weight.new_zeros(
                    (inputs.input_ids.numel(), self.config.hidden_size)
                )
            )
        return super().forward(inputs, fmha_impl)

    @torch.inference_mode()
    def forward_commit(self, inputs, fmha_impl=None):
        if self.kv_cache is None:
            return PyModelOutputs(
                self.embed_tokens.weight.new_empty((0, self.config.hidden_size))
            )
        features = inputs.input_hiddens
        if (
            features is None
            or features.ndim != 2
            or features.shape[0] != inputs.input_ids.numel()
            or features.shape[1] != self.aux_feature_dim
        ):
            raise ValueError(
                "Qwen3 DSpark commit requires row-aligned target aux features"
            )
        hidden = self.fc(features)
        rows = hidden.shape[0]
        head_dim = self.attn_configs.size_per_head
        all_kv = self.context_kv_projection(self.hidden_norm(hidden)).view(
            rows, self.layer_num, 2, self.attn_configs.kv_head_num, head_dim
        )
        writers = (
            fmha_impl if fmha_impl is not None else self.prepare_forward_commit(inputs)
        )
        dummy_q = hidden.new_zeros((rows, self.attn_configs.head_num * head_dim))
        for layer_id, k_norm in enumerate(self.context_k_norms):
            writer = select_fmha_impl_for_layer(writers, self.kv_cache, layer_id)
            key, value = all_kv[:, layer_id].unbind(1)
            key = k_norm(key.reshape(-1, head_dim)).reshape(rows, -1)
            qkv = torch.cat((dummy_q, key, value.reshape(rows, -1)), dim=1)
            writer.forward(qkv, self.kv_cache.get_layer_cache(layer_id))
        return PyModelOutputs(hidden)

    def forward(self, inputs, fmha_impl=None):
        raise RuntimeError("Qwen3 DSpark requires forward_propose or forward_commit")
