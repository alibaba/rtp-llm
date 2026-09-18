from dataclasses import replace
from typing import Any, Dict, Optional

import torch
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    DenseMLP,
    Embedding,
    LinearFactory,
    MlaAttention,
    RMSNorm,
)
from rtp_llm.models_py.modules.kimi_k3.input_preparation import prepare_draft_round
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W
from torch import nn


class _GatedEagle3MLA(MlaAttention):
    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        q_replicated = bool(parallelism_config.decode_cp_q_replicated)
        super().__init__(
            config.attn_config,
            parallelism_config,
            weights,
            layer_idx=0,
            layernorm_eps=config.layernorm_eps,
            quant_config=config.quant_config,
            replicate_query_heads=q_replicated,
        )

    def _project_qkv_a_input(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        fused_qkv_gate = self.fused_qkv_a_proj(hidden_states)
        qkv_width = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        return fused_qkv_gate[..., :qkv_width], fused_qkv_gate[..., qkv_width:]

    def _apply_output_gate(
        self, attn_output: torch.Tensor, output_gate: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if output_gate is None:
            raise RuntimeError("Kimi K3 Eagle3 MLA requires an output gate")
        return attn_output * torch.sigmoid(output_gate.reshape_as(attn_output))


class _KimiK3Eagle3Layer(nn.Module):
    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.embedding_norm = RMSNorm(
            weights[W.eagle3_input_norm_gamma], eps=config.layernorm_eps
        )
        self.hidden_norm = RMSNorm(
            weights[W.eagle3_fc_norm_gamma], eps=config.layernorm_eps
        )
        self.attention = _GatedEagle3MLA(config, parallelism_config, weights)
        self.post_attention_norm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            config.quant_config,
        )

    def forward(self, embedding, hidden_states, fmha_impl, kv_cache):
        attention_input = torch.cat(
            (self.embedding_norm(embedding), self.hidden_norm(hidden_states)), dim=-1
        )
        hidden_states = hidden_states + self.attention(
            attention_input, fmha_impl, kv_cache
        )
        return hidden_states + self.mlp(self.post_attention_norm(hidden_states))


class KimiK3Eagle3Model(GptModelBase):
    """Runtime graph matching ``KimiK3MLASWAEagle3`` from the training code."""

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
        self.embedding = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.aux_projection = LinearFactory.create_linear_from_weights(
            weights.weights[0], W.eagle3_fc_proj
        )
        self.embedding_dtype = weights.get_global_weight(W.embedding).dtype
        self._draft_layer_cache = None
        self._draft_cache_group = None
        self.weight._k3_kv_b_projections = {
            0: LinearFactory.create_linear_from_weights(
                weights.weights[0],
                W.mla_kv_b_w,
                W.mla_kv_b_s,
                None,
                model_config.quant_config,
            )
        }
        self.hidden_size = model_config.hidden_size
        self.layer = _KimiK3Eagle3Layer(
            model_config, parallelism_config, weights.weights[0]
        )
        self.final_norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def initialize(self, init_resource):
        result = super().initialize(init_resource)
        self._draft_layer_cache = (
            self.kv_cache.get_layer_cache(0) if self.kv_cache else None
        )
        group = int(self._draft_layer_cache.group_id) if self._draft_layer_cache else -1
        self._draft_cache_group = group if group >= 0 else None
        return result

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        prepared = prepare_draft_round(self, inputs, None)
        impl = super().prepare_fmha_impl(inputs, is_cuda_graph)
        if is_cuda_graph:
            impl._k3_draft_prepared = replace(
                prepared, fmha_impl=None
            )
        return impl

    def _embed_prepared(self, prepared):
        embedding = self.embedding(prepared.embedding_ids)
        for location, feature in prepared.embedding_injections:
            embedding[location : location + feature.shape[0]].copy_(feature)
        return embedding

    def _embed_shifted_multimodal(self, inputs):
        return self._embed_prepared(prepare_draft_round(self, inputs, None))

    def forward(self, inputs: PyModelInputs, fmha_impl: Optional[Any] = None):
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        prepared = getattr(fmha_impl, "_k3_draft_prepared", None)
        if prepared is None:
            prepared = prepare_draft_round(self, inputs, fmha_impl)
        else:
            prepared = replace(prepared, fmha_impl=fmha_impl)
        return self.forward_prepared(prepared)

    def forward_prepared(self, prepared):
        self._recurrent = None
        inputs, fmha_impl = prepared.inputs, prepared.fmha_impl
        if inputs.input_hiddens is None:
            raise ValueError("Kimi K3 EAGLE-3 requires merged auxiliary hidden states")
        embedding = self._embed_prepared(prepared)
        hidden_width = inputs.input_hiddens.shape[-1]
        if hidden_width == self.hidden_size * 3:
            # The teacher/target pass supplies the three selected target-layer
            # states concatenated along the hidden dimension.  This is the
            # only step that goes through EAGLE-3's fc projection.
            hidden_states = self.aux_projection(inputs.input_hiddens)
        elif hidden_width == self.hidden_size:
            # Autoregressive draft steps consume the previous draft layer's
            # pre-norm hidden directly, matching KimiK3MLASWAEagle3.decode_step.
            hidden_states = inputs.input_hiddens
        else:
            raise ValueError(
                "Kimi K3 EAGLE-3 expected either three concatenated target "
                f"hidden states ({self.hidden_size * 3}) or one recurrent draft "
                f"hidden state ({self.hidden_size}), got {hidden_width}"
            )
        hidden_states = self.layer(
            embedding,
            hidden_states,
            fmha_impl,
            prepared.layer_cache,
        )
        return PyModelOutputs(self.final_norm(hidden_states), fmha_impl.fmha_params)


__all__ = ["KimiK3Eagle3Model"]
