from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules import (
    AddBiasResLayerNorm,
    AttnImplFactory,
    CausalAttention,
    DenseMLP,
    EmbeddingBert,
    FMHAImplBase,
    FusedQKRMSNorm,
    LayerNorm,
)
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


def pad_flat_hidden_states(
    hidden_states: torch.Tensor, input_lengths: torch.Tensor
) -> torch.Tensor:
    batch = int(input_lengths.numel())
    max_len = int(input_lengths.max().item())
    hidden = hidden_states.size(-1)
    padded = hidden_states.new_zeros((batch, max_len, hidden))
    positions = torch.arange(max_len, device=hidden_states.device).unsqueeze(0)
    valid_tokens = (
        positions < input_lengths.to(device=hidden_states.device).unsqueeze(1)
    )
    padded[valid_tokens] = hidden_states
    return padded


def unpad_padded_hidden_states(
    padded_hidden_states: torch.Tensor, input_lengths: torch.Tensor
) -> torch.Tensor:
    max_len = padded_hidden_states.size(1)
    positions = torch.arange(max_len, device=padded_hidden_states.device).unsqueeze(0)
    valid_tokens = (
        positions < input_lengths.to(device=padded_hidden_states.device).unsqueeze(1)
    )
    return padded_hidden_states[valid_tokens]


def apply_module_to_padded_hidden_states(
    module: nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    shape = hidden_states.shape
    output = module(hidden_states.reshape(-1, shape[-1]))
    return output.reshape(*shape[:-1], output.size(-1))


def apply_add_bias_res_layernorm_to_padded_hidden_states(
    module: nn.Module, hidden_states: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    shape = hidden_states.shape
    output = module(
        hidden_states.reshape(-1, shape[-1]),
        residual.reshape(-1, shape[-1]),
        torch.empty(0),
    )
    return output.reshape(shape)


class BertDenseMaskedSelfAttention(nn.Module):
    def __init__(
        self,
        attn_config,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        quant_config: Optional[object],
        hw_kernel_config: Optional["HWKernelConfig"],
    ):
        super().__init__()
        self.tp_size = parallelism_config.get_attn_tp_size()
        self.head_num = attn_config.head_num
        self.kv_head_num = attn_config.kv_head_num
        self.size_per_head = attn_config.size_per_head
        self.q_size = self.head_num * self.size_per_head
        self.qkv_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_qkv_w,
            W.attn_qkv_s,
            W.attn_qkv_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.attn_qkv_s2,
            input_scale_key=W.attn_qkv_i_s,
        )
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_o_w,
            W.attn_o_s,
            W.attn_o_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.attn_o_s2,
            input_scale_key=W.attn_o_i_s,
        )
        self.o_proj.maybe_cache_quant_scale(1024)
        self.qk_fuse_norm = None
        if W.q_ln_gamma in weights and W.k_ln_gamma in weights:
            self.qk_fuse_norm = FusedQKRMSNorm(
                weights[W.q_ln_gamma],
                weights[W.k_ln_gamma],
                attn_config.head_num,
                attn_config.kv_head_num,
                attn_config.size_per_head,
                layernorm_eps,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        padded = pad_flat_hidden_states(hidden_states, input_lengths)
        attn_out = self.forward_padded(padded, attention_mask)
        return unpad_padded_hidden_states(attn_out, input_lengths)

    def forward_padded(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch = hidden_states.size(0)
        max_len = hidden_states.size(1)
        qkv = self.qkv_proj(hidden_states)
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv.reshape(batch * max_len, -1)).view(
                batch, max_len, -1
            )
        qkv = qkv.view(
            batch,
            max_len,
            self.head_num + 2 * self.kv_head_num,
            self.size_per_head,
        )
        q = qkv[:, :, : self.head_num].permute(0, 2, 1, 3)
        k = qkv[
            :, :, self.head_num : self.head_num + self.kv_head_num
        ].permute(0, 2, 1, 3)
        v = qkv[:, :, self.head_num + self.kv_head_num :].permute(0, 2, 1, 3)
        if self.kv_head_num != self.head_num:
            repeat_factor = self.head_num // self.kv_head_num
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        mask = attention_mask[:, None, :max_len, :max_len].to(
            dtype=torch.bool, device=q.device
        )
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch, max_len, self.q_size
        )
        attn_out = self.o_proj(attn_out)
        if self.tp_size > 1:
            attn_out = all_reduce(attn_out, group=Group.TP)
        return attn_out


class BertDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        attn_configs = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
        attn_configs.need_rope_kv_cache = False
        self.self_attn = CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
        )
        self.masked_self_attn = BertDenseMaskedSelfAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
        )
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config,
        )
        self.input_layernorm = AddBiasResLayerNorm(
            weights[W.post_ln_gamma],
            beta=weights[W.post_ln_beta],
            eps=config.layernorm_eps,
        )
        self.post_attention_layernorm = AddBiasResLayerNorm(
            weights[W.post_ffn_ln_gamma],
            beta=weights[W.post_ffn_ln_beta],
            eps=config.layernorm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        input_lengths: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        # Self Attention
        if attention_mask is not None and attention_mask.numel() > 0:
            if hidden_states.dim() == 3:
                hidden_states = self.masked_self_attn.forward_padded(
                    hidden_states, attention_mask
                )
            else:
                assert input_lengths is not None
                hidden_states = self.masked_self_attn(
                    hidden_states, input_lengths, attention_mask
                )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
            )
        if hidden_states.dim() == 3:
            hidden_states = apply_add_bias_res_layernorm_to_padded_hidden_states(
                self.input_layernorm, hidden_states, residual
            )
        else:
            hidden_states = self.input_layernorm(
                hidden_states, residual, torch.empty(0)
            )

        # Fully Connected
        residual = hidden_states
        if hidden_states.dim() == 3:
            hidden_states = apply_module_to_padded_hidden_states(
                self.mlp, hidden_states
            )
            hidden_states = apply_add_bias_res_layernorm_to_padded_hidden_states(
                self.post_attention_layernorm, hidden_states, residual
            )
        else:
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_attention_layernorm(
                hidden_states, residual, torch.empty(0)
            )
        return hidden_states


class BertModel(GptModelBase):
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
        self.embed_tokens = EmbeddingBert(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.pre_decoder_layernorm = LayerNorm(
            weight=weights.get_global_weight(W.pre_decoder_ln_gamma),
            beta=weights.get_global_weight(W.pre_decoder_ln_beta),
            eps=config.layernorm_eps,
        )
        self.layers = nn.ModuleList(
            [
                BertDecoderLayer(
                    config,
                    parallelism_config,
                    weights.weights[idx],
                    quant_config,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )

    def forward(
        self, inputs: PyModelInputs, fmha_impl: FMHAImplBase = None
    ) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        bert_embedding_inputs = inputs.bert_embedding_inputs
        inputs_embeds = self.embed_tokens(
            input_ids,
            bert_embedding_inputs.combo_position_ids,
            bert_embedding_inputs.position_encoding,
            bert_embedding_inputs.combo_tokens_type_ids,
            bert_embedding_inputs.token_type_embedding,
            bert_embedding_inputs.input_embedding_scalar,
        )
        hidden_states = self.pre_decoder_layernorm(inputs_embeds)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        attention_mask = inputs.attention_inputs.attention_mask
        use_dense_mask = attention_mask is not None and attention_mask.numel() > 0
        input_lengths = inputs.attention_inputs.input_lengths
        if use_dense_mask:
            hidden_states = pad_flat_hidden_states(hidden_states, input_lengths)
            attention_mask = attention_mask.to(
                dtype=torch.bool, device=hidden_states.device
            )
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                input_lengths=input_lengths,
                attention_mask=attention_mask if use_dense_mask else None,
            )
        if use_dense_mask:
            hidden_states = unpad_padded_hidden_states(hidden_states, input_lengths)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
