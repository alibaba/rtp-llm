import sys
from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    AttnImplFactory,
    CausalAttention,
    Embedding,
    FMHAImplBase,
    LinearFactory,
    RMSNorm,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    CausalConv1dMetadata,
    causal_conv1d_fn,
    causal_conv1d_update,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated
from rtp_llm.models_py.triton_kernels.fla.block import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.ops import HybridAttentionType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class Qwen3NextMetadata(object):
    def __init__(self, prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None):
        self.prefill_conv1d_meta = prefill_conv1d_meta

    def get_prefill_conv1d_meta(self) -> Optional[CausalConv1dMetadata]:
        return self.prefill_conv1d_meta


class Qwen3NextGatedDeltaNetBase(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weights = weights
        # params
        self.head_k_dim: int = config.linear_attention_config.linear_key_head_dim
        self.head_v_dim: int = config.linear_attention_config.linear_value_head_dim
        self.local_num_k_heads: int = (
            config.linear_attention_config.linear_num_key_heads
            // parallelism_config.tp_size
        )
        self.local_num_v_heads: int = (
            config.linear_attention_config.linear_num_value_heads
            // parallelism_config.tp_size
        )
        self.num_key_value_heads: int = self.local_num_v_heads // self.local_num_k_heads
        self.linear_conv_kernel_dim: int = (
            self.config.linear_attention_config.linear_conv_kernel_dim
        )
        self.ssm_state_size: int = (
            self.local_num_v_heads * self.head_k_dim * self.head_v_dim
        )
        self.qkv_size: int = (
            self.head_k_dim * self.local_num_k_heads * 2
            + self.head_v_dim * self.local_num_v_heads
        )
        self.conv_state_size: int = (self.linear_conv_kernel_dim - 1) * self.qkv_size
        # weights
        self.conv_weights = weights[W.linear_attn_conv1d_w].squeeze(1)
        self.dt_bias = weights[W.linear_attn_dt_b]
        self.alog = weights[W.linear_attn_alog]

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[KVCache],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_conv_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        _, block_size = kv_cache_tensor.view(kv_cache_tensor.shape[0], -1).shape
        assert (
            block_size >= self.ssm_state_size + self.conv_state_size
        ), "block_size is too small, please check seq_size_per_block"
        conv_states = torch.as_strided(
            kv_cache_tensor,
            (kv_cache_tensor.shape[0], self.linear_conv_kernel_dim - 1, self.qkv_size),
            (kv_cache_tensor.stride()[0], self.qkv_size, 1),
            storage_offset=self.ssm_state_size + kv_cache_tensor.storage_offset(),
        )
        return conv_states

    def _get_ssm_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        # maybe should support smsm cahe with difference dtype(fp32/bf16/fp16)
        _, block_size = kv_cache_tensor.view(kv_cache_tensor.shape[0], -1).shape
        assert (
            block_size >= self.ssm_state_size + self.conv_state_size
        ), "block_size is too small, please check seq_size_per_block"
        ssm_states = torch.as_strided(
            kv_cache_tensor,
            (
                kv_cache_tensor.shape[0],
                self.local_num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
            ),
            (
                kv_cache_tensor.stride()[0],
                self.head_k_dim * self.head_v_dim,
                self.head_k_dim,
                1,
            ),
            storage_offset=kv_cache_tensor.storage_offset(),
        )
        return ssm_states


class Qwen3NextGatedDeltaNetPrefill(Qwen3NextGatedDeltaNetBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, parallelism_config, weights)

    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        metadata: Optional[CausalConv1dMetadata] = None,
    ) -> torch.Tensor:
        # cu_seqlen_without_padding = attn_inputs.cu_seqlens[
        #     : attn_inputs.input_lengths.size(0) + 1
        # ]
        cu_seqlen_without_padding = attn_inputs.cu_seqlens
        conv_states = self._get_conv_states(kv_cache_tensor)
        out = causal_conv1d_fn(
            x=mixed_qkv.transpose(0, 1),
            weight=self.conv_weights,
            bias=None,
            conv_states=conv_states.transpose(1, 2),
            query_start_loc=cu_seqlen_without_padding,
            block_map=attn_inputs.kv_cache_block_id_device,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attn_inputs.prefix_lengths_d,
            metadata=metadata,
        ).transpose(0, 1)
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        g, beta = fused_gdn_gating(self.alog, a, b, self.dt_bias)
        ssm_states = self._get_ssm_states(kv_cache_tensor)
        context_batch_size = attn_inputs.input_lengths.shape[0]
        # cu_seqlens_without_padding = attn_inputs.cu_seqlens[: context_batch_size + 1]
        cu_seqlens_without_padding = attn_inputs.cu_seqlens
        initial_states = torch.empty(
            context_batch_size,
            self.local_num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=mixed_qkv.device,
            dtype=mixed_qkv.dtype,
        )
        load_initial_state_from_block_map(
            attn_inputs.prefix_lengths_d,
            attn_inputs.kv_cache_block_id_device,
            ssm_states,
            initial_states,
            seq_size_per_block,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_v_heads * self.head_v_dim,
            ],
            dim=-1,
        )
        query = query.view(1, query.shape[0], self.local_num_k_heads, self.head_k_dim)
        key = key.view(1, key.shape[0], self.local_num_k_heads, self.head_k_dim)
        value = value.view(1, value.shape[0], self.local_num_v_heads, self.head_v_dim)
        attn_out, h, final_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlens_without_padding,
            use_qk_l2norm_in_kernel=True,
        )
        store_ssm_state_to_block_map(
            h,
            final_state.to(h.dtype),
            attn_inputs.prefix_lengths_d,
            cu_seqlens_without_padding,
            attn_inputs.kv_cache_block_id_device,
            ssm_states,
            seq_size_per_block,
            chunk_size=64,
        )
        return attn_out.squeeze_(0)

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[KVCache],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for prefill"
        kv_cache_tensor: torch.Tensor = kv_cache.kv_cache_base.reshape(
            kv_cache.kv_cache_base.shape[0], -1
        )
        mixed_qkv = self._conv1d(
            mixed_qkv,
            kv_cache_tensor,
            kv_cache.seq_size_per_block,
            attn_inputs,
            metadata=attn_meta.get_prefill_conv1d_meta(),
        )
        attn_out = self._fla(
            mixed_qkv, b, a, kv_cache_tensor, kv_cache.seq_size_per_block, attn_inputs
        )
        return attn_out


class Qwen3NextGatedDeltaNetDecode(Qwen3NextGatedDeltaNetBase):
    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        conv_states = self._get_conv_states(kv_cache_tensor)
        out = causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(1, 2),
            self.conv_weights,
            bias=None,
            activation="silu",
            cache_seqlens=None,
            block_map=attn_inputs.kv_cache_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_d,
            num_accepted_tokens=None,
            max_query_len=1,
            query_start_loc=attn_inputs.decode_cu_seqlens_d,
        )
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        seq_len = mixed_qkv.shape[0]
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_v_heads * self.head_v_dim,
            ],
            dim=-1,
        )
        g, beta = fused_gdn_gating(self.alog, a, b, self.dt_bias)

        # contiguous will be applyed when call fused_recurrent_gated_delta_rule
        query = query.view(1, seq_len, self.local_num_k_heads, self.head_k_dim)
        key = key.view(1, seq_len, self.local_num_k_heads, self.head_k_dim)
        value = value.view(1, seq_len, self.local_num_v_heads, self.head_v_dim)
        ssm_states = self._get_ssm_states(kv_cache_tensor)
        core_attn_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            scale=None,
            initial_state=ssm_states,
            inplace_final_state=True,
            cu_seqlens=attn_inputs.decode_cu_seqlens_d,
            block_map=attn_inputs.kv_cache_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_d,
            num_accepted_tokens=None,
            use_qk_l2norm_in_kernel=True,
        )
        return core_attn_out.squeeze(0)

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[KVCache],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for decode"
        kv_cache_tensor: torch.Tensor = kv_cache.kv_cache_base.reshape(
            kv_cache.kv_cache_base.shape[0], -1
        )
        mixed_qkv = self._conv1d(
            mixed_qkv, kv_cache_tensor, kv_cache.seq_size_per_block, attn_inputs
        )
        attn_out = self._fla(
            mixed_qkv, b, a, kv_cache_tensor, kv_cache.seq_size_per_block, attn_inputs
        )
        return attn_out


class Qwen3NextAttention(CausalAttention):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ):
        quant_config = config.quant_config
        super().__init__(config, parallelism_config, weights, quant_config)
        # maybe fuse gate in qkv_proj later
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.attn_gate_w, W.attn_gate_s, None, quant_config
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: Qwen3NextMetadata = Qwen3NextMetadata(),
    ) -> torch.Tensor:
        gate = self.gate(hidden_states)
        attn_out = super().forward(hidden_states, fmha_impl, kv_cache, True, gate)
        return attn_out


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weights = weights
        quant_config = config.quant_config
        # in_proj_qkvz is bf16 / fp8
        self.in_proj_qkvz = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_qkvz_w, W.linear_attn_qkvz_s, None, quant_config
        )
        # in_proj_ba is bf16
        self.in_proj_ba = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_ba_w, None, None, quant_config
        )
        self.head_k_dim = config.linear_attention_config.linear_key_head_dim
        self.head_v_dim = config.linear_attention_config.linear_value_head_dim
        self.local_num_k_heads = (
            config.linear_attention_config.linear_num_key_heads
            // parallelism_config.tp_size
        )
        self.local_num_v_heads = (
            config.linear_attention_config.linear_num_value_heads
            // parallelism_config.tp_size
        )
        self.num_key_value_heads = self.local_num_v_heads // self.local_num_k_heads

        self.prefill_gdn = Qwen3NextGatedDeltaNetPrefill(
            config, parallelism_config, weights
        )
        self.decode_gdn = Qwen3NextGatedDeltaNetDecode(
            config, parallelism_config, weights
        )
        self.norm = RmsNormGated(
            weights[W.linear_attn_norm_w],
            eps=config.layernorm_eps,
            group_size=config.linear_attention_config.linear_value_head_dim,
        )
        self.out_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_out_w, W.linear_attn_out_s, None, quant_config
        )

    # mixed_qkvz, mixed_ba -> q, k, v, z, b, a
    def fix_query_key_value_ordering(
        self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        split_arg_list_qkvz = [
            self.head_k_dim * self.local_num_k_heads
            + self.head_k_dim * self.local_num_k_heads
            + self.head_v_dim * self.local_num_v_heads,
            self.head_v_dim * self.local_num_v_heads,
        ]

        mixed_qkv, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=1)
        b, a = torch.split(
            mixed_ba, [self.local_num_v_heads, self.local_num_v_heads], dim=1
        )
        # reshape to [token, v_head_num, v_head_dim]
        # b,a should be contiguous for fused_gdn_gating
        return mixed_qkv, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required"
        assert attention_inputs is not None, "attention_inputs is required"
        assert (
            not attention_inputs.is_prefill
            or attn_meta.get_prefill_conv1d_meta() is not None
        ), "prefill_conv1d_meta is required for prefill"
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        mixed_qkv, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        if attention_inputs.is_prefill:
            attn_output = self.prefill_gdn(
                mixed_qkv, b, a, attention_inputs, kv_cache, attn_meta
            )
        else:
            attn_output = self.decode_gdn(
                mixed_qkv, b, a, attention_inputs, kv_cache, attn_meta
            )
        attn_output = self.norm(
            attn_output.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        # from [token * head, dim] -> [token, head * dim]
        attn_output = attn_output.reshape(-1, self.local_num_v_heads * self.head_v_dim)
        attn_output = self.out_proj(attn_output)
        if self.parallelism_config.tp_size > 1:
            attn_output = all_reduce(attn_output, group=Group.TP)
        return attn_output


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]
        if self.layer_type == HybridAttentionType.LINEAR:
            self.self_attn = Qwen3NextGatedDeltaNet(config, parallelism_config, weights)
        else:
            self.self_attn = Qwen3NextAttention(config, parallelism_config, weights)
        self.mlp = GenericMoeLayer(
            config,
            parallelism_config,
            weights,
            moe_config,
            max_generate_batch_size,
            enable_cuda_graph,
        )

        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: Qwen3NextMetadata = Qwen3NextMetadata(),
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
            attention_inputs=attention_inputs,
            attn_meta=attn_meta,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3NextModel(GptModelBase):
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
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        prefill_conv1d_meta = None
        if attention_inputs.is_prefill:
            # cu_seqlen_without_padding = attention_inputs.cu_seqlens[
            #     : attention_inputs.input_lengths.size(0) + 1
            # ]
            cu_seqlen_without_padding = attention_inputs.cu_seqlens
            prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                query_start_loc=cu_seqlen_without_padding,
                device=hidden_states.device,
            )
        attn_meta = Qwen3NextMetadata(prefill_conv1d_meta)
        fmha_impl = AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self.weight,
            attention_inputs,
            self.fmha_config,
        )
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
