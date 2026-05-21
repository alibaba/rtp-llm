"""Kimi Linear model description.

Hybrid architecture:
  - KDA (Kimi Delta Attention) linear attention layers
  - MLA (Multi-head Latent Attention) full attention layers
  - MoE FFN with sigmoid routing (layer 1+), Dense FFN (layer 0)
"""

import logging
from typing import Any, Dict, Optional

import torch
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import DecodeLayerOutput, GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    DenseMLP,
    Embedding,
    FMHAImplBase,
    LinearFactory,
    MlaAttention,
    RMSResNorm,
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
from rtp_llm.models_py.triton_kernels.kimi_kda import (
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
)
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops import (
    AttentionConfigs,
    HybridAttentionType,
    LinearAttentionConfig,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import to_torch_dtype


class KimiLinearMetadata(object):
    def __init__(
        self,
        prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None,
        is_target_verify: bool = False,
    ):
        self.prefill_conv1d_meta = prefill_conv1d_meta
        self.is_target_verify = is_target_verify

    def get_prefill_conv1d_meta(self) -> Optional[CausalConv1dMetadata]:
        return self.prefill_conv1d_meta


class KimiLinearKDABase(nn.Module):
    """Base class for KDA (Kimi Delta Attention) prefill/decode.

    KDA differs from GDN in:
      - qkv_size = q + k + v (no z component, since output gate is separate LoRA)
      - beta is [B, H] scalar per head (from b_proj), not packed with a
      - forget gate g is [B, H, D] per-dim vector (from LoRA + fused_kda_gate)
    """

    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config
        self.weights = weights
        # params
        self.head_k_dim: int = linear_attn_config.linear_key_head_dim
        self.head_v_dim: int = linear_attn_config.linear_value_head_dim
        assert (
            self.head_k_dim == self.head_v_dim
        ), "head_k_dim and head_v_dim must be the same now"
        self.local_num_k_heads: int = (
            linear_attn_config.linear_num_key_heads // parallelism_config.tp_size
        )
        self.local_num_v_heads: int = (
            linear_attn_config.linear_num_value_heads // parallelism_config.tp_size
        )
        self.linear_conv_kernel_dim: int = (
            self.linear_attn_config.linear_conv_kernel_dim
        )
        self.ssm_state_size: int = (
            self.local_num_v_heads * self.head_k_dim * self.head_v_dim
        )
        # KDA: qkv only (no z), since output gate is separate LoRA
        self.qkv_size: int = (
            self.head_k_dim * self.local_num_k_heads
            + self.head_k_dim * self.local_num_k_heads
            + self.head_v_dim * self.local_num_v_heads
        )
        self.conv_state_size: int = (self.linear_conv_kernel_dim - 1) * self.qkv_size
        self.ssm_state_dtype: torch.dtype = to_torch_dtype(
            linear_attn_config.ssm_state_dtype
        )
        self.conv_state_dtype: torch.dtype = to_torch_dtype(
            linear_attn_config.conv_state_dtype
        )
        self.linear_cache_converter = LinearCacheConverter(
            local_num_v_heads=self.local_num_v_heads,
            head_v_dim=self.head_v_dim,
            head_k_dim=self.head_k_dim,
            ssm_state_dtype=self.ssm_state_dtype,
            linear_conv_kernel_dim=self.linear_conv_kernel_dim,
            qkv_size=self.qkv_size,
            conv_state_dtype=self.conv_state_dtype,
        )
        # weights
        self.conv_weights = weights[W.linear_attn_conv1d_w].squeeze(1)
        self.dt_bias = weights[W.linear_attn_dt_b_kda]
        self.alog = weights[W.linear_attn_alog]

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_conv_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        conv_states = self.linear_cache_converter.get_conv_state_tensor(kv_cache_tensor)
        return conv_states

    def _get_ssm_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        ssm_states = self.linear_cache_converter.get_ssm_state_tensor(kv_cache_tensor)
        return ssm_states


class KimiLinearKDAPrefill(KimiLinearKDABase):
    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(linear_attn_config, parallelism_config, weights)

    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        metadata: Optional[CausalConv1dMetadata] = None,
    ) -> torch.Tensor:
        cu_seqlen_without_padding = attn_inputs.cu_seqlens
        conv_states = (
            self._get_conv_states(kv_cache_tensor).transpose(1, 2)
            if kv_cache_tensor is not None
            else None
        )
        out = causal_conv1d_fn(
            x=mixed_qkv.transpose(0, 1),
            weight=self.conv_weights,
            bias=None,
            conv_states=conv_states,
            query_start_loc=cu_seqlen_without_padding,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attn_inputs.prefix_lengths_d,
            metadata=metadata,
        ).transpose(0, 1)
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        # Compute gate: [T, local_H*D] -> [1, T, local_H, D]
        g = forget_gate.view(
            1, forget_gate.shape[0], self.local_num_v_heads, self.head_k_dim
        ).contiguous()
        # beta: [token, H] -> apply sigmoid in float32 -> [1, token, H]
        beta_reshaped = beta.float().sigmoid().unsqueeze(0)

        ssm_states = (
            self._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        context_batch_size = attn_inputs.input_lengths.shape[0]
        cu_seqlens_without_padding = attn_inputs.cu_seqlens
        initial_states: Optional[torch.Tensor] = None
        if ssm_states is not None:
            initial_states = torch.empty(
                context_batch_size,
                self.local_num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                device=mixed_qkv.device,
                dtype=self.ssm_state_dtype,
            )
            load_initial_state_from_block_map(
                attn_inputs.prefix_lengths_d,
                attn_inputs.kv_cache_kernel_block_id_device,
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
        query = query.view(
            1, query.shape[0], self.local_num_k_heads, self.head_k_dim
        ).contiguous()
        key = key.view(
            1, key.shape[0], self.local_num_k_heads, self.head_k_dim
        ).contiguous()
        value = value.view(
            1, value.shape[0], self.local_num_v_heads, self.head_v_dim
        ).contiguous()

        q_len = query.shape[1]
        # Prefill: use chunk_kda with fused gate (use_gate_in_kernel=True)
        attn_out, final_state, h = chunk_kda(
            query,
            key,
            value,
            g,
            beta_reshaped,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlens_without_padding,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            return_intermediate_states=True,
            A_log=self.alog,
            dt_bias=self.dt_bias,
        )
        h_from_chunk = h

        if ssm_states is not None and final_state is not None:
            _target_dtype = ssm_states.dtype
            h_for_store = (
                h_from_chunk
                if h_from_chunk is not None
                else final_state.unsqueeze(0).unsqueeze(0)
            )
            # store_ssm_state_to_block_map requires h and final_state in float32
            if h_for_store.dtype != torch.float32:
                h_for_store = h_for_store.float()
            if final_state.dtype != torch.float32:
                final_state = final_state.float()

            store_ssm_state_to_block_map(
                h_for_store,
                final_state,
                attn_inputs.prefix_lengths_d,
                cu_seqlens_without_padding,
                attn_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                seq_size_per_block,
                chunk_size=64,
            )

        return attn_out.squeeze_(0)

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        kv_cache_tensor: Optional[torch.Tensor] = None
        seq_size_per_block = 1
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block
        mixed_qkv = self._conv1d(
            mixed_qkv,
            kv_cache_tensor,
            seq_size_per_block,
            attn_inputs,
            metadata=attn_meta.get_prefill_conv1d_meta(),
        )
        attn_out = self._fla(
            mixed_qkv,
            forget_gate,
            beta,
            kv_cache_tensor,
            seq_size_per_block,
            attn_inputs,
        )
        if kv_cache is not None:
            compute_ops.write_cache_store(
                attn_inputs.input_lengths,
                attn_inputs.prefix_lengths,
                attn_inputs.kv_cache_block_id_host,
                attn_inputs.cache_store_inputs,
                kv_cache,
            )
        return attn_out


class KimiLinearKDADecode(KimiLinearKDABase):
    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        conv_states = self._get_conv_states(kv_cache_tensor)
        batch, seq = self._get_bs_from_attention_input(
            mixed_qkv, attn_inputs, is_target_verify
        )

        origin_shape = mixed_qkv.shape
        mixed_qkv = mixed_qkv.reshape(batch, seq, -1).transpose(1, 2)
        out = causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(1, 2),
            self.conv_weights,
            bias=None,
            activation="silu",
            cache_seqlens=None,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_d,
        )
        out = out.transpose(1, 2).reshape(origin_shape)
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        batch, seq = self._get_bs_from_attention_input(
            mixed_qkv, attn_inputs, is_target_verify
        )
        # Split qkv
        mixed_qkv = mixed_qkv.reshape(
            batch,
            seq,
            self.local_num_k_heads * 2 + self.local_num_v_heads,
            self.head_k_dim,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.local_num_k_heads,
                self.local_num_k_heads,
                self.local_num_v_heads,
            ],
            dim=2,
        )

        # Compute gate: reshape to [B, S, local_H, D] for fused_recurrent
        forget_gate_2d = forget_gate.reshape(batch * seq, -1)
        g = forget_gate_2d.view(
            batch, seq, self.local_num_v_heads, self.head_k_dim
        ).contiguous()
        # beta: [batch*seq, H] -> sigmoid in float32 -> [batch, seq, H]
        beta_out = beta.reshape(batch * seq, -1).float().sigmoid()
        beta_out = beta_out.view(batch, seq, self.local_num_v_heads)

        ssm_states = self._get_ssm_states(kv_cache_tensor)

        core_attn_out, _ = fused_recurrent_kda(
            q=query.contiguous(),
            k=key.contiguous(),
            v=value.contiguous(),
            g=g,
            beta=beta_out,
            scale=None,
            initial_state=ssm_states,
            A_log=self.alog,
            dt_bias=self.dt_bias,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_d,
        )

        res = core_attn_out.reshape(
            [-1, core_attn_out.shape[2], core_attn_out.shape[3]]
        )
        return res

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for decode"
        assert (
            kv_cache.kv_cache_base is not None
        ), "kv_cache_tensor is required for decode"
        kv_cache_tensor: torch.Tensor = kv_cache.kv_cache_base.reshape(
            kv_cache.kv_cache_base.shape[0], -1
        )
        is_target_verify = attn_meta.is_target_verify

        mixed_qkv = self._conv1d(
            mixed_qkv,
            kv_cache_tensor,
            kv_cache.seq_size_per_block,
            attn_inputs,
            is_target_verify,
        )

        attn_out = self._fla(
            mixed_qkv,
            forget_gate,
            beta,
            kv_cache_tensor,
            kv_cache.seq_size_per_block,
            attn_inputs,
            is_target_verify,
        )
        return attn_out

    def _get_bs_from_attention_input(
        self,
        mixed_qkv: torch.Tensor,
        attention_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> tuple[int, int]:
        token, _ = mixed_qkv.shape
        if not is_target_verify:
            return token, 1
        assert (
            attention_inputs.prefix_lengths.size(0) > 0
        ), f"prefill_lengths size: {attention_inputs.prefix_lengths.size(0)} <=0 when target verify"
        assert (
            token % attention_inputs.prefix_lengths.size(0) == 0
        ), f"token: {token} is not divisible by prefill_lengths size: {attention_inputs.prefix_lengths.size(0)} when target verify"
        b = attention_inputs.prefix_lengths.size(0)
        s = token // b
        return b, s


class KimiLinearKDA(nn.Module):
    """KDA (Kimi Delta Attention) module.

    Forward flow:
      1. qkv = in_proj_qkv(hidden)           # merged q+k+v projection
      2. beta_input = in_proj_b(hidden)       # beta gate input [token, H]
      3. forget_gate = f_b_proj(f_a_proj(hidden))  # LoRA forget gate [token, H*D]
      4. g_proj = g_b_proj(g_a_proj(hidden))  # LoRA output gate [token, H*D]
      5. conv1d on qkv
      6. FLA attention (chunk_kda for prefill, fused_recurrent_kda for decode)
         - fused_kda_gate applies: g = -exp(A_log) * softplus(forget_gate + dt_bias)
         - beta = sigmoid(beta_input)
      7. o_norm(attn_out, g_proj) with sigmoid activation
      8. o_proj(result) + all_reduce
    """

    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        quant_config: Optional[object] = None,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config

        # Projections
        self.in_proj_qkv = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_qkv_w, None, None, quant_config
        )
        self.in_proj_b = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_b_w, None, None, quant_config
        )
        # LoRA forget gate
        self.f_a_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_f_a_w, None, None, quant_config
        )
        self.f_b_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_f_b_w, None, None, quant_config
        )
        # LoRA output gate
        self.g_a_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_g_a_w, None, None, quant_config
        )
        self.g_b_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_g_b_w, None, None, quant_config
        )

        self.head_k_dim = linear_attn_config.linear_key_head_dim
        self.head_v_dim = linear_attn_config.linear_value_head_dim
        self.local_num_v_heads = (
            linear_attn_config.linear_num_value_heads // parallelism_config.tp_size
        )

        self.prefill_kda = KimiLinearKDAPrefill(
            linear_attn_config, parallelism_config, weights
        )
        self.decode_kda = KimiLinearKDADecode(
            linear_attn_config, parallelism_config, weights
        )
        # o_norm with sigmoid activation (not SwiGLU)
        self.norm = RmsNormGated(
            weights[W.linear_attn_norm_w],
            eps=layernorm_eps,
            group_size=linear_attn_config.linear_value_head_dim,
            activation="sigmoid",
        )
        self.out_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_out_w, None, None, quant_config
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        assert attention_inputs is not None, "attention_inputs is required"
        assert (
            attention_inputs.is_target_verify
            or not attention_inputs.is_prefill
            or attn_meta.get_prefill_conv1d_meta() is not None
        ), "prefill_conv1d_meta is required for prefill"

        # 1. Projections
        projected_qkv = self.in_proj_qkv(hidden_states)
        beta_input = self.in_proj_b(hidden_states)  # [token, H]
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))  # [token, H*D]
        g_proj = self.g_b_proj(self.g_a_proj(hidden_states))  # [token, H*D]

        # 2. Prefill or decode
        if attention_inputs.is_prefill and not attn_meta.is_target_verify:
            attn_output = self.prefill_kda(
                projected_qkv,
                forget_gate,
                beta_input,
                attention_inputs,
                kv_cache,
                attn_meta,
            )
        else:
            attn_output = self.decode_kda(
                projected_qkv,
                forget_gate,
                beta_input,
                attention_inputs,
                kv_cache,
                attn_meta,
            )

        # 3. o_norm with sigmoid gating: y = RMSNorm(attn_out) * sigmoid(g_proj)
        attn_output = self.norm(
            attn_output.reshape(-1, self.head_v_dim),
            g_proj.reshape(-1, self.head_v_dim),
        )

        # from [token * head, dim] -> [token, head * dim]
        attn_output = attn_output.reshape(-1, self.local_num_v_heads * self.head_v_dim)

        # 4. Output projection + all_reduce
        attn_output = self.out_proj(attn_output)

        if self.parallelism_config.get_attn_tp_size() > 1:
            attn_output = all_reduce(attn_output, group=Group.TP)

        return attn_output


class KimiLinearDecoderLayer(nn.Module):
    """Kimi Linear decoder layer.

    Hybrid KDA/MLA attention + MoE/Dense FFN.
    Uses RMSResNorm (fused residual add + layernorm) like GenericMoeDecoderLayer.
    """

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

        quant_config = config.quant_config

        if self.layer_type == HybridAttentionType.LINEAR:
            self.self_attn = KimiLinearKDA(
                config.linear_attention_config,
                parallelism_config,
                weights,
                config.layernorm_eps,
                quant_config,
            )
        else:
            # Full MLA attention layer
            self.self_attn = MlaAttention(
                config.attn_config,
                parallelism_config,
                weights,
                layer_idx,
                config.layernorm_eps,
                quant_config,
            )

        # FFN: Dense (layer 0) or MoE (layer 1+)
        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type, parallelism_config, weights, quant_config
            )
        else:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
            )

        # RMSResNorm: fused residual add + layernorm
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: KimiLinearMetadata = KimiLinearMetadata(),
    ) -> DecodeLayerOutput:
        # Fused: residual = residual + hidden_states, hidden_states = RMSNorm(residual)
        hidden_states = self.input_layernorm(hidden_states, residual)

        # Self Attention (KDA or MLA)
        if self.layer_type == HybridAttentionType.LINEAR:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
            )

        # Fused: residual = residual + hidden_states, hidden_states = RMSNorm(residual)
        hidden_states = self.post_attention_layernorm(hidden_states, residual)

        # MLP (Dense or MoE)
        hidden_states = self.mlp(hidden_states)

        return DecodeLayerOutput(hidden_states, residual)


class KimiLinearModel(GptModelBase):
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
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                KimiLinearDecoderLayer(
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
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        prefill_conv1d_meta = None
        is_target_verify = attention_inputs.is_target_verify
        if attention_inputs.is_prefill and not is_target_verify:
            cu_seqlen_without_padding = attention_inputs.cu_seqlens
            prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                query_start_loc=cu_seqlen_without_padding,
                device=hidden_states.device,
            )

        attn_meta = KimiLinearMetadata(prefill_conv1d_meta, is_target_verify)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        residual = torch.zeros_like(hidden_states)

        for i, decoder_layer in enumerate(self.layers):
            select_block_map_for_layer(attention_inputs, i)
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )
            hidden_states = output.hidden_states
            residual = output.residual

        hidden_states = self.norm(hidden_states, residual)

        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
