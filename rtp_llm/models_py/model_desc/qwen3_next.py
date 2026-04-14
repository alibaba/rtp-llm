import logging
import sys
from typing import Any, Dict, Optional

import torch
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    _get_group,
    all_gather,
    all_gather_async,
    all_reduce,
    broadcast_async,
)
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    LinearFactory,
    RMSNorm,
)
from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
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
from rtp_llm.models_py.triton_kernels.fla.chunk_cp_scan import (
    chunk_gated_delta_rule_fwd_cp_scan,
)
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.utils.debug import cudagraph_debug_kernel
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


class CpChunkAlignInfo(object):
    """Precomputed indices for chunk-aligned padding/reorder in CP linear attention."""

    def __init__(
        self,
        need_pad: bool,
        orig_full_cu: torch.Tensor,
        padded_cu: torch.Tensor,
        local_cu: torch.Tensor,
        local_total: int,
        local_NT: int,
        full_NT: int,
        chunk_size: int,
    ):
        self.need_pad = need_pad
        self.orig_full_cu = orig_full_cu
        self.padded_cu = padded_cu
        self.local_cu = local_cu
        self.local_total = local_total
        self.local_NT = local_NT
        self.full_NT = full_NT
        self.chunk_size = chunk_size

    @classmethod
    def build(
        cls,
        full_cu: torch.Tensor,
        cp_size: int,
        cp_rank: int,
        device: torch.device,
        chunk_size: int = 64,
    ) -> "CpChunkAlignInfo":
        batch_size = full_cu.shape[0] - 1
        align = cp_size * chunk_size
        orig_full_lengths = full_cu[1:] - full_cu[:-1]
        orig_full_cu = full_cu.clone()
        padded_lengths = ((orig_full_lengths + align - 1) // align) * align
        need_pad = (padded_lengths != orig_full_lengths).any().item()

        padded_cu = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        padded_cu[1:] = padded_lengths.cumsum(0)

        local_lengths = padded_lengths // cp_size
        local_cu = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        local_cu[1:] = local_lengths.cumsum(0)
        local_total = local_cu[-1].item()

        local_chunks_per_seq = local_lengths // chunk_size
        local_NT = local_chunks_per_seq.sum().item()
        full_NT = local_NT * cp_size

        return cls(
            need_pad=need_pad,
            orig_full_cu=orig_full_cu,
            padded_cu=padded_cu,
            local_cu=local_cu,
            local_total=local_total,
            local_NT=local_NT,
            full_NT=full_NT,
            chunk_size=chunk_size,
        )


class Qwen3NextMetadata(object):
    def __init__(
        self,
        prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None,
        is_target_verify: bool = False,
        full_prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None,
        full_prefill_cu_seqlens: Optional[torch.Tensor] = None,
        cp_restore_indices: Optional[torch.Tensor] = None,
        cp_local_extract_indices: Optional[torch.Tensor] = None,
        cp_local_valid_mask: Optional[torch.Tensor] = None,
        cp_write_cache_store_impl: Optional[WriteCacheStoreOp] = None,
        cp_chunk_align_info: Optional[CpChunkAlignInfo] = None,
    ):
        self.prefill_conv1d_meta = prefill_conv1d_meta
        self.is_target_verify = is_target_verify
        self.full_prefill_conv1d_meta = full_prefill_conv1d_meta
        self.full_prefill_cu_seqlens = full_prefill_cu_seqlens
        self.cp_restore_indices = cp_restore_indices
        self.cp_local_extract_indices = cp_local_extract_indices
        self.cp_local_valid_mask = cp_local_valid_mask
        self.cp_write_cache_store_impl = cp_write_cache_store_impl
        self.cp_chunk_align_info = cp_chunk_align_info

    def get_prefill_conv1d_meta(self) -> Optional[CausalConv1dMetadata]:
        return self.prefill_conv1d_meta

    @property
    def is_cp_linear_attn(self) -> bool:
        return self.cp_restore_indices is not None


class Qwen3NextGatedDeltaNetBase(torch.nn.Module):
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
        attn_tp_size = parallelism_config.get_attn_tp_size()
        self.local_num_k_heads: int = (
            linear_attn_config.linear_num_key_heads // attn_tp_size
        )
        self.local_num_v_heads: int = (
            linear_attn_config.linear_num_value_heads // attn_tp_size
        )
        self.num_key_value_heads: int = self.local_num_v_heads // self.local_num_k_heads
        self.linear_conv_kernel_dim: int = (
            self.linear_attn_config.linear_conv_kernel_dim
        )
        self.ssm_state_size: int = (
            self.local_num_v_heads * self.head_k_dim * self.head_v_dim
        )
        self.qkv_size: int = (
            self.head_k_dim * self.local_num_k_heads * 2
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
        self.dt_bias = weights[W.linear_attn_dt_b]
        self.alog = weights[W.linear_attn_alog]

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_conv_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        conv_states = self.linear_cache_converter.get_conv_state_tensor(kv_cache_tensor)
        return conv_states

    def _get_ssm_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        ssm_states = self.linear_cache_converter.get_ssm_state_tensor(kv_cache_tensor)
        return ssm_states


class Qwen3NextGatedDeltaNetPrefill(Qwen3NextGatedDeltaNetBase):
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
        # cu_seqlen_without_padding = attn_inputs.cu_seqlens[
        #     : attn_inputs.input_lengths.size(0) + 1
        # ]
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
        b: torch.Tensor,
        a: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        g, beta = fused_gdn_gating(self.alog, a, b, self.dt_bias)
        ssm_states = (
            self._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        context_batch_size = attn_inputs.input_lengths.shape[0]
        # cu_seqlens_without_padding = attn_inputs.cu_seqlens[: context_batch_size + 1]
        cu_seqlens_without_padding = attn_inputs.cu_seqlens
        initial_states: Optional[torch.Tensor] = None
        if ssm_states is not None:
            initial_states = torch.empty(
                context_batch_size,
                self.local_num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
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
        if ssm_states is not None:
            store_ssm_state_to_block_map(
                h,
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
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: Qwen3NextMetadata,
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
            mixed_qkv, b, a, kv_cache_tensor, seq_size_per_block, attn_inputs
        )
        if kv_cache is not None:
            # write kvcache to cache store
            compute_ops.write_cache_store(
                attn_inputs.input_lengths,
                attn_inputs.prefix_lengths,
                attn_inputs.kv_cache_block_id_host,
                attn_inputs.cache_store_inputs,
                kv_cache,
            )
        return attn_out


class Qwen3NextGatedDeltaNetDecode(Qwen3NextGatedDeltaNetBase):
    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        conv_states = self._get_conv_states(kv_cache_tensor)
        # (batch, dim) -> # (batch, dim, 1)
        batch, seq = self._get_bs_from_attenion_input(
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
        b: torch.Tensor,
        a: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        batch, seq = self._get_bs_from_attenion_input(
            mixed_qkv, attn_inputs, is_target_verify
        )
        # asserr head_k_dim == head_v_dim
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
        g, beta = fused_gdn_gating(self.alog, a, b, self.dt_bias)

        # contiguous will be applyed when call fused_recurrent_gated_delta_rule
        g = g.view(batch, seq, self.local_num_v_heads)
        beta = beta.view(batch, seq, self.local_num_v_heads)
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
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_d,
            use_qk_l2norm_in_kernel=True,
        )
        res = core_attn_out.reshape(
            [-1, core_attn_out.shape[2], core_attn_out.shape[3]]
        )
        return res

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: Qwen3NextMetadata,
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
            b,
            a,
            kv_cache_tensor,
            kv_cache.seq_size_per_block,
            attn_inputs,
            is_target_verify,
        )

        return attn_out

    def _get_bs_from_attenion_input(
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
        b, s = attention_inputs.prefix_lengths.size(
            0
        ), token // attention_inputs.prefix_lengths.size(0)
        return b, s


class Qwen3NextAttention(CausalAttention):
    def __init__(
        self,
        attn_config: AttentionConfigs,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        quant_config: Optional[object] = None,
    ):
        super().__init__(
            attn_config, parallelism_config, weights, layernorm_eps, quant_config
        )
        # maybe fuse gate in qkv_proj later
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.attn_gate_w, W.attn_gate_s, None, quant_config
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: Qwen3NextMetadata = Qwen3NextMetadata(),
    ) -> torch.Tensor:
        gate = self.gate(hidden_states)
        attn_out = super().forward(hidden_states, fmha_impl, kv_cache, gate)
        return attn_out


class Qwen3NextGatedDeltaNet(nn.Module):
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
        self.weights = weights
        self.quant_config = quant_config
        # in_proj_qkvz is bf16 / fp8
        self.in_proj_qkvz = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_qkvz_w, W.linear_attn_qkvz_s, None, quant_config
        )
        # in_proj_ba is bf16
        self.in_proj_ba = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_ba_w, None, None, quant_config
        )
        self.head_k_dim = linear_attn_config.linear_key_head_dim
        self.head_v_dim = linear_attn_config.linear_value_head_dim
        attn_tp_size = parallelism_config.get_attn_tp_size()
        self.local_num_k_heads = linear_attn_config.linear_num_key_heads // attn_tp_size
        self.local_num_v_heads = (
            linear_attn_config.linear_num_value_heads // attn_tp_size
        )
        self.num_key_value_heads = self.local_num_v_heads // self.local_num_k_heads

        self.prefill_gdn = Qwen3NextGatedDeltaNetPrefill(
            linear_attn_config, parallelism_config, weights
        )
        self.decode_gdn = Qwen3NextGatedDeltaNetDecode(
            linear_attn_config, parallelism_config, weights
        )
        self.norm = RmsNormGated(
            weights[W.linear_attn_norm_w],
            eps=layernorm_eps,
            group_size=linear_attn_config.linear_value_head_dim,
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

    # TODO: extract shared conv1d/FLA/ssm-state logic with Qwen3NextGatedDeltaNetPrefill
    # to eliminate duplication
    def _forward_cp_prefill(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attention_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        """CP prefill path: all-gather projected states, compute on full sequence,
        extract local zigzag tokens."""
        cp_info = attention_inputs.context_parallel_info

        packed = torch.cat([mixed_qkv, b, a], dim=-1)
        full_packed = all_gather(packed, group=Group.TP)

        padding_mask = cp_info.prefill_qkv_padding_mask
        restore_indices = cp_info.prefill_qkv_restore_indice
        unpad_restore = restore_indices[padding_mask == 1]
        full_packed = full_packed[unpad_restore]

        qkv_dim = mixed_qkv.shape[-1]
        b_dim = b.shape[-1]
        full_mixed_qkv = full_packed[:, :qkv_dim].contiguous()
        full_b = full_packed[:, qkv_dim : qkv_dim + b_dim].contiguous()
        full_a = full_packed[:, qkv_dim + b_dim :].contiguous()

        gdn = self.prefill_gdn
        full_cu = attn_meta.full_prefill_cu_seqlens
        full_conv_meta = attn_meta.full_prefill_conv1d_meta

        kv_cache_tensor: Optional[torch.Tensor] = None
        seq_size_per_block = 1
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block

        conv_states = (
            gdn._get_conv_states(kv_cache_tensor).transpose(1, 2)
            if kv_cache_tensor is not None
            else None
        )
        full_mixed_qkv = causal_conv1d_fn(
            x=full_mixed_qkv.transpose(0, 1),
            weight=gdn.conv_weights,
            bias=None,
            conv_states=conv_states,
            query_start_loc=full_cu,
            block_map=attention_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attention_inputs.prefix_lengths_d,
            metadata=full_conv_meta,
        ).transpose(0, 1)

        g, beta = fused_gdn_gating(gdn.alog, full_a, full_b, gdn.dt_bias)
        ssm_states = (
            gdn._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        context_batch_size = attention_inputs.input_lengths.shape[0]
        initial_states: Optional[torch.Tensor] = None
        if ssm_states is not None:
            initial_states = torch.empty(
                context_batch_size,
                gdn.local_num_v_heads,
                gdn.head_v_dim,
                gdn.head_k_dim,
                device=full_mixed_qkv.device,
                dtype=gdn.ssm_state_dtype,
            )
            load_initial_state_from_block_map(
                attention_inputs.prefix_lengths_d,
                attention_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                initial_states,
                seq_size_per_block,
            )

        query, key, value = torch.split(
            full_mixed_qkv,
            [
                gdn.local_num_k_heads * gdn.head_k_dim,
                gdn.local_num_k_heads * gdn.head_k_dim,
                gdn.local_num_v_heads * gdn.head_v_dim,
            ],
            dim=-1,
        )
        query = query.view(1, -1, gdn.local_num_k_heads, gdn.head_k_dim)
        key = key.view(1, -1, gdn.local_num_k_heads, gdn.head_k_dim)
        value = value.view(1, -1, gdn.local_num_v_heads, gdn.head_v_dim)

        cp_size = self.parallelism_config.tp_size
        cp_rank = self.parallelism_config.tp_rank
        cp_group = _get_group(Group.TP)

        chunk_align = attn_meta.cp_chunk_align_info
        chunk_size = chunk_align.chunk_size
        orig_full_cu = chunk_align.orig_full_cu
        padded_cu = chunk_align.padded_cu
        local_cu = chunk_align.local_cu
        local_total = chunk_align.local_total

        # Fused padding + local extraction: copy valid tokens directly from
        # original data into the local buffer, skipping the intermediate padded buffer.
        # Zero-init ensures padding positions are neutral (exp(0)=1, beta=0 → h unchanged).
        local_q = query.new_zeros(1, local_total, *query.shape[2:])
        local_k = key.new_zeros(1, local_total, *key.shape[2:])
        local_v = value.new_zeros(1, local_total, *value.shape[2:])
        local_g = g.new_zeros(1, local_total, g.shape[2])
        local_beta = beta.new_zeros(1, local_total, beta.shape[2])

        for seq_b in range(context_batch_size):
            orig_s = orig_full_cu[seq_b].item()
            orig_len = orig_full_cu[seq_b + 1].item() - orig_s
            ll = local_cu[seq_b + 1].item() - local_cu[seq_b].item()
            rank_offset = cp_rank * ll
            valid_len = min(ll, max(0, orig_len - rank_offset))
            if valid_len > 0:
                src_s = orig_s + rank_offset
                dst_s = local_cu[seq_b].item()
                local_q[:, dst_s : dst_s + valid_len] = query[
                    :, src_s : src_s + valid_len
                ]
                local_k[:, dst_s : dst_s + valid_len] = key[
                    :, src_s : src_s + valid_len
                ]
                local_v[:, dst_s : dst_s + valid_len] = value[
                    :, src_s : src_s + valid_len
                ]
                local_g[:, dst_s : dst_s + valid_len] = g[:, src_s : src_s + valid_len]
                local_beta[:, dst_s : dst_s + valid_len] = beta[
                    :, src_s : src_s + valid_len
                ]

        # Single call with cu_seqlens
        o, h, final_state = chunk_gated_delta_rule_fwd_cp_scan(
            q=local_q,
            k=local_k,
            v=local_v,
            g=local_g,
            beta=local_beta,
            initial_state=initial_states,
            output_final_state=(ssm_states is not None),
            cp_group=cp_group,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=local_cu,
        )

        # local output
        local_attn_out = o.squeeze(0)  # [local_total, H, V]
        out_shape_tail = local_attn_out.shape[1:]  # (H, V)
        local_total = local_attn_out.shape[0]

        # All-gather output back to full sequence
        local_flat = local_attn_out.reshape(local_total, -1)  # [local_total, H*V]
        full_flat = all_gather(
            local_flat, group=Group.TP
        )  # [cp_size * local_total, H*V]

        # Reorder from [rank0_all_seqs, rank1_all_seqs, ...] to [seq0_full, seq1_full, ...]
        reordered = torch.empty_like(full_flat)
        local_cu_list = local_cu.tolist()
        padded_cu_list = padded_cu.tolist()
        for seq_b in range(context_batch_size):
            ll = local_cu_list[seq_b + 1] - local_cu_list[seq_b]
            dst = padded_cu_list[seq_b]
            src_base = local_cu_list[seq_b]
            for r in range(cp_size):
                src = r * local_total + src_base
                reordered[dst + r * ll : dst + (r + 1) * ll] = full_flat[src : src + ll]

        full_attn_out = reordered.view(-1, *out_shape_tail)

        # Strip padding tokens to restore original sequence lengths
        if chunk_align.need_pad:
            real_pieces = []
            for seq_b in range(context_batch_size):
                padded_start = padded_cu_list[seq_b]
                orig_len = orig_full_cu[seq_b + 1].item() - orig_full_cu[seq_b].item()
                real_pieces.append(
                    full_attn_out[padded_start : padded_start + orig_len]
                )
            full_attn_out = torch.cat(real_pieces, dim=0)

        # Launch async h all-gather and final_state broadcast so comm overlaps with extract + norm + out_proj
        if ssm_states is not None:
            gathered_h, h_work = all_gather_async(h.contiguous(), group=Group.TP)
            global_final = final_state.contiguous()
            fs_work = broadcast_async(global_final, src=cp_size - 1, group=Group.TP)

        n_local = z.shape[0]
        local_attn_out = torch.zeros(
            n_local,
            *full_attn_out.shape[1:],
            device=full_attn_out.device,
            dtype=full_attn_out.dtype,
        )
        valid_mask = attn_meta.cp_local_valid_mask
        local_attn_out[valid_mask] = full_attn_out[attn_meta.cp_local_extract_indices]

        local_attn_out = self.norm(
            local_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        local_attn_out = local_attn_out.reshape(
            -1, self.local_num_v_heads * self.head_v_dim
        )
        local_attn_out = self.out_proj(local_attn_out)

        # Now wait for h all-gather and do reorder + store
        if ssm_states is not None:
            h_work.wait()
            fs_work.wait()

            chunk_tail = h.shape[2:]  # (H, K, V)
            local_NT = chunk_align.local_NT
            full_NT = chunk_align.full_NT
            local_chunks_per_seq = [
                local_cu_list[b + 1] // chunk_size - local_cu_list[b] // chunk_size
                for b in range(context_batch_size)
            ]
            local_chunk_offsets = [0]
            for lc in local_chunks_per_seq:
                local_chunk_offsets.append(local_chunk_offsets[-1] + lc)

            full_h = torch.empty(
                1, full_NT, *chunk_tail, device=h.device, dtype=h.dtype
            )
            for seq_b in range(context_batch_size):
                lc = local_chunks_per_seq[seq_b]
                dst_base = local_chunk_offsets[seq_b] * cp_size
                src_base = local_chunk_offsets[seq_b]
                for r in range(cp_size):
                    full_h[0, dst_base + r * lc : dst_base + (r + 1) * lc] = gathered_h[
                        r, src_base : src_base + lc
                    ]

            if chunk_align.need_pad:
                orig_full_lengths = orig_full_cu[1:] - orig_full_cu[:-1]
                padded_lengths = padded_cu[1:] - padded_cu[:-1]
                orig_chunks_per_seq = (
                    (orig_full_lengths + chunk_size - 1) // chunk_size
                ).tolist()
                padded_chunks_per_seq = (padded_lengths // chunk_size).tolist()
                real_h_pieces = []
                padded_chunk_offset = 0
                for seq_b in range(context_batch_size):
                    oc = orig_chunks_per_seq[seq_b]
                    pc = padded_chunks_per_seq[seq_b]
                    real_h_pieces.append(
                        full_h[0, padded_chunk_offset : padded_chunk_offset + oc]
                    )
                    padded_chunk_offset += pc
                full_h = torch.cat(real_h_pieces, dim=0).unsqueeze(0)

            store_ssm_state_to_block_map(
                full_h,
                global_final,
                attention_inputs.prefix_lengths_d,
                chunk_align.orig_full_cu,
                attention_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                seq_size_per_block,
                chunk_size=chunk_size,
            )

        if kv_cache is not None and attn_meta.cp_write_cache_store_impl is not None:
            attn_meta.cp_write_cache_store_impl(kv_cache)

        return local_attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        assert attention_inputs is not None, "attention_inputs is required"
        assert (
            attention_inputs.is_target_verify
            or not attention_inputs.is_prefill
            or attn_meta.get_prefill_conv1d_meta() is not None
            or attn_meta.is_cp_linear_attn
        ), "prefill_conv1d_meta is required for prefill"
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        mixed_qkv, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        if attention_inputs.is_prefill and not attn_meta.is_target_verify:
            if attn_meta.is_cp_linear_attn:
                return self._forward_cp_prefill(
                    mixed_qkv, z, b, a, attention_inputs, kv_cache, attn_meta
                )
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
        if self.parallelism_config.get_attn_tp_size() > 1:
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
            self.self_attn = Qwen3NextGatedDeltaNet(
                config.linear_attention_config,
                parallelism_config,
                weights,
                config.layernorm_eps,
                config.quant_config,
            )
        else:
            attn_configs = config.getAttentionConfigs(
                parallelism_config.get_attn_tp_size()
            )
            self.self_attn = Qwen3NextAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                config.quant_config,
            )

        if config.moe_style == 2:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph,
            )
        elif config.moe_style == 0:
            self.mlp = DenseMLP(
                config.activation_type, parallelism_config, weights, config.quant_config
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
        kv_cache: Optional[LayerKVCache] = None,
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

    def _build_cp_linear_attn_metadata(
        self,
        attention_inputs: PyAttentionInputs,
        device: torch.device,
    ) -> tuple[
        Optional[CausalConv1dMetadata],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[CpChunkAlignInfo],
    ]:
        """Precompute metadata for CP linear attention (per-layer all-gather path).

        Returns (full_conv1d_meta, full_cu_seqlens, restore_indices,
                 local_extract_indices, local_valid_mask, cp_chunk_align_info).
        """
        cp_info = attention_inputs.context_parallel_info
        if cp_info is None:
            return None, None, None, None, None, None

        # In CP mode the TP group is repurposed as the CP group, so tp_size == cp_size.
        cp_size = self.parallelism_config.tp_size
        cp_rank = self.parallelism_config.tp_rank

        full_new_lengths = cp_info.prefill_actual_input_lengths_cpu
        full_cu = torch.zeros(
            full_new_lengths.shape[0] + 1, dtype=torch.int32, device=device
        )
        full_cu[1:] = full_new_lengths.cumsum(0).to(device)
        full_conv1d_meta = prepare_causal_conv1d_metadata(
            query_start_loc=full_cu, device=device
        )

        restore_indices = cp_info.prefill_qkv_restore_indice
        padding_mask = cp_info.prefill_qkv_padding_mask
        unpad_restore = restore_indices[padding_mask == 1]

        total_ag = padding_mask.shape[0]
        local_chunk_total = total_ag // cp_size
        local_start = cp_rank * local_chunk_total

        inv_restore = torch.full((total_ag,), -1, dtype=torch.long, device=device)
        inv_restore[unpad_restore.long()] = torch.arange(
            unpad_restore.shape[0], device=device
        )

        local_inv = inv_restore[local_start : local_start + local_chunk_total]
        cp_local_valid_mask = local_inv >= 0
        cp_local_extract_indices = local_inv[cp_local_valid_mask]

        chunk_align = CpChunkAlignInfo.build(
            full_cu=full_cu,
            cp_size=cp_size,
            cp_rank=cp_rank,
            device=device,
        )

        return (
            full_conv1d_meta,
            full_cu,
            restore_indices,
            cp_local_extract_indices,
            cp_local_valid_mask,
            chunk_align,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:

        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        prefill_conv1d_meta = None
        is_target_verify = attention_inputs.is_target_verify
        is_cp = self.parallelism_config.prefill_cp_config.is_enabled()

        full_prefill_conv1d_meta = None
        full_prefill_cu_seqlens = None
        cp_restore_indices = None
        cp_local_extract_indices = None
        cp_local_valid_mask = None
        cp_write_cache_store_impl = None
        cp_chunk_align_info = None

        if attention_inputs.is_prefill and not is_target_verify:
            if is_cp:
                (
                    full_prefill_conv1d_meta,
                    full_prefill_cu_seqlens,
                    cp_restore_indices,
                    cp_local_extract_indices,
                    cp_local_valid_mask,
                    cp_chunk_align_info,
                ) = self._build_cp_linear_attn_metadata(
                    attention_inputs, hidden_states.device
                )
                if attention_inputs.cache_store_inputs:
                    cp_info = attention_inputs.context_parallel_info
                    cp_write_cache_store_impl = WriteCacheStoreOp(
                        cp_info.prefill_actual_input_lengths_cpu,
                        attention_inputs.prefix_lengths,
                        attention_inputs.kv_cache_block_id_host,
                        attention_inputs.cache_store_inputs,
                    )
            else:
                cu_seqlen_without_padding = attention_inputs.cu_seqlens
                prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                    query_start_loc=cu_seqlen_without_padding,
                    device=hidden_states.device,
                )

        attn_meta = Qwen3NextMetadata(
            prefill_conv1d_meta=prefill_conv1d_meta,
            is_target_verify=is_target_verify,
            full_prefill_conv1d_meta=full_prefill_conv1d_meta,
            full_prefill_cu_seqlens=full_prefill_cu_seqlens,
            cp_restore_indices=cp_restore_indices,
            cp_local_extract_indices=cp_local_extract_indices,
            cp_local_valid_mask=cp_local_valid_mask,
            cp_write_cache_store_impl=cp_write_cache_store_impl,
            cp_chunk_align_info=cp_chunk_align_info,
        )

        # qwen3_next model has only one full group (group 0): use fmha_impl from input param
        # if there is a model with more than 1 full groups,
        # we should prepare fmha_impl for each full group/ fix later

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        for i, decoder_layer in enumerate(self.layers):
            # Switch to correct block_map for this layer in hybrid attention mode
            select_block_map_for_layer(attention_inputs, i)
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=(self.kv_cache.get_layer_cache(i) if self.kv_cache else None),
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )

        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
