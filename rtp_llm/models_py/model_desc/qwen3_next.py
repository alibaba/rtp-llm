import logging
import sys
from typing import Any, Dict, Optional

import torch
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    LinearFactory,
    MultimodalEmbeddingInjector,
    RMSNorm,
    RMSResNorm,
)
from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    CausalConv1dMetadata,
    causal_conv1d_fn,
    causal_conv1d_update,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated
from rtp_llm.models_py.triton_kernels.common.scatter_qkv import scatter_qkv
from rtp_llm.models_py.triton_kernels.fla.block import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.triton_kernels.fla.chunk import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_flydsl_with_cache_store,
    is_flydsl_chunk_gdn_enabled,
    is_flydsl_chunk_gdn_shape_supported,
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
    ):
        self.prefill_conv1d_meta = prefill_conv1d_meta
        self.is_target_verify = is_target_verify
        self.full_prefill_conv1d_meta = full_prefill_conv1d_meta
        self.full_prefill_cu_seqlens = full_prefill_cu_seqlens
        self.cp_restore_indices = cp_restore_indices
        self.cp_local_extract_indices = cp_local_extract_indices
        self.cp_local_valid_mask = cp_local_valid_mask
        self.cp_write_cache_store_impl = cp_write_cache_store_impl

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
        # Validate on CPU once before entering the FlyDSL hot path; GPU tensors are
        # assumed to have been produced from already-validated host lengths.
        if not attn_inputs.input_lengths.is_cuda and (attn_inputs.input_lengths <= 0).any():
            raise ValueError(
                "FlyDSL GDN requires all input lengths > 0, but got "
                f"{attn_inputs.input_lengths.tolist()}"
            )
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
        # M >= 2048: scatter_qkv (Triton, SGLang port) avoids the .view() ->
        # .contiguous() copies that torch.split + view triggers. Below 2048,
        # kernel launch overhead beats the savings (microbench measured).
        if mixed_qkv.shape[0] >= 2048 and self.head_k_dim == self.head_v_dim:
            query, key, value = scatter_qkv(
                mixed_qkv,
                self.local_num_k_heads,
                self.local_num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
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
            )
            key = key.view(1, key.shape[0], self.local_num_k_heads, self.head_k_dim)
            value = value.view(
                1, value.shape[0], self.local_num_v_heads, self.head_v_dim
            )
        use_flydsl_chunk_gdn = (
            is_flydsl_chunk_gdn_enabled()
            and is_flydsl_chunk_gdn_shape_supported(query, key, value, beta)
        )
        if use_flydsl_chunk_gdn:
            # When ssm_states is provided the megakernel writes cache blocks
            # directly, so final_state is not consumed — skip allocation.
            need_final_state = ssm_states is None
            attn_out, final_state = chunk_gated_delta_rule_flydsl_with_cache_store(
                query,
                key,
                value,
                g,
                beta,
                prefix_lengths=(
                    attn_inputs.prefix_lengths_d if ssm_states is not None else None
                ),
                block_map=(
                    attn_inputs.kv_cache_kernel_block_id_device
                    if ssm_states is not None
                    else None
                ),
                ssm_states=ssm_states,
                seq_size_per_block=(
                    seq_size_per_block if ssm_states is not None else None
                ),
                initial_state=initial_states,
                output_final_state=need_final_state,
                cu_seqlens=cu_seqlens_without_padding,
                use_qk_l2norm_in_kernel=True,
            )
        else:
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
        if ssm_states is not None and not use_flydsl_chunk_gdn:
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
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        layer_idx: int = 0,
    ):
        super().__init__(
            attn_config,
            parallelism_config,
            weights,
            layernorm_eps,
            quant_config,
            hw_kernel_config=hw_kernel_config,
            layer_idx=layer_idx,
        )
        # maybe fuse gate in qkv_proj later
        self.gate = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_gate_w,
            W.attn_gate_s,
            None,
            quant_config,
            hw_kernel_config=hw_kernel_config,
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
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config
        self.weights = weights
        self.quant_config = quant_config
        self.head_k_dim = linear_attn_config.linear_key_head_dim
        self.head_v_dim = linear_attn_config.linear_value_head_dim
        attn_tp_size = parallelism_config.get_attn_tp_size()
        self.local_num_k_heads = linear_attn_config.linear_num_key_heads // attn_tp_size
        self.local_num_v_heads = (
            linear_attn_config.linear_num_value_heads // attn_tp_size
        )
        self.num_key_value_heads = self.local_num_v_heads // self.local_num_k_heads

        # qkvz+ba fusion (BF16 only): combine two in-projection GEMMs into one.
        # Saves a small kernel launch on each forward; on decode (M=1) HBM-access
        # merging shaves a few us per layer (trace measurement: -0.094 ms/step
        # on Qwen3.5-9B TP=2 in the original session).
        # FP8/quantized: qkvz has scales but ba doesn't, dtypes mismatch -> fall
        # back to the original 2-GEMM path.
        self._qkvz_ba_fused = weights.get(W.linear_attn_qkvz_s) is None
        if self._qkvz_ba_fused:
            qkvz_w = weights[W.linear_attn_qkvz_w]
            ba_w = weights[W.linear_attn_ba_w]
            self._qkvz_size = qkvz_w.shape[1]
            self._ba_size = ba_w.shape[1]
            _is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
            if _is_rocm:
                # ROCm: cat in [N, K] space then .t() to preserve column-major
                # physical layout that hipb_mm / swizzle kernels expect.
                fused_w = torch.cat([qkvz_w.t(), ba_w.t()], dim=0).t()
            else:
                # CUDA: row-major contiguous buffer (cuBLAS compatible).
                K = qkvz_w.shape[0]
                fused_w = torch.empty(
                    K,
                    self._qkvz_size + self._ba_size,
                    dtype=qkvz_w.dtype,
                    device=qkvz_w.device,
                )
                fused_w[:, : self._qkvz_size].copy_(qkvz_w)
                fused_w[:, self._qkvz_size :].copy_(ba_w)
            weights[W.linear_attn_qkvz_w] = fused_w[:, : self._qkvz_size]
            weights[W.linear_attn_ba_w] = fused_w[:, self._qkvz_size :]
            del qkvz_w, ba_w
            self.in_proj_fused = LinearFactory.create_linear(
                fused_w, None, None, quant_config, hw_kernel_config=hw_kernel_config
            )
            self.in_proj_qkvz = None
            self.in_proj_ba = None
        else:
            self.in_proj_qkvz = LinearFactory.create_linear_from_weights(
                weights,
                W.linear_attn_qkvz_w,
                W.linear_attn_qkvz_s,
                None,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )
            self.in_proj_ba = LinearFactory.create_linear_from_weights(
                weights,
                W.linear_attn_ba_w,
                None,
                None,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )
            self.in_proj_fused = None

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
            weights,
            W.linear_attn_out_w,
            W.linear_attn_out_s,
            None,
            quant_config,
            hw_kernel_config=hw_kernel_config,
        )

    def _input_project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the input projection and return (projected_qkvz, projected_ba).

        Hides the fusion vs 2-GEMM dispatch from callers (forward + tests).
        Both branches produce tensors with identical shape/semantics; the
        fused branch slices a single GEMM output, the fallback runs two.
        """
        if self._qkvz_ba_fused:
            fused = self.in_proj_fused(hidden_states)
            return fused[..., : self._qkvz_size], fused[..., self._qkvz_size :]
        return self.in_proj_qkvz(hidden_states), self.in_proj_ba(hidden_states)

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

        if full_mixed_qkv.shape[0] >= 2048 and gdn.head_k_dim == gdn.head_v_dim:
            query, key, value = scatter_qkv(
                full_mixed_qkv,
                gdn.local_num_k_heads,
                gdn.local_num_v_heads,
                gdn.head_k_dim,
                gdn.head_v_dim,
            )
        else:
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

        use_flydsl_chunk_gdn = (
            is_flydsl_chunk_gdn_enabled()
            and is_flydsl_chunk_gdn_shape_supported(query, key, value, beta)
        )
        if use_flydsl_chunk_gdn:
            need_final_state = ssm_states is None
            attn_out, final_state = chunk_gated_delta_rule_flydsl_with_cache_store(
                query,
                key,
                value,
                g,
                beta,
                prefix_lengths=(
                    attention_inputs.prefix_lengths_d
                    if ssm_states is not None
                    else None
                ),
                block_map=(
                    attention_inputs.kv_cache_kernel_block_id_device
                    if ssm_states is not None
                    else None
                ),
                ssm_states=ssm_states,
                seq_size_per_block=(
                    seq_size_per_block if ssm_states is not None else None
                ),
                initial_state=initial_states,
                output_final_state=need_final_state,
                cu_seqlens=full_cu,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            attn_out, h, final_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_states,
                output_final_state=True,
                cu_seqlens=full_cu,
                use_qk_l2norm_in_kernel=True,
            )

        if ssm_states is not None and not use_flydsl_chunk_gdn:
            store_ssm_state_to_block_map(
                h,
                final_state,
                attention_inputs.prefix_lengths_d,
                full_cu,
                attention_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                seq_size_per_block,
                chunk_size=64,
            )

        if kv_cache is not None and attn_meta.cp_write_cache_store_impl is not None:
            attn_meta.cp_write_cache_store_impl(kv_cache)

        full_attn_out = attn_out.squeeze_(0)

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
        projected_states_qkvz, projected_states_ba = self._input_project(hidden_states)
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
        hw_kernel_config: Optional["HWKernelConfig"] = None,
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
                hw_kernel_config=hw_kernel_config,
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
                hw_kernel_config=hw_kernel_config,
                layer_idx=layer_idx,
            )

        if config.moe_style == 2:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )
        elif config.moe_style == 0:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                config.quant_config,
                hw_kernel_config=hw_kernel_config,
            )

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
        attn_meta: Qwen3NextMetadata = Qwen3NextMetadata(),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
            attention_inputs=attention_inputs,
            attn_meta=attn_meta,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


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
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
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
    ]:
        """Precompute metadata for CP linear attention (per-layer all-gather path).

        Returns (full_conv1d_meta, full_cu_seqlens, restore_indices,
                 local_extract_indices, local_valid_mask).
        """
        cp_info = attention_inputs.context_parallel_info
        if cp_info is None:
            return None, None, None, None, None

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

        return (
            full_conv1d_meta,
            full_cu,
            restore_indices,
            cp_local_extract_indices,
            cp_local_valid_mask,
        )

    def word_embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        input_ids: torch.Tensor = inputs.input_ids
        return self.embed_tokens(input_ids)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        hidden_states = self.word_embedding(inputs)

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

        if attention_inputs.is_prefill and not is_target_verify:
            if is_cp:
                (
                    full_prefill_conv1d_meta,
                    full_prefill_cu_seqlens,
                    cp_restore_indices,
                    cp_local_extract_indices,
                    cp_local_valid_mask,
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
        )

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        residual = torch.zeros_like(hidden_states)

        for i, decoder_layer in enumerate(self.layers):
            select_block_map_for_layer(attention_inputs, i)
            hidden_states, residual = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )

        hidden_states, residual = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


class Qwen35Model(Qwen3NextModel):
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
            moe_config,
            max_generate_batch_size,
            fmha_config,
            py_hw_kernel_config,
            device_resource_config,
        )
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()

    def word_embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        input_ids: torch.Tensor = inputs.input_ids

        position_ids = inputs.combo_position_ids
        token_type_ids = inputs.embedding_inputs.combo_tokens_type_ids
        text_tokens_mask = inputs.embedding_inputs.text_tokens_mask
        mm_features = inputs.multimodal_inputs.multimodal_features
        mm_feature_locs = inputs.multimodal_inputs.mm_features_locs

        inputs_embeds = self.embed_tokens(
            input_ids, position_ids, token_type_ids, text_tokens_mask
        )
        hidden_states = self.multimodal_embedding_injector(
            inputs_embeds, mm_features, mm_feature_locs
        )
        return hidden_states
