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
    store_conv_state_to_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.chunk_cp_scan import (
    chunk_gated_delta_rule_fwd_cp_scan,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_cp_zigzag import (
    chunk_gated_delta_rule_fwd_cp_zigzag,
    exchange_conv_context,
    prepend_conv_context,
    strip_conv_context,
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
    """Precomputed indices for chunk-aligned padding/reorder in CP linear attention.

    Plan A: align padded sequence to (cp_size * 2 * chunk_size) so that each rank's
    half-segment is a chunk_size multiple — segment-internal chunk boundaries then
    coincide with full-sequence chunk boundaries, making h states reusable for
    SSM cache writes. Padding tokens must be made identity-preserving in the
    GDN recurrence (see _forward_cp_prefill mask application).
    """

    def __init__(
        self,
        orig_full_cu: torch.Tensor,
        local_cu: torch.Tensor,
        local_total: int,
        chunk_size: int,
        h_causal_indices: torch.Tensor,
        seg_cu: torch.Tensor,
        causal_order: torch.Tensor,
        local_cu_cpu: list,
        half_lengths_cpu: list,
        orig_full_lengths_cpu: list,
        padded_full_cu_d: torch.Tensor,
        orig_full_lengths_d: torch.Tensor,
        local_padding_mask: Optional[torch.Tensor] = None,
    ):
        self.orig_full_cu = orig_full_cu
        self.local_cu = local_cu
        self.local_total = local_total
        self.chunk_size = chunk_size
        self.h_causal_indices = h_causal_indices
        self.seg_cu = seg_cu
        self.causal_order = causal_order
        self.local_cu_cpu = local_cu_cpu
        self.half_lengths_cpu = half_lengths_cpu
        self.orig_full_lengths_cpu = orig_full_lengths_cpu
        self.padded_full_cu_d = padded_full_cu_d
        self.orig_full_lengths_d = orig_full_lengths_d
        self.local_padding_mask = local_padding_mask

    @classmethod
    def build(
        cls,
        cp_size: int,
        cp_rank: int,
        device: torch.device,
        orig_lengths_cpu: list,
        padded_full_cu: torch.Tensor,
        chunk_size: int = 64,
        local_padding_mask: Optional[torch.Tensor] = None,
    ) -> "CpChunkAlignInfo":
        """Build CP metadata from CPU-side sequence lengths.

        orig_lengths_cpu: per-sequence original lengths (Python list of ints).
        padded_full_cu: cumulative padded lengths from C++, int32 on `device`.
            Stored as-is; no re-cumsum, no GPU sync.
        local_padding_mask: optional pre-built mask. Pass `cp_local_valid_mask`
            when any sequence was padded; None when caller verified no padding.
        """
        from rtp_llm.models_py.triton_kernels.fla.chunk_cp_zigzag import (
            zigzag_causal_order,
        )

        batch_size = len(orig_lengths_cpu)
        align = cp_size * 2 * chunk_size

        padded_lengths_cpu = [
            ((L + align - 1) // align) * align for L in orig_lengths_cpu
        ]
        local_lengths_cpu = [L // cp_size for L in padded_lengths_cpu]
        half_lengths_cpu = [L // 2 for L in local_lengths_cpu]

        local_cu_cpu = [0]
        for L in local_lengths_cpu:
            local_cu_cpu.append(local_cu_cpu[-1] + L)
        local_total = local_cu_cpu[-1]

        orig_full_cu_cpu = [0]
        for L in orig_lengths_cpu:
            orig_full_cu_cpu.append(orig_full_cu_cpu[-1] + L)

        # seg_cu: treats each seq's two halves as separate sequences
        seg_cu_cpu = [0]
        for h in half_lengths_cpu:
            seg_cu_cpu.append(seg_cu_cpu[-1] + h)
            seg_cu_cpu.append(seg_cu_cpu[-1] + h)

        local_chunks_per_seq = [L // chunk_size for L in local_lengths_cpu]
        local_NT = sum(local_chunks_per_seq)
        half_chunks = [n // 2 for n in local_chunks_per_seq]
        causal_order = zigzag_causal_order(cp_size)

        seg_chunk_offsets = [0]
        for hc in half_chunks:
            seg_chunk_offsets.append(seg_chunk_offsets[-1] + 2 * hc)
        orig_chunks_per_seq = [
            (L + chunk_size - 1) // chunk_size for L in orig_lengths_cpu
        ]

        indices = []
        for b in range(batch_size):
            hc = half_chunks[b]
            seg_base = seg_chunk_offsets[b]
            remaining = orig_chunks_per_seq[b]
            for pos in range(2 * cp_size):
                ag_idx = causal_order[pos]
                r = ag_idx // 2
                seg = ag_idx % 2
                src_start = r * local_NT + seg_base + seg * hc
                n = min(hc, remaining)
                if n <= 0:
                    break
                indices.extend(range(src_start, src_start + n))
                remaining -= n

        local_cu = torch.tensor(local_cu_cpu, dtype=torch.long, device=device)
        seg_cu = torch.tensor(seg_cu_cpu, dtype=torch.long, device=device)
        orig_full_cu = torch.tensor(orig_full_cu_cpu, dtype=torch.long, device=device)
        h_causal_indices = torch.tensor(indices, dtype=torch.long, device=device)
        causal_order_t = torch.tensor(causal_order, dtype=torch.long, device=device)
        orig_full_lengths_d = torch.tensor(
            orig_lengths_cpu, dtype=torch.int32, device=device
        )

        return cls(
            orig_full_cu=orig_full_cu,
            local_cu=local_cu,
            local_total=local_total,
            chunk_size=chunk_size,
            h_causal_indices=h_causal_indices,
            seg_cu=seg_cu,
            causal_order=causal_order_t,
            local_cu_cpu=local_cu_cpu,
            half_lengths_cpu=half_lengths_cpu,
            orig_full_lengths_cpu=orig_lengths_cpu,
            padded_full_cu_d=padded_full_cu,  # reuse caller's tensor as-is
            orig_full_lengths_d=orig_full_lengths_d,
            local_padding_mask=local_padding_mask,
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
        """CP prefill path (zigzag): each rank computes on its own zigzag tokens.
        No QKV all-gather needed. Only SSM state affine pairs are communicated."""
        gdn = self.prefill_gdn
        cp_size = self.parallelism_config.tp_size
        cp_rank = self.parallelism_config.tp_rank
        cp_group = _get_group(Group.TP)
        context_batch_size = attention_inputs.input_lengths.shape[0]
        chunk_align = attn_meta.cp_chunk_align_info
        chunk_size = chunk_align.chunk_size
        local_cu = chunk_align.local_cu

        kv_cache_tensor: Optional[torch.Tensor] = None
        seq_size_per_block = 1
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block

        # ---- Phase 0: Conv1d with context exchange ----
        ctx_len = gdn.linear_conv_kernel_dim - 1
        local_cu_cpu = chunk_align.local_cu_cpu
        half_lengths_cpu = chunk_align.half_lengths_cpu

        seg0_ctx, seg1_ctx = exchange_conv_context(
            mixed_qkv.transpose(0, 1),  # [dim, tokens]
            local_cu_cpu,
            half_lengths_cpu,
            cp_rank,
            cp_size,
            cp_group,
            ctx_len=ctx_len,
        )
        # seg0_ctx is None for rank 0 — kernel reads from block cache via prefix_lengths

        padded_qkv, padded_seg_cu, padded_seg_cu_cpu = prepend_conv_context(
            mixed_qkv.transpose(0, 1),
            local_cu_cpu,
            half_lengths_cpu,
            seg0_ctx,
            seg1_ctx,
            ctx_len=ctx_len,
        )

        padded_batch = len(padded_seg_cu_cpu) - 1
        if not hasattr(attn_meta, "_cp_conv1d_meta"):
            attn_meta._cp_conv1d_meta = prepare_causal_conv1d_metadata(
                query_start_loc=padded_seg_cu.int(), device=mixed_qkv.device
            )
            attn_meta._cp_padded_seg_cu = padded_seg_cu
            attn_meta._cp_padded_seg_cu_cpu = padded_seg_cu_cpu
            attn_meta._cp_padded_batch = padded_batch

            # Build 2*batch prefix_lengths and block_map for conv1d kernel
            # Only rank 0's seg0 (even indices) needs prefix reading from block cache
            padded_prefix = torch.zeros(
                padded_batch, dtype=torch.int32, device=mixed_qkv.device
            )
            max_blocks = (
                attention_inputs.kv_cache_kernel_block_id_device.shape[1]
                if kv_cache_tensor is not None
                else 0
            )
            padded_block_map = torch.zeros(
                padded_batch,
                max(max_blocks, 1),
                dtype=torch.int32,
                device=mixed_qkv.device,
            )
            if cp_rank == 0 and kv_cache_tensor is not None:
                padded_prefix[0::2] = attention_inputs.prefix_lengths_d
                padded_block_map[0::2] = (
                    attention_inputs.kv_cache_kernel_block_id_device
                )
            attn_meta._cp_padded_prefix = padded_prefix
            attn_meta._cp_padded_block_map = padded_block_map
        else:
            padded_seg_cu = attn_meta._cp_padded_seg_cu
            padded_seg_cu_cpu = attn_meta._cp_padded_seg_cu_cpu
            padded_batch = attn_meta._cp_padded_batch

        conv_states = (
            gdn._get_conv_states(kv_cache_tensor).transpose(1, 2)
            if kv_cache_tensor is not None
            else None
        )
        conv_out = causal_conv1d_fn(
            x=padded_qkv,
            weight=gdn.conv_weights,
            bias=None,
            conv_states=conv_states,
            query_start_loc=padded_seg_cu.int(),
            block_map=attn_meta._cp_padded_block_map,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attn_meta._cp_padded_prefix,
            metadata=attn_meta._cp_conv1d_meta,
        )
        local_mixed_qkv = strip_conv_context(
            conv_out,
            local_cu_cpu,
            half_lengths_cpu,
            padded_seg_cu_cpu,
            seg0_has_ctx=(seg0_ctx is not None),
            local_total=chunk_align.local_total,
            ctx_len=ctx_len,
        ).transpose(
            0, 1
        )  # [tokens, dim]

        local_padding_mask = chunk_align.local_padding_mask
        if local_padding_mask is not None:
            local_mixed_qkv = local_mixed_qkv * local_padding_mask.unsqueeze(-1)

        g, beta = fused_gdn_gating(gdn.alog, a, b, gdn.dt_bias)
        if local_padding_mask is not None:
            # g=0 → exp(g)=1 (state preserved); beta=0 → no update contribution.
            mask_f = local_padding_mask.to(g.dtype)
            g = g * mask_f.unsqueeze(-1)
            beta = beta * mask_f.unsqueeze(-1)

        query, key, value = torch.split(
            local_mixed_qkv,
            [
                gdn.local_num_k_heads * gdn.head_k_dim,
                gdn.local_num_k_heads * gdn.head_k_dim,
                gdn.local_num_v_heads * gdn.head_v_dim,
            ],
            dim=-1,
        )
        query = query.view(1, -1, gdn.local_num_k_heads, gdn.head_k_dim).contiguous()
        key = key.view(1, -1, gdn.local_num_k_heads, gdn.head_k_dim).contiguous()
        value = value.view(1, -1, gdn.local_num_v_heads, gdn.head_v_dim).contiguous()

        ssm_states = (
            gdn._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        initial_states: Optional[torch.Tensor] = None
        if ssm_states is not None:
            initial_states = torch.empty(
                context_batch_size,
                gdn.local_num_v_heads,
                gdn.head_v_dim,
                gdn.head_k_dim,
                device=mixed_qkv.device,
                dtype=gdn.ssm_state_dtype,
            )
            load_initial_state_from_block_map(
                attention_inputs.prefix_lengths_d,
                attention_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                initial_states,
                seq_size_per_block,
            )

        o, h, final_state = chunk_gated_delta_rule_fwd_cp_zigzag(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=initial_states,
            output_final_state=(ssm_states is not None),
            cp_group=cp_group,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=local_cu,
            seg_cu=chunk_align.seg_cu,
            causal_order=chunk_align.causal_order,
        )

        local_attn_out = o.squeeze(0)  # [local_total, H, V]

        if ssm_states is not None:
            gathered_h, h_work = all_gather_async(h, group=Group.TP)
            fs_work = broadcast_async(final_state, src=cp_size - 1, group=Group.TP)
        if kv_cache_tensor is not None:
            gathered_qkv, qkv_work = all_gather_async(
                mixed_qkv.contiguous(), group=Group.TP
            )

        local_attn_out = self.norm(
            local_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        local_attn_out = local_attn_out.reshape(
            -1, self.local_num_v_heads * self.head_v_dim
        )
        local_attn_out = self.out_proj(local_attn_out)

        # ---- Wait for h all-gather and store SSM states ----
        if ssm_states is not None:
            h_work.wait()
            fs_work.wait()

            chunk_tail = h.shape[2:]  # (H, K, V)
            gathered_flat = gathered_h.reshape(-1, *chunk_tail)
            full_h = gathered_flat[chunk_align.h_causal_indices].unsqueeze(0)

            store_ssm_state_to_block_map(
                full_h,
                final_state,
                attention_inputs.prefix_lengths_d,
                chunk_align.orig_full_cu,
                attention_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                seq_size_per_block,
                chunk_size=chunk_size,
            )

        # ---- Wait for qkv all-gather and store conv states (single-card semantic) ----
        if kv_cache_tensor is not None:
            qkv_work.wait()

            restore_indices = attn_meta.cp_restore_indices
            full_padded_mixed_qkv = gathered_qkv[
                restore_indices
            ]  # [total_padded, qkv_dim]

            conv_states_cache = gdn._get_conv_states(kv_cache_tensor).transpose(1, 2)

            # Per-request: cache max_writes_per_seq for grid sizing + assert prefix
            # block-aligned (matches single-card causal_conv1d.py:342 assumption).
            B = seq_size_per_block
            if not hasattr(attn_meta, "_cp_conv_max_writes"):
                orig_lengths_cpu = chunk_align.orig_full_lengths_cpu
                prefix_lengths_cpu = attention_inputs.prefix_lengths.tolist()
                for b in range(context_batch_size):
                    assert prefix_lengths_cpu[b] % B == 0, (
                        f"prefix_length must be a multiple of seq_size_per_block "
                        f"(got prefix={prefix_lengths_cpu[b]}, block={B})"
                    )
                attn_meta._cp_conv_max_writes = max(
                    (N + B - 1) // B for N in orig_lengths_cpu
                )

            store_conv_state_to_block_map(
                full_qkv=full_padded_mixed_qkv.T,  # [qkv_dim, total_padded]
                conv_states=conv_states_cache,
                prefix_lengths_d=attention_inputs.prefix_lengths_d,
                input_lengths_d=chunk_align.orig_full_lengths_d,
                padded_cu_d=chunk_align.padded_full_cu_d,
                block_map_d=attention_inputs.kv_cache_kernel_block_id_device,
                seq_size_per_block=B,
                max_writes_per_seq=attn_meta._cp_conv_max_writes,
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

        # C++ handleInputs already padded each sequence to (2 * cp_size * chunk_size);
        # prefill_cp_chunk_lengths is per-rank padded chunk length, so
        # padded_full_length = chunk_length * cp_size. No re-padding on Python side.
        full_new_lengths = cp_info.prefill_actual_input_lengths_cpu  # orig
        padded_chunk_lengths = cp_info.prefill_cp_chunk_lengths  # per-rank padded
        full_cu = torch.zeros(
            full_new_lengths.shape[0] + 1, dtype=torch.int32, device=device
        )
        full_cu[1:] = full_new_lengths.cumsum(0).to(device)
        padded_full_cu = torch.zeros(
            full_new_lengths.shape[0] + 1, dtype=torch.int32, device=device
        )
        padded_full_cu[1:] = (padded_chunk_lengths.to(device).long() * cp_size).cumsum(
            0
        )
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

        need_pad = int(full_new_lengths.sum().item()) != total_ag

        chunk_align = CpChunkAlignInfo.build(
            cp_size=cp_size,
            cp_rank=cp_rank,
            device=device,
            orig_lengths_cpu=full_new_lengths.tolist(),
            padded_full_cu=padded_full_cu,
            local_padding_mask=cp_local_valid_mask if need_pad else None,
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
