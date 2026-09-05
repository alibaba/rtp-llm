import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.qwen3_next.constants import GDN_STATE_CHUNK_SIZE
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    all_reduce,
    broadcast_from_group_rank,
)
from rtp_llm.models_py.model_desc.block_map import (
    get_group_tags_for_layers,
    get_primary_attention_inputs,
    select_attention_inputs_for_layer,
    select_fmha_impl_for_layer,
)
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
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.linear_attn_utils import (
    ZigzagCPPlan,
    get_segment_valid_lengths,
)
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
from rtp_llm.utils.swizzle_utils import (
    can_fuse_swizzled_kn,
    should_swizzle_linear_attn_ba,
)
from rtp_llm.utils.util import to_torch_dtype

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _warn_qkvz_ba_swizzle_fallback(
    qkvz_shape: tuple[int, ...], ba_shape: tuple[int, ...]
) -> None:
    logger.warning(
        "Disabling Qwen3Next qkvz+ba fusion because ROCm swizzle cannot safely "
        "combine qkvz shape %s with ba shape %s; using separate swizzled qkvz "
        "and unswizzled ba GEMMs",
        qkvz_shape,
        ba_shape,
    )


@dataclass
class Qwen3NextMetadata:
    prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None
    is_target_verify: bool = False
    full_prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None
    full_prefill_cu_seqlens: Optional[torch.Tensor] = None
    cp_plan: Optional[ZigzagCPPlan] = None
    cp_segment_valid_lengths: Optional[tuple[int, ...]] = None
    cp_local_conv1d_meta: Optional[CausalConv1dMetadata] = None
    cp_local_conv_cu_seqlens: Optional[torch.Tensor] = None
    cp_local_conv_prefix_lengths: Optional[torch.Tensor] = None
    cp_unpad_restore_indices: Optional[torch.Tensor] = None
    cp_local_extract_indices: Optional[torch.Tensor] = None
    cp_local_valid_mask: Optional[torch.Tensor] = None

    def get_prefill_conv1d_meta(self) -> Optional[CausalConv1dMetadata]:
        return self.prefill_conv1d_meta

    @property
    def is_cp_linear_attn(self) -> bool:
        return self.cp_plan is not None

    def prepare_cp_fallback_metadata(
        self, attention_inputs: PyAttentionInputs, device: torch.device
    ) -> None:
        if self.full_prefill_cu_seqlens is not None:
            return
        if self.cp_plan is None:
            raise RuntimeError("CP fallback metadata requires a CP plan")

        cp_info = attention_inputs.context_parallel_info
        full_new_lengths = cp_info.prefill_actual_input_lengths_cpu
        full_cu = torch.zeros(
            full_new_lengths.shape[0] + 1, dtype=torch.int32, device=device
        )
        full_cu[1:] = full_new_lengths.cumsum(0).to(device)

        restore_indices = cp_info.prefill_qkv_restore_indice
        padding_mask = cp_info.prefill_qkv_padding_mask
        unpad_restore = restore_indices[padding_mask == 1]
        total_ag = padding_mask.shape[0]
        local_chunk_total = total_ag // self.cp_plan.cp_size
        local_start = self.cp_plan.cp_rank * local_chunk_total

        inv_restore = torch.full((total_ag,), -1, dtype=torch.long, device=device)
        inv_restore[unpad_restore.long()] = torch.arange(
            unpad_restore.shape[0], device=device
        )
        local_inv = inv_restore[local_start : local_start + local_chunk_total]

        self.full_prefill_cu_seqlens = full_cu
        self.full_prefill_conv1d_meta = prepare_causal_conv1d_metadata(
            query_start_loc=full_cu, device=device
        )
        self.cp_unpad_restore_indices = unpad_restore
        self.cp_local_valid_mask = local_inv >= 0
        self.cp_local_extract_indices = local_inv[self.cp_local_valid_mask]


def _write_cp_cache_store(
    attention_inputs: PyAttentionInputs, kv_cache: LayerKVCache
) -> None:
    """Write a CP linear layer using that layer's tag-local cache metadata."""
    cache_store_inputs = attention_inputs.cache_store_inputs
    cache_store_writer = attention_inputs.cache_store_writer
    if cache_store_inputs is None or cache_store_writer is None:
        return
    cache_store_writer.write(cache_store_inputs, kv_cache)


def _maybe_write_cp_cache_store(
    attention_inputs: PyAttentionInputs,
    kv_cache: Optional[LayerKVCache],
    attn_meta: Qwen3NextMetadata,
) -> None:
    """Keep CacheStore writes on the CP linear-attention path only."""
    if kv_cache is None or not attn_meta.is_cp_linear_attn:
        return
    _write_cp_cache_store(attention_inputs, kv_cache)


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
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
            query_start_loc=(
                attn_inputs.cu_seqlens_device if cu_seqlens is None else cu_seqlens
            ),
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attn_inputs.prefix_lengths_device,
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
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g, beta = fused_gdn_gating(self.alog, a, b, self.dt_bias)
        ssm_states = (
            self._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        context_batch_size = attn_inputs.input_lengths.shape[0]
        cu_seqlens_without_padding = (
            attn_inputs.cu_seqlens_device if cu_seqlens is None else cu_seqlens
        )
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
                attn_inputs.prefix_lengths_device,
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
                    attn_inputs.prefix_lengths_device
                    if ssm_states is not None
                    else None
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
                attn_inputs.prefix_lengths_device,
                cu_seqlens_without_padding,
                attn_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                seq_size_per_block,
                chunk_size=GDN_STATE_CHUNK_SIZE,
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
        cache_store_inputs = attn_inputs.cache_store_inputs
        cache_store_writer = attn_inputs.cache_store_writer
        if (
            kv_cache is not None
            and cache_store_inputs is not None
            and cache_store_writer is not None
        ):
            cache_store_writer.write(cache_store_inputs, kv_cache)
        return attn_out


class Qwen3NextGatedDeltaNetDecode(Qwen3NextGatedDeltaNetBase):
    def _get_fla_block_map(self, attn_inputs: PyAttentionInputs) -> torch.Tensor:
        block_map = attn_inputs.kv_cache_kernel_block_id_device
        if (
            attn_inputs.is_cuda_graph
            and block_map is not None
            and block_map.ndim == 2
            and block_map.shape[1] > 1
        ):
            # CUDA graph capture allocates a fixed-width block table, while the
            # recurrent FLA decode kernel consumes only the first logical block.
            # Keep the original row stride in this narrow view: FLA receives it
            # explicitly and uses it to advance between batch rows.
            return block_map[:, :1]
        return block_map

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
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_device,
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
            block_map=self._get_fla_block_map(attn_inputs),
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_device,
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
        b, s = (
            attention_inputs.prefix_lengths.size(0),
            token // attention_inputs.prefix_lengths.size(0),
        )
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
    ):
        super().__init__(
            attn_config,
            parallelism_config,
            weights,
            layernorm_eps,
            quant_config,
            hw_kernel_config=hw_kernel_config,
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
    _linear_cp_fatal_reasons = frozenset(
        {
            "missing_prefix_cache",
            "unaligned_internal_prefix",
            "missing_prefix_block_map",
            "prefix_block_out_of_range",
            "missing_prefix_boundary_state",
        }
    )

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
        # Fall back to two GEMMs when qkvz is quantized, or when ROCm swizzle
        # cannot represent both source weights and their fused output layout.
        qkvz_w = weights[W.linear_attn_qkvz_w]
        ba_w = weights[W.linear_attn_ba_w]
        qkvz_is_bf16 = weights.get(W.linear_attn_qkvz_s) is None
        _is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        rocm_swizzle_enabled = (
            _is_rocm and hw_kernel_config is not None and hw_kernel_config.use_swizzleA
        )
        rocm_fused_layout_safe = not rocm_swizzle_enabled or can_fuse_swizzled_kn(
            qkvz_w, ba_w
        )
        self._qkvz_ba_fused = qkvz_is_bf16 and rocm_fused_layout_safe
        if qkvz_is_bf16 and rocm_swizzle_enabled and not rocm_fused_layout_safe:
            _warn_qkvz_ba_swizzle_fallback(tuple(qkvz_w.shape), tuple(ba_w.shape))

        if self._qkvz_ba_fused:
            self._qkvz_size = qkvz_w.shape[1]
            self._ba_size = ba_w.shape[1]
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
            # BA stays BF16 when qkvz is quantized. Keep runtime dispatch paired
            # with the loader's shape-based layout decision: aligned TP-local
            # BA uses WithSwizzle; unaligned BA uses NoSwizzle.
            ba_hw_kernel_config = hw_kernel_config
            if rocm_swizzle_enabled and not should_swizzle_linear_attn_ba(ba_w):
                ba_hw_kernel_config = None
            self.in_proj_ba = LinearFactory.create_linear_from_weights(
                weights,
                W.linear_attn_ba_w,
                None,
                None,
                quant_config,
                hw_kernel_config=ba_hw_kernel_config,
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

    def _get_linear_cp_relay_fallback_reason(
        self,
        attention_inputs: PyAttentionInputs,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_meta: Qwen3NextMetadata,
    ) -> Optional[str]:
        """Return why the linear-attention CP relay cannot run, if applicable."""
        if self.parallelism_config.prefill_cp_config.kv_cache_sharded:
            return "kv_cache_sharded"
        if attention_inputs.input_lengths.shape[0] != 1:
            return "unsupported_context_batch_size"
        if attention_inputs.is_cuda_graph:
            return "cuda_graph"

        prefix_len = int(attention_inputs.prefix_lengths[0].item())
        if prefix_len > 0:
            block_map = attention_inputs.kv_cache_kernel_block_id
            if kv_cache_tensor is None:
                return "missing_prefix_cache"
            if prefix_len % seq_size_per_block != 0:
                return "unaligned_internal_prefix"
            if block_map is None:
                return "missing_prefix_block_map"
            prefix_block_pos = prefix_len // seq_size_per_block - 1
            if prefix_block_pos >= block_map.shape[1]:
                return "prefix_block_out_of_range"
            if int(block_map[0, prefix_block_pos].item()) <= 0:
                return "missing_prefix_boundary_state"
        if (
            attn_meta.cp_segment_valid_lengths is None
            or attn_meta.cp_local_valid_mask is None
            or attn_meta.cp_local_conv1d_meta is None
            or attn_meta.cp_local_conv_cu_seqlens is None
            or attn_meta.cp_local_conv_prefix_lengths is None
        ):
            return "missing_local_conv_metadata"
        return None

    @classmethod
    def _raise_for_invalid_cp_state(
        cls, reason: Optional[str], prefix_len: int
    ) -> None:
        if reason not in cls._linear_cp_fatal_reasons:
            return
        raise RuntimeError(
            "Qwen3.5 CP prefill invariant violated: "
            f"reason={reason}, prefix_len={prefix_len}. "
            "The general path cannot safely reconstruct this input."
        )

    def _run_linear_cp_gdn_segment(
        self,
        mixed_qkv: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one contiguous GDN segment from an explicit FP32 boundary state."""
        gdn = self.prefill_gdn
        if mixed_qkv.shape[0] >= 2048 and gdn.head_k_dim == gdn.head_v_dim:
            query, key, value = scatter_qkv(
                mixed_qkv,
                gdn.local_num_k_heads,
                gdn.local_num_v_heads,
                gdn.head_k_dim,
                gdn.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv,
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

        cu_seqlens = torch.tensor(
            [0, mixed_qkv.shape[0]], dtype=torch.int32, device=mixed_qkv.device
        )
        attn_out, chunk_states, final_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        return attn_out.squeeze_(0), chunk_states, final_state

    @staticmethod
    def _get_linear_cp_cache_blocks(
        attention_inputs: PyAttentionInputs,
        seq_size_per_block: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return new-token block ends and their layer-local cache block IDs."""
        new_len = int(
            attention_inputs.context_parallel_info.prefill_actual_input_lengths_cpu[
                0
            ].item()
        )
        prefix_len = int(attention_inputs.prefix_lengths[0].item())
        total_len = prefix_len + new_len
        device = attention_inputs.kv_cache_kernel_block_id_device.device
        first_full_block_end = prefix_len + seq_size_per_block
        absolute_block_ends = torch.arange(
            first_full_block_end,
            max(first_full_block_end, total_len),
            seq_size_per_block,
            dtype=torch.long,
            device=device,
        )
        absolute_block_ends = torch.cat(
            [
                absolute_block_ends,
                torch.tensor([total_len], dtype=torch.long, device=device),
            ]
        )
        block_positions = (absolute_block_ends - 1) // seq_size_per_block
        block_ids = attention_inputs.kv_cache_kernel_block_id_device[
            0, block_positions
        ].long()
        return absolute_block_ends - prefix_len, block_ids

    @staticmethod
    def _copy_linear_cp_prefix_ssm_state(
        ssm_states: torch.Tensor, prefix_block_id: int
    ) -> torch.Tensor:
        """Copy a cached boundary into the FP32 relay buffer."""
        return (
            ssm_states[prefix_block_id].to(dtype=torch.float32, copy=True).unsqueeze(0)
        )

    @staticmethod
    def _sync_linear_cp_cache_states(
        local_states: torch.Tensor,
        cache_states: torch.Tensor,
        block_ids: torch.Tensor,
    ) -> None:
        """Merge rank-owned states and write valid allocated cache blocks."""
        local_states = all_reduce(local_states, group=Group.TP)
        valid_positions = torch.nonzero(block_ids > 0, as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            return
        cache_states.index_copy_(
            0,
            block_ids[valid_positions],
            local_states[valid_positions].to(cache_states.dtype),
        )

    @staticmethod
    def _record_linear_cp_segment_ssm_states(
        local_block_states: torch.Tensor,
        block_ends: torch.Tensor,
        segment_start: int,
        segment_end: int,
        chunk_states: torch.Tensor,
        final_state: torch.Tensor,
    ) -> None:
        """Record cache boundaries owned by one chunk-aligned segment."""
        block_positions = torch.nonzero(
            (block_ends > segment_start) & (block_ends <= segment_end),
            as_tuple=False,
        ).flatten()
        if block_positions.numel() == 0:
            return

        offsets = block_ends[block_positions] - segment_start
        chunk_indices = offsets // GDN_STATE_CHUNK_SIZE
        selected_states = chunk_states[
            0, chunk_indices.clamp_max(chunk_states.shape[1] - 1)
        ]
        ends_at_segment = offsets == segment_end - segment_start
        selected_states = torch.where(
            ends_at_segment[:, None, None, None], final_state[0], selected_states
        )
        local_block_states.index_copy_(
            0, block_positions, selected_states.to(local_block_states.dtype)
        )

    @staticmethod
    def _record_linear_cp_segment_conv_states(
        local_block_states: torch.Tensor,
        block_ends: torch.Tensor,
        segment_start: int,
        segment_valid_length: int,
        segment_with_halo: torch.Tensor,
    ) -> None:
        """Record real conv tails, using the predecessor halo at segment start."""
        if segment_valid_length == 0:
            return
        segment_end = segment_start + segment_valid_length
        block_positions = torch.nonzero(
            (block_ends > segment_start) & (block_ends <= segment_end),
            as_tuple=False,
        ).flatten()
        if block_positions.numel() == 0:
            return

        state_len = local_block_states.shape[1]
        offsets = block_ends[block_positions] - segment_start
        state_offsets = torch.arange(
            state_len, dtype=torch.long, device=segment_with_halo.device
        )
        state_indices = offsets[:, None] + state_offsets[None, :]
        local_block_states.index_copy_(
            0, block_positions, segment_with_halo[state_indices]
        )

    def _forward_linear_cp_conv(
        self,
        mixed_qkv: torch.Tensor,
        prefix_conv_state: Optional[torch.Tensor],
        kv_cache_tensor: Optional[torch.Tensor],
        cache_block_ends: Optional[torch.Tensor],
        cache_block_ids: Optional[torch.Tensor],
        attn_meta: Qwen3NextMetadata,
        cp_plan: ZigzagCPPlan,
        segment_valid_lengths: tuple[int, ...],
    ) -> torch.Tensor:
        """Run local causal conv using fixed-size zigzag predecessor halos."""
        gdn = self.prefill_gdn
        state_len = gdn.linear_conv_kernel_dim - 1
        segment_tokens = mixed_qkv.shape[0] // 2
        local_segments = mixed_qkv.reshape(2, segment_tokens, -1)
        local_tails = local_segments[:, -state_len:, :].contiguous()

        gathered_tails = all_gather(local_tails.flatten(0, 1), group=Group.TP).reshape(
            cp_plan.cp_size, 2, state_len, -1
        )
        front_source, back_source = cp_plan.halo_sources
        first_halo = (
            prefix_conv_state
            if prefix_conv_state is not None
            else torch.zeros_like(local_tails[0])
        )
        if front_source is not None:
            first_halo = gathered_tails[front_source]
        second_halo = gathered_tails[back_source]
        local_segment_halos = first_halo, second_halo

        haloed_qkv = torch.cat(
            [first_halo, local_segments[0], second_halo, local_segments[1]], dim=0
        )
        haloed_out = causal_conv1d_fn(
            x=haloed_qkv.transpose(0, 1),
            weight=gdn.conv_weights,
            bias=None,
            conv_states=None,
            query_start_loc=attn_meta.cp_local_conv_cu_seqlens,
            block_map=None,
            prefix_lengths=attn_meta.cp_local_conv_prefix_lengths,
            seq_size_per_block=1,
            metadata=attn_meta.cp_local_conv1d_meta,
        ).transpose(0, 1)

        haloed_segment_tokens = segment_tokens + state_len
        local_out = torch.cat(
            [
                haloed_out[state_len:haloed_segment_tokens],
                haloed_out[
                    haloed_segment_tokens + state_len : 2 * haloed_segment_tokens
                ],
            ],
            dim=0,
        )

        if (
            kv_cache_tensor is not None
            and cache_block_ends is not None
            and cache_block_ids is not None
        ):
            conv_states = gdn._get_conv_states(kv_cache_tensor)
            local_block_states = conv_states.new_zeros(
                cache_block_ends.shape[0], state_len, mixed_qkv.shape[-1]
            )
            for local_segment_id, global_segment_id in enumerate(
                cp_plan.local_global_segments
            ):
                segment_with_halo = torch.cat(
                    [
                        local_segment_halos[local_segment_id],
                        local_segments[local_segment_id],
                    ],
                    dim=0,
                )
                self._record_linear_cp_segment_conv_states(
                    local_block_states,
                    cache_block_ends,
                    global_segment_id * segment_tokens,
                    segment_valid_lengths[global_segment_id],
                    segment_with_halo,
                )
            self._sync_linear_cp_cache_states(
                local_block_states, conv_states, cache_block_ids
            )
        return local_out

    def _forward_linear_cp_relay(
        self,
        local_mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        prefix_ssm_state: Optional[torch.Tensor],
        kv_cache_tensor: Optional[torch.Tensor],
        cache_block_ends: Optional[torch.Tensor],
        cache_block_ids: Optional[torch.Tensor],
        cp_plan: ZigzagCPPlan,
        segment_valid_lengths: tuple[int, ...],
        local_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute each GDN token once using ordered FP32 state broadcasts."""
        gdn = self.prefill_gdn
        g, beta = fused_gdn_gating(gdn.alog, a, b, gdn.dt_bias)
        local_tokens = local_mixed_qkv.shape[0]
        segment_tokens = local_tokens // 2
        rank = cp_plan.cp_rank
        state = (
            prefix_ssm_state
            if prefix_ssm_state is not None
            else torch.zeros(
                1,
                gdn.local_num_v_heads,
                gdn.head_v_dim,
                gdn.head_k_dim,
                dtype=torch.float32,
                device=local_mixed_qkv.device,
            )
        )
        local_attn_out = torch.zeros(
            local_tokens,
            gdn.local_num_v_heads,
            gdn.head_v_dim,
            dtype=local_mixed_qkv.dtype,
            device=local_mixed_qkv.device,
        )
        ssm_states = (
            gdn._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        local_block_states = (
            ssm_states.new_zeros(cache_block_ends.shape[0], *ssm_states.shape[1:])
            if ssm_states is not None and cache_block_ends is not None
            else None
        )

        relay_steps = cp_plan.relay_steps
        ub_communicator = None
        if getattr(self.parallelism_config, "use_ub_comm", False):
            from rtp_llm.models_py.distributed.user_buffers import (
                get_user_buffers_communicator,
            )

            ub_communicator = get_user_buffers_communicator()
        use_ub_relay = (
            ub_communicator is not None
            and ub_communicator.group_size == cp_plan.cp_size
            and ub_communicator.can_handle_tensor(state)
        )
        for step_index, step in enumerate(relay_steps):
            valid_tokens = step.valid_token_count(segment_valid_lengths)
            if rank == step.owner_rank and valid_tokens > 0:
                local_start = step.first_local_segment * segment_tokens
                local_end = local_start + valid_tokens
                global_start = step.first_global_segment * segment_tokens
                local_attn_out[local_start:local_end], chunk_states, state = (
                    self._run_linear_cp_gdn_segment(
                        local_mixed_qkv[local_start:local_end],
                        g[:, local_start:local_end].contiguous(),
                        beta[:, local_start:local_end].contiguous(),
                        state,
                    )
                )
                if local_block_states is not None:
                    self._record_linear_cp_segment_ssm_states(
                        local_block_states,
                        cache_block_ends,
                        global_start,
                        global_start + valid_tokens,
                        chunk_states,
                        state,
                    )
            if step_index + 1 >= len(relay_steps):
                continue

            next_step = relay_steps[step_index + 1]
            if use_ub_relay:
                if rank == step.owner_rank:
                    if not ub_communicator.send(state, dst=next_step.owner_rank):
                        raise RuntimeError(
                            "Qwen3.5 CP relay state does not fit the user-buffer "
                            "communication capacity"
                        )
                elif rank == next_step.owner_rank:
                    if not ub_communicator.recv(state, src=step.owner_rank):
                        raise RuntimeError(
                            "Qwen3.5 CP relay state receive failed in user-buffer "
                            "communication"
                        )
            else:
                broadcast_from_group_rank(state, src=step.owner_rank, group=Group.TP)

        if (
            local_block_states is not None
            and ssm_states is not None
            and cache_block_ids is not None
        ):
            self._sync_linear_cp_cache_states(
                local_block_states, ssm_states, cache_block_ids
            )

        local_attn_out = self.norm(
            local_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        local_attn_out = local_attn_out.reshape(
            -1, self.local_num_v_heads * self.head_v_dim
        )
        local_attn_out = self.out_proj(local_attn_out)
        return torch.where(
            local_valid_mask[:, None],
            local_attn_out,
            torch.zeros_like(local_attn_out),
        )

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
        """Run partitioned linear attention, with full gather as a fallback."""
        gdn = self.prefill_gdn
        kv_cache_tensor: Optional[torch.Tensor] = None
        seq_size_per_block = 1
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block

        fallback_reason = self._get_linear_cp_relay_fallback_reason(
            attention_inputs,
            kv_cache_tensor,
            seq_size_per_block,
            attn_meta,
        )
        if fallback_reason is None:
            cp_plan = attn_meta.cp_plan
            segment_valid_lengths = attn_meta.cp_segment_valid_lengths
            local_valid_mask = attn_meta.cp_local_valid_mask
            if (
                cp_plan is None
                or segment_valid_lengths is None
                or local_valid_mask is None
            ):
                raise RuntimeError("CP relay metadata is incomplete")

            cache_block_ends: Optional[torch.Tensor] = None
            cache_block_ids: Optional[torch.Tensor] = None
            prefix_conv_state: Optional[torch.Tensor] = None
            prefix_ssm_state: Optional[torch.Tensor] = None
            if kv_cache_tensor is not None:
                cache_block_ends, cache_block_ids = self._get_linear_cp_cache_blocks(
                    attention_inputs, seq_size_per_block
                )
                prefix_len = int(attention_inputs.prefix_lengths[0].item())
                if prefix_len > 0:
                    prefix_block_pos = prefix_len // seq_size_per_block - 1
                    prefix_block_id = int(
                        attention_inputs.kv_cache_kernel_block_id[
                            0, prefix_block_pos
                        ].item()
                    )
                    prefix_conv_state = self.prefill_gdn._get_conv_states(
                        kv_cache_tensor
                    )[prefix_block_id]
                    prefix_ssm_state = self._copy_linear_cp_prefix_ssm_state(
                        self.prefill_gdn._get_ssm_states(kv_cache_tensor),
                        prefix_block_id,
                    )
            local_mixed_qkv = self._forward_linear_cp_conv(
                mixed_qkv,
                prefix_conv_state,
                kv_cache_tensor,
                cache_block_ends,
                cache_block_ids,
                attn_meta,
                cp_plan,
                segment_valid_lengths,
            )
            local_attn_out = self._forward_linear_cp_relay(
                local_mixed_qkv,
                z,
                b,
                a,
                prefix_ssm_state,
                kv_cache_tensor,
                cache_block_ends,
                cache_block_ids,
                cp_plan,
                segment_valid_lengths,
                local_valid_mask,
            )
            _maybe_write_cp_cache_store(attention_inputs, kv_cache, attn_meta)
            return local_attn_out

        self._raise_for_invalid_cp_state(
            fallback_reason, int(attention_inputs.prefix_lengths[0].item())
        )
        attn_meta.prepare_cp_fallback_metadata(attention_inputs, mixed_qkv.device)
        full_cu = attn_meta.full_prefill_cu_seqlens
        full_conv_meta = attn_meta.full_prefill_conv1d_meta
        unpad_restore = attn_meta.cp_unpad_restore_indices
        local_extract_indices = attn_meta.cp_local_extract_indices
        local_valid_mask = attn_meta.cp_local_valid_mask
        if (
            full_cu is None
            or full_conv_meta is None
            or unpad_restore is None
            or local_extract_indices is None
            or local_valid_mask is None
        ):
            raise RuntimeError("CP fallback metadata is incomplete")

        packed = torch.cat([mixed_qkv, b, a], dim=-1)
        full_packed = all_gather(packed, group=Group.TP)[unpad_restore]
        qkv_dim = mixed_qkv.shape[-1]
        b_dim = b.shape[-1]
        full_mixed_qkv = full_packed[:, :qkv_dim].contiguous()
        full_b = full_packed[:, qkv_dim : qkv_dim + b_dim].contiguous()
        full_a = full_packed[:, qkv_dim + b_dim :].contiguous()

        full_mixed_qkv = gdn._conv1d(
            full_mixed_qkv,
            kv_cache_tensor,
            seq_size_per_block,
            attention_inputs,
            metadata=full_conv_meta,
            cu_seqlens=full_cu,
        )
        full_attn_out = gdn._fla(
            full_mixed_qkv,
            full_b,
            full_a,
            kv_cache_tensor,
            seq_size_per_block,
            attention_inputs,
            cu_seqlens=full_cu,
        )
        _maybe_write_cp_cache_store(attention_inputs, kv_cache, attn_meta)

        local_attn_out = torch.zeros(
            z.shape[0],
            *full_attn_out.shape[1:],
            device=full_attn_out.device,
            dtype=full_attn_out.dtype,
        )
        local_attn_out[local_valid_mask] = full_attn_out[local_extract_indices]
        local_attn_out = self.norm(
            local_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        local_attn_out = local_attn_out.reshape(
            -1, self.local_num_v_heads * self.head_v_dim
        )
        return self.out_proj(local_attn_out)

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

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        if self.kv_cache is None:
            return None
        full_attention_layers = (
            layer_idx
            for layer_idx, layer in enumerate(self.layers)
            if layer.layer_type != HybridAttentionType.LINEAR
        )
        return get_group_tags_for_layers(self.kv_cache, full_attention_layers)

    def _build_cp_linear_attn_metadata(
        self,
        attention_inputs: PyAttentionInputs,
        device: torch.device,
    ) -> Qwen3NextMetadata:
        """Build request-level metadata for the partitioned linear-attention path."""
        cp_info = attention_inputs.context_parallel_info
        if cp_info is None:
            return Qwen3NextMetadata()

        cp_plan = ZigzagCPPlan(
            cp_size=self.parallelism_config.tp_size,
            cp_rank=self.parallelism_config.tp_rank,
        )
        full_new_lengths = cp_info.prefill_actual_input_lengths_cpu
        local_chunk_total = cp_info.prefill_qkv_padding_mask.shape[0] // cp_plan.cp_size

        segment_valid_lengths = None
        local_valid_mask = None
        local_conv1d_meta = None
        local_conv_cu_seqlens = None
        local_conv_prefix_lengths = None
        if full_new_lengths.shape[0] == 1:
            segment_tokens = local_chunk_total // 2
            if local_chunk_total % 2 == 0:
                actual_tokens = int(full_new_lengths[0].item())
                segment_valid_lengths = get_segment_valid_lengths(
                    actual_tokens, segment_tokens, cp_plan.cp_size
                )
                local_segment_lengths = tuple(
                    segment_valid_lengths[segment_id]
                    for segment_id in cp_plan.local_global_segments
                )
                local_valid_mask = torch.cat(
                    [
                        torch.arange(segment_tokens, device=device) < valid_length
                        for valid_length in local_segment_lengths
                    ]
                )

                state_len = (
                    self.config.linear_attention_config.linear_conv_kernel_dim - 1
                )
                if segment_tokens >= state_len:
                    haloed_segment_tokens = segment_tokens + state_len
                    local_conv_cu_seqlens = torch.tensor(
                        [0, haloed_segment_tokens, 2 * haloed_segment_tokens],
                        dtype=torch.int32,
                        device=device,
                    )
                    local_conv_prefix_lengths = torch.zeros(
                        2, dtype=torch.int32, device=device
                    )
                    local_conv1d_meta = prepare_causal_conv1d_metadata(
                        query_start_loc=local_conv_cu_seqlens,
                        device=device,
                    )

        return Qwen3NextMetadata(
            cp_plan=cp_plan,
            cp_segment_valid_lengths=segment_valid_lengths,
            cp_local_conv1d_meta=local_conv1d_meta,
            cp_local_conv_cu_seqlens=local_conv_cu_seqlens,
            cp_local_conv_prefix_lengths=local_conv_prefix_lengths,
            cp_local_valid_mask=local_valid_mask,
        )

    def word_embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        input_ids: torch.Tensor = inputs.input_ids
        return self.embed_tokens(input_ids)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        hidden_states = self.word_embedding(inputs)

        attention_inputs = get_primary_attention_inputs(inputs, self.kv_cache)
        is_target_verify = attention_inputs.is_target_verify
        attn_meta = Qwen3NextMetadata(is_target_verify=is_target_verify)
        if attention_inputs.is_prefill and not is_target_verify:
            if attention_inputs.context_parallel_info is not None:
                attn_meta = self._build_cp_linear_attn_metadata(
                    attention_inputs, hidden_states.device
                )
            else:
                attn_meta.prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                    query_start_loc=attention_inputs.cu_seqlens_device,
                    device=hidden_states.device,
                )

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        residual = torch.zeros_like(hidden_states)

        for i, decoder_layer in enumerate(self.layers):
            layer_attention_inputs = select_attention_inputs_for_layer(
                inputs, self.kv_cache, i
            )
            layer_fmha_impl = (
                None
                if decoder_layer.layer_type == HybridAttentionType.LINEAR
                else select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            )
            hidden_states, residual = decoder_layer(
                hidden_states,
                residual,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=layer_attention_inputs,
                attn_meta=attn_meta,
            )

        hidden_states, residual = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states)


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
