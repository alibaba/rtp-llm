"""FlashAttention-3 paged attention for fixed-width short-query graphs.

The generic CUDA-graph runtime owns graph bucketing and replay, while the
existing fused RoPE/KV-cache operator owns position encoding and cache writes.
FA3 only consumes the resulting query and the main paged KV cache.
"""

import copy
from typing import Optional, Tuple

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.utils.arch import get_num_device_sms, is_sm90
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOpQOut,
    LayerKVCache,
    PyAttentionInputs,
)

try:
    # CUDA 12.9 builds carry the FA3 dependency. CUDA 13 currently does not,
    # so an unavailable binary remains a clean factory miss.
    from flash_attn_interface import flash_attn_with_kvcache

    _HAS_FLASH_ATTN_3 = True
except (ImportError, OSError):
    flash_attn_with_kvcache = None
    _HAS_FLASH_ATTN_3 = False


# A one-block limit covers speculative verify and hot-prefix suffix graphs
# without turning large/chunked prefill into a fixed-shape graph workload.
_MAX_QUERY_WIDTH = 64
_MAX_SPLITS = 32
_TARGET_CTA_PER_SM = 4
_SINGLE_SPLIT_CTA_PER_SM = 3


def _short_graph_num_splits(
    batch_size: int, head_num: int, sm_count: int
) -> int:
    """Choose FA3 split-K occupancy for a fixed-width short-query graph."""
    if batch_size <= 0 or head_num <= 0 or sm_count <= 0:
        raise ValueError("batch_size, head_num and sm_count must be positive")

    active_ctas = batch_size * head_num
    if active_ctas >= _SINGLE_SPLIT_CTA_PER_SM * sm_count:
        return 1
    target_ctas = _TARGET_CTA_PER_SM * sm_count
    return max(
        1,
        min(_MAX_SPLITS, (target_ctas + active_ctas - 1) // active_ctas),
    )


def _fixed_width_active_prefix(
    input_lengths: Optional[torch.Tensor],
) -> Optional[Tuple[int, int]]:
    """Return ``(graph_batch_size, query_width)`` for padded graph geometry."""
    if input_lengths is None or input_lengths.dim() != 1 or input_lengths.numel() == 0:
        return None

    # Factory selection happens while graph objects are constructed, outside
    # capture/replay. Replay continues to consume device-resident metadata.
    lengths = [int(length) for length in input_lengths.detach().cpu().tolist()]
    query_width = lengths[0]
    if query_width <= 0:
        return None

    active_batch_size = 0
    while (
        active_batch_size < len(lengths)
        and lengths[active_batch_size] == query_width
    ):
        active_batch_size += 1

    if active_batch_size == 0 or any(length != 0 for length in lengths[active_batch_size:]):
        return None
    return active_batch_size, query_width


def _rope_capture_inputs(attn_inputs: PyAttentionInputs) -> PyAttentionInputs:
    """Retain graph-owned device length buffers in fused-RoPE parameters."""
    rope_inputs = copy.copy(attn_inputs)
    if (
        attn_inputs.input_lengths_device is not None
        and attn_inputs.input_lengths_device.numel() > 0
    ):
        rope_inputs.input_lengths = attn_inputs.input_lengths_device
    if (
        attn_inputs.prefix_lengths_device is not None
        and attn_inputs.prefix_lengths_device.numel() > 0
    ):
        rope_inputs.prefix_lengths = attn_inputs.prefix_lengths_device
    return rope_inputs


class FlashAttn3PagedShortGraphImpl(FMHAImplBase):
    """FA3 paged attention for uniform one-block CUDA graphs.

    This implementation is phase- and model-agnostic. Any causal MHA/GQA
    graph with a fixed query width and BF16/FP16 paged KV cache may select it.
    """

    cuda_graph_device_metadata_only = True

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        graph_geometry = _fixed_width_active_prefix(attn_inputs.input_lengths)
        if graph_geometry is None:
            raise ValueError("invalid fixed-width CUDA graph input geometry")

        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.batch_size, self.query_width = graph_geometry
        self.num_splits = _short_graph_num_splits(
            self.batch_size,
            attn_configs.head_num,
            get_num_device_sms(),
        )
        self.softmax_scale = (
            attn_configs.softmax_extra_scale
            / attn_configs.q_scaling
            * attn_configs.size_per_head**-0.5
        )

        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(attn_configs)
        self.rope_params = self.rope_kvcache_impl.prepare(
            _rope_capture_inputs(attn_inputs)
        )
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        self.cache_sequence_lengths = torch.empty(
            self.batch_size, dtype=torch.int32, device="cuda"
        )

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        if not _HAS_FLASH_ATTN_3 or not is_sm90():
            return False
        if (
            attn_configs.use_mla
            or attn_configs.kv_cache_dtype != KvCacheDataType.BASE
            or attn_configs.dtype not in (torch.bfloat16, torch.float16)
            or attn_configs.size_per_head not in (64, 96, 128, 192, 256)
            or attn_configs.kernel_tokens_per_block <= 0
        ):
            return False
        if not attn_inputs.is_prefill or not attn_inputs.is_cuda_graph:
            return False
        if (
            attn_inputs.input_lengths is None
            or attn_inputs.input_lengths.numel() == 0
            or attn_inputs.prefix_lengths is None
            or attn_inputs.prefix_lengths.numel() == 0
            or attn_inputs.kv_cache_kernel_block_id_device is None
            or attn_inputs.kv_cache_kernel_block_id_device.numel() == 0
        ):
            return False

        graph_geometry = _fixed_width_active_prefix(attn_inputs.input_lengths)
        if graph_geometry is None:
            return False
        graph_batch_size, query_width = graph_geometry
        return (
            1 < query_width <= _MAX_QUERY_WIDTH
            and attn_inputs.prefix_lengths.numel() >= graph_batch_size
            and attn_inputs.kv_cache_kernel_block_id_device.size(0)
            >= graph_batch_size
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        assert flash_attn_with_kvcache is not None
        assert kv_cache is not None

        query = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        head_num = self.attn_configs.head_num
        head_dim = self.attn_configs.size_per_head
        query = query.view(self.batch_size, self.query_width, head_num, head_dim)

        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache,
                self.attn_configs.kv_head_num,
                self.attn_configs.kernel_tokens_per_block,
                head_dim,
            )
        key_cache = paged_kv_cache[:, 0].transpose(1, 2)
        value_cache = paged_kv_cache[:, 1].transpose(1, 2)

        prefix_lengths = self.attn_inputs.prefix_lengths_device
        if prefix_lengths is None or prefix_lengths.numel() == 0:
            prefix_lengths = self.attn_inputs.prefix_lengths
        prefix_lengths = prefix_lengths[: self.batch_size]
        input_lengths = self.attn_inputs.input_lengths_device
        if input_lengths is None or input_lengths.numel() == 0:
            input_lengths = self.attn_inputs.input_lengths
        input_lengths = input_lengths[: self.batch_size]
        torch.add(prefix_lengths, input_lengths, out=self.cache_sequence_lengths)

        output = flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            cache_seqlens=self.cache_sequence_lengths,
            page_table=self.attn_inputs.kv_cache_kernel_block_id_device[
                : self.batch_size
            ],
            max_seqlen_q=self.query_width,
            softmax_scale=self.softmax_scale,
            causal=self.attn_configs.is_causal,
            num_splits=self.num_splits,
        )
        return output.view(-1, head_num * head_dim)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        # Query width and launch geometry are fixed at capture; replay updates
        # only the request's page table dependent fused-RoPE offset.
        new_offset = self.rope_kvcache_impl.prepare_kv_cache_offset(attn_inputs)
        if new_offset is not None:
            common.copy_kv_cache_offset(
                self.rope_params.kv_cache_offset, new_offset
            )
