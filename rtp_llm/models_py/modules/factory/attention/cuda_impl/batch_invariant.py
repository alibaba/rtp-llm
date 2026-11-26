from typing import Optional

import torch

from rtp_llm.config.cuda_graph import (
    CudaGraphSelectionMode,
    GenerationPrefillCudaGraphUnsupportedBackend,
)
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules.factory.attention.common import (
    copy_kv_cache_offset,
    reshape_paged_kv_cache,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQOut,
)


class BatchInvariantAttention(FMHAImplBase):
    """Reuse RTP's RoPE and KV writers with one attention reduction for both phases."""

    def __init__(self, config: AttentionConfigs, inputs: PyAttentionInputs):
        from rtp_llm.models_py.determinism.batch_invariant import (
            batch_invariant_paged_attention,
        )

        self.paged_attention = batch_invariant_paged_attention
        self.config = config
        self.is_prefill = inputs.is_prefill
        self.block_table = inputs.kv_cache_kernel_block_id_device
        if inputs.is_prefill:
            device = self.block_table.device
            self.rope = FusedRopeKVCachePrefillOpQOut(config)
            self.request_ids = torch.repeat_interleave(
                torch.arange(inputs.input_lengths.numel(), device=device),
                inputs.input_lengths_device,
                output_size=inputs.total_tokens,
            )
            starts = inputs.cu_seqlens_device[self.request_ids]
            prefixes = inputs.prefix_lengths_device[self.request_ids]
            self.kv_lengths = (
                torch.arange(self.request_ids.numel(), device=device, dtype=torch.int32)
                - starts
                + prefixes
                + 1
            )
        else:
            self.rope = FusedRopeKVCacheDecodeOp(config)
            self.request_ids = inputs.decode_cu_seqlens_device[:-1]
            self.kv_lengths = inputs.sequence_lengths_plus_1_device
        self.rope_params = self.rope.prepare(inputs)

    @staticmethod
    def support(config: AttentionConfigs, inputs: PyAttentionInputs) -> bool:
        return (
            config.is_causal
            and config.need_rope_kv_cache
            and not config.use_mla
            and config.kv_cache_dtype == KvCacheDataType.BASE
            and not inputs.is_target_verify
        )

    def support_cuda_graph(self) -> bool:
        return not self.is_prefill

    def prepare_cuda_graph(self, inputs: PyAttentionInputs) -> None:
        # Lengths, positions and the page table are captured in place. RoPE
        # derives an offset table from the current pages and owns its length buffer.
        params = self.rope.prepare(inputs, forbid_reallocation=True)
        copy_kv_cache_offset(self.rope_params.kv_cache_offset, params.kv_cache_offset)

    def forward(
        self, qkv: torch.Tensor, kv_cache: Optional[LayerKVCache], layer_idx: int = 0
    ) -> torch.Tensor:
        if kv_cache is None:
            raise ValueError("batch-invariant attention requires paged KV cache")
        config = self.config
        q = self.rope.forward(qkv, kv_cache, self.rope_params)
        q = q.reshape(-1, config.head_num, config.size_per_head)
        cache = reshape_paged_kv_cache(
            kv_cache.kv_cache_base,
            config.kv_head_num,
            config.kernel_tokens_per_block,
            config.size_per_head,
        )
        return self.paged_attention(
            q,
            cache,
            self.block_table,
            self.request_ids,
            self.kv_lengths,
            config.size_per_head**-0.5,
        )


def create_batch_invariant_impl(
    attn_configs: AttentionConfigs,
    attn_inputs: PyAttentionInputs,
    is_cuda_graph: bool,
    cuda_graph_selection_mode: Optional[str],
) -> BatchInvariantAttention:
    if get_device_type() != DeviceType.Cuda:
        raise ValueError("batch-invariant attention requires a CUDA backend")
    if not BatchInvariantAttention.support(attn_configs, attn_inputs):
        raise ValueError("batch-invariant attention requires causal BF16 GQA")
    if (
        is_cuda_graph
        and cuda_graph_selection_mode == CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH
    ):
        raise GenerationPrefillCudaGraphUnsupportedBackend(
            "batch-invariant attention supports pure decode CUDA Graphs"
        )
    expected_mode = (
        CudaGraphSelectionMode.DECODE_GRAPH
        if is_cuda_graph
        else CudaGraphSelectionMode.EAGER
    )
    if cuda_graph_selection_mode not in (None, expected_mode):
        raise ValueError("inconsistent batch-invariant CUDA graph selection mode")
    if is_cuda_graph and attn_inputs.is_prefill:
        raise ValueError("batch-invariant CUDA graph requires pure decode")
    attn_inputs.is_cuda_graph = is_cuda_graph
    return BatchInvariantAttention(attn_configs, attn_inputs)
