"""Shared NVIDIA attention path for batch-invariant prefill and decode."""

import torch

from rtp_llm.models_py.kernels.cuda import batch_invariant
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, RopeStyle, get_rope_cache
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


class BatchInvariantAttention(FMHAImplBase):
    cuda_graph_device_metadata_only = True

    def __init__(
        self,
        config: AttentionConfigs,
        inputs: PyAttentionInputs,
        cos_sin: torch.Tensor | None,
    ) -> None:
        self.config = config
        self.inputs = inputs
        self.scale = (
            config.softmax_extra_scale / config.q_scaling * config.size_per_head**-0.5
        )
        self.cache_store = common.create_write_cache_store_impl(inputs)
        self.cos_sin = cos_sin

    @staticmethod
    def create_rope_cache(
        config: AttentionConfigs, device: torch.device
    ) -> torch.Tensor:
        with torch.cuda.device(device):
            return get_rope_cache(
                config.rope_config, config.max_seq_len, interleave=False
            )

    @staticmethod
    def supports_config(config: AttentionConfigs) -> bool:
        return (
            not config.use_mla
            and not config.is_sparse
            and not config.use_logn_attn
            and config.sliding_window == 0
            and config.is_causal
            and config.kernel_tokens_per_block > 0
            and config.kv_cache_dtype in (KvCacheDataType.BASE, KvCacheDataType.FP8)
            and config.dtype in (torch.float16, torch.bfloat16)
            and config.rope_config.style in (RopeStyle.No, RopeStyle.Base)
            and config.rope_config.offset == 0
            and config.rope_config.index_factor == 1
            and 0 < config.size_per_head <= 256
            and (
                config.rope_config.style == RopeStyle.No
                or (
                    0 < config.rope_config.dim <= config.size_per_head
                    and config.rope_config.dim % 2 == 0
                    and config.rope_config.is_neox_style
                    and config.rope_config.scale >= 1
                )
            )
            and config.head_num > 0
            and 0 < config.kv_head_num
            and config.head_num % config.kv_head_num == 0
        )

    @staticmethod
    def support(config: AttentionConfigs, inputs: PyAttentionInputs) -> bool:
        return (
            BatchInvariantAttention.supports_config(config)
            and not inputs.headwise_config
            and inputs.context_parallel_info is None
        )

    def supports_generation_prefill_cuda_graph(self) -> bool:
        return True

    def prepare_cuda_graph(self, _inputs: PyAttentionInputs) -> None:
        # The runner updates the graph-owned length and page-table tensors in place.
        pass

    def forward(
        self, qkv: torch.Tensor, kv_cache: LayerKVCache | None, layer_idx: int = 0
    ) -> torch.Tensor:
        config, inputs = self.config, self.inputs
        qkv = qkv.contiguous()
        rows = qkv.shape[0]
        query, key, value = qkv.view(
            rows, config.head_num + 2 * config.kv_head_num, config.size_per_head
        ).split((config.head_num, config.kv_head_num, config.kv_head_num), dim=1)
        pages = None
        cache = None
        if kv_cache is not None:
            pages = inputs.kv_cache_kernel_block_id_device
            if pages is None or pages.numel() == 0:
                raise ValueError(
                    "batch-invariant paged attention requires a block table"
                )
            cache = common.reshape_paged_kv_cache(
                kv_cache.kv_cache_base,
                config.kv_head_num,
                config.kernel_tokens_per_block,
                config.size_per_head,
            )
            expected_dtype = (
                torch.float8_e4m3fn
                if config.kv_cache_dtype == KvCacheDataType.FP8
                else qkv.dtype
            )
            if cache.dtype != expected_dtype or cache.shape[1:] != (
                2,
                config.kv_head_num,
                config.kernel_tokens_per_block,
                config.size_per_head,
            ):
                raise ValueError("unsupported batch-invariant KV cache layout")
            if pages.stride(1) != 1:
                pages = pages.contiguous()
        prefixes_or_lengths = (
            inputs.prefix_lengths_device
            if inputs.is_prefill
            else inputs.sequence_lengths_plus_1_device
        )
        layout = batch_invariant.attention_token_layout(
            inputs.cu_seqlens_device if inputs.is_prefill else None,
            prefixes_or_lengths,
            pages,
            rows,
            config.kernel_tokens_per_block,
        )
        if self.cos_sin is not None:
            batch_invariant.apply_rope(
                query, key, layout, self.cos_sin, inputs.combo_position_ids
            )
        if cache is not None:
            key_to_store, value_to_store = key, value
            storage = cache
            scales = None
            if cache.dtype == torch.float8_e4m3fn:
                # FP8 cache values use a direct cast, with a unit scale.
                key_to_store = key.clamp(-448, 448).to(cache.dtype).view(torch.uint8)
                value_to_store = (
                    value.clamp(-448, 448).to(cache.dtype).view(torch.uint8)
                )
                storage = cache.view(torch.uint8)
                scales = kv_cache.kv_scale_base
                if scales is not None and (
                    scales.dtype != torch.float32
                    or not scales.is_contiguous()
                    or scales.numel()
                    != cache.size(0)
                    * 2
                    * config.kv_head_num
                    * config.kernel_tokens_per_block
                ):
                    raise ValueError("unsupported batch-invariant FP8 KV scale layout")
            batch_invariant.append_kv(
                key_to_store, value_to_store, storage, layout, scales
            )
        common.apply_write_cache_store(self.cache_store, inputs, kv_cache)
        return batch_invariant.attention(
            query,
            key,
            value,
            layout,
            cache,
            pages,
            causal=True,
            scale=self.scale,
        )
