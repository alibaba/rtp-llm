"""ROCm beam cascade adapter using the existing AITER RoPE/cache writers."""

import logging

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
    AiterDecodeImplBase,
    _is_mrope_interleaved_supported,
    prefill_writes_vectorized_v,
)
from rtp_llm.models_py.triton_kernels.attention.beam_search_cascade import (
    RocmBeamSearchCascadeAttention,
)
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOpAsm,
    FusedRopeKVCacheDecodeOpNonAsm,
)


class AiterCascadeDecodeImpl(AiterDecodeImplBase):
    @classmethod
    def support(cls, configs, inputs):
        return (
            torch.version.hip is not None
            and not configs.use_mla
            and not configs.is_sparse
            and configs.is_causal
            and configs.q_scaling == 1.0
            and configs.softmax_extra_scale == 1.0
            and _is_mrope_interleaved_supported(configs)
            and not inputs.is_prefill
            and not inputs.is_target_verify
            and inputs.input_lengths.numel() >= RocmBeamSearchCascadeAttention.MIN_BEAMS
            and inputs.sequence_lengths is not None
            and not inputs.sequence_lengths.is_cuda
            and inputs.kv_cache_kernel_block_id is not None
            and not inputs.kv_cache_kernel_block_id.is_cuda
            and configs.dtype in (torch.float16, torch.bfloat16)
            and configs.kv_cache_dtype == KvCacheDataType.BASE
            and configs.kv_head_num > 0
            and configs.head_num % configs.kv_head_num == 0
            and configs.head_num // configs.kv_head_num in (1, 2, 4, 8)
            and configs.size_per_head in (64, 128, 256)
            and configs.kernel_tokens_per_block > 0
            and configs.kernel_tokens_per_block % 8 == 0
        )

    def __init__(
        self,
        configs,
        inputs,
        parallelism_config,
        max_requests,
        max_seq_len,
        fmha_config=None,
    ):
        self.configs, self.attn_inputs = configs, inputs
        self.need_rope_kv_cache = configs.need_rope_kv_cache
        linear_v = not prefill_writes_vectorized_v(configs, fmha_config)
        self.rope_kvcache_impl = (
            FusedRopeKVCacheDecodeOpNonAsm if linear_v else FusedRopeKVCacheDecodeOpAsm
        )(configs)
        self.rope_params = self.rope_kvcache_impl.prepare(inputs)
        self.cascade = RocmBeamSearchCascadeAttention(
            inputs.input_lengths.numel(),
            configs.head_num,
            configs.kv_head_num,
            configs.size_per_head,
            configs.kernel_tokens_per_block,
            (max_seq_len + configs.kernel_tokens_per_block - 1)
            // configs.kernel_tokens_per_block,
            configs.dtype,
            torch.device("cuda", torch.cuda.current_device()),
            linear_v=linear_v,
            max_requests=max_requests,
        )
        self.cascade.plan(inputs.kv_cache_kernel_block_id, inputs.sequence_lengths)
        self.write_cache_store_impl = common.create_write_cache_store_impl(inputs)
        logging.info(
            "ROCm beam cascade: rows=%d, query_tile=%d, native_prefix=%s, linear_v=%s",
            self.cascade.rows,
            self.cascade.BEAMS_PER_TILE,
            self.cascade.native_prefix,
            linear_v,
        )

    def prepare_cuda_graph(self, inputs):
        self.cascade.plan(inputs.kv_cache_kernel_block_id, inputs.sequence_lengths)
        if inputs.kv_cache_kernel_block_id_device is not None:
            self.rope_params.update_kv_cache_offset(
                inputs.kv_cache_kernel_block_id_device
            )

    def forward(self, qkv, kv_cache, layer_idx=0):
        q = (
            self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
            if self.need_rope_kv_cache
            else qkv
        )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        q = q.reshape(q.shape[0], self.configs.head_num, self.configs.size_per_head)
        kv = common.reshape_paged_kv_cache(
            kv_cache.kv_cache_base,
            self.configs.kv_head_num,
            self.configs.kernel_tokens_per_block,
            self.configs.size_per_head,
        )
        return self.cascade.run(q, kv).reshape(q.shape[0], -1)
