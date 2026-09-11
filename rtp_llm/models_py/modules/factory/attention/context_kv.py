"""Commit projected context through the existing RoPE/KV-cache operators."""

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules.factory.attention import common


class ContextKVWriter:
    def __init__(self, attn_config, attn_inputs, fmha_config=None):
        if get_device_type() == DeviceType.ROCm:
            from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
                prefill_writes_vectorized_v,
            )
            from rtp_llm.ops.compute_ops import (
                FusedRopeKVCachePrefillOpAsm,
                FusedRopeKVCachePrefillOpNonAsm,
            )

            op = (
                FusedRopeKVCachePrefillOpAsm
                if prefill_writes_vectorized_v(attn_config, fmha_config)
                else FusedRopeKVCachePrefillOpNonAsm
            )
            self.rope = op(attn_config)
            self.rope.use_paged_fmha = True
        else:
            from rtp_llm.ops.compute_ops import FusedRopeKVCachePrefillOpQOut

            self.rope = FusedRopeKVCachePrefillOpQOut(attn_config)
        self.params = self.rope.prepare(attn_inputs)
        self.write_cache_store = common.create_write_cache_store_impl(attn_inputs)

    def prepare_cuda_graph(self, attn_inputs):
        if get_device_type() == DeviceType.ROCm:
            self.params.prepare_in_place(attn_inputs)
        else:
            # CUDA params retain the graph runner's input tensors; only the
            # derived page offsets need updating before replay.
            common.copy_kv_cache_offset(
                self.params.kv_cache_offset,
                self.rope.prepare(attn_inputs).kv_cache_offset,
            )

    def forward(self, qkv, kv_cache):
        self.rope.forward(qkv, kv_cache, self.params)
        if self.write_cache_store is not None:
            self.write_cache_store(kv_cache)
