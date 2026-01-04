import logging
from typing import Any

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
)
from rtp_llm.ops import AttentionConfigs, FMHAType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    PyAttentionInputs,
    XQAAttnOp,
)


class XQAImpl(FMHADecodeImplBase):

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            XQAAttnOp(attn_configs),
            FusedRopeKVCacheDecodeOp(attn_configs),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.XQA

    def support_cuda_graph(self) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        """Unified prepare method supporting initial preparation and replay.

        Automatically detects whether this is first-time preparation or replay
        based on whether fmha_params exists.
        """
        assert self.fmha_impl is not None
        assert self.rope_kvcache_impl is not None

        # Detect if this is first call or replay
        is_first_call = self.fmha_params is None

        if is_first_call:
            # First-time: create new params
            self.fmha_params = self.fmha_impl.prepare(attn_inputs)
            self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        else:
            # Replay: update existing params by copying offsets
            new_fmha_params = self.fmha_impl.prepare(attn_inputs)
            new_offset = new_fmha_params.kv_cache_offset
            old_offset = self.fmha_params.kv_cache_offset
            self.copy_kv_cache_offset(old_offset, new_offset)

            new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
            new_offset = new_rope_params.kv_cache_offset
            old_offset = self.rope_params.kv_cache_offset
            self.copy_kv_cache_offset(old_offset, new_offset)
