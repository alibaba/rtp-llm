"""Minimal V4.1 DSpARK prefill attention for committing target features."""

import torch

from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
from rtp_llm.models_py.modules.dsv41.fp8._kv_cache_utils import pool_entry_count
from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import ENTRY_BYTES
from rtp_llm.models_py.modules.dsv41.fp8.attention import CommitOnlyAttentionFP8
from rtp_llm.models_py.modules.dsv41.kv_cache_layout import use_v4_kv_cache


class CommitOnlyAttentionV41FP8(CommitOnlyAttentionFP8):
    """Keep the commit-only weight set while using V4.1's SWA cache ABI."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Match the allocator and normal V4.1 attention: retain the inherited
        # V4 planar ABI on Hopper; other architectures use compact V4.1 records.
        if not use_v4_kv_cache(self.kv_norm.device):
            self._pool_spec[SWA_KV] = (torch.uint8, ENTRY_BYTES)

    def _swa_entries_per_block(self) -> int:
        tag = getattr(self, "_swa_cache_region", SWA_KV)
        if self._kv_cache is not None:
            entries = pool_entry_count(self._kv_cache, tag)
            if entries is not None:
                return entries
        # CP slices include physical page padding, not additional ring slots.
        # Without logical metadata, reconstructing the ring from bytes is unsafe.
        if self._swa_cp_byte_sliced() and self._pool_raw_u8(SWA_KV) is not None:
            raise RuntimeError(
                f"DSpARK CP SWA cache requires logical entries_per_block for {tag}"
            )
        return self._pool_entries_per_block(SWA_KV)
