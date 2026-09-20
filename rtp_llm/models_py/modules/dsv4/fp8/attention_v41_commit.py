"""Minimal V4.1 DSpARK prefill attention for committing target features."""

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV
from rtp_llm.models_py.modules.dsv4.fp8._v41_swa_triton import ENTRY_BYTES
from rtp_llm.models_py.modules.dsv4.fp8.attention import CommitOnlyAttentionFP8


class CommitOnlyAttentionV41FP8(CommitOnlyAttentionFP8):
    """Keep the commit-only weight set while using V4.1's SWA cache ABI."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pool_spec[SWA_KV] = (torch.uint8, ENTRY_BYTES)

    def _swa_entries_per_block(self) -> int:
        if self._swa_cp_byte_sliced():
            raw = self._pool_raw_u8(SWA_KV)
            if raw is not None:
                return int(raw.shape[1]) * int(self._cp_ctx.cp_size) // ENTRY_BYTES
        return self._pool_entries_per_block(SWA_KV)
