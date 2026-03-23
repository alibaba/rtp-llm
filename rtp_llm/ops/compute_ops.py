import logging

from librtp_compute_ops import *
from librtp_compute_ops.rtp_llm_ops import *

from rtp_llm.models_py.utils.arch import is_cuda, is_ppu

if is_cuda():
    # Use rtp_kernel-based implementation
    from .fused_rope_kvcache_op import (
        FusedRopeKVCacheDecodeOp,
        FusedRopeKVCachePrefillOpQKVOut,
        FusedRopeKVCachePrefillOpQOut,
    )
else:
    logging.info("Skip import rtp_kernel based ops on non-CUDA platform.")

    if is_ppu():
        logging.info("Fallback to pybind implementation on PPU.")

        from librtp_compute_ops.rtp_llm_ops import (
            FusedRopeKVCacheDecodeOp as PybindFusedRopeKVCacheDecodeOp,
            FusedRopeKVCachePrefillOpQKVOut as PybindFusedRopeKVCachePrefillOpQKVOut,
            FusedRopeKVCachePrefillOpQOut as PybindFusedRopeKVCachePrefillOpQOut,
        )
        from libth_transformer_config import AttentionConfigs

        def _make_compatibility_wrapper(base_class):
            """
            Create a wrapper class that accepts but ignores max_seq_len parameter.

            This provides API compatibility with the rtp_kernel implementation,
            which requires max_seq_len, while the pybind implementation doesn't.
            """
            class CompatibilityWrapper(base_class):
                def __init__(
                    self, attn_configs: AttentionConfigs, max_seq_len: int
                ) -> None:
                    # Accept max_seq_len for API compatibility but don't use it
                    # since the pybind implementation doesn't need it
                    super().__init__(attn_configs)

            # Preserve original class name for better debugging/logging
            CompatibilityWrapper.__name__ = base_class.__name__
            CompatibilityWrapper.__qualname__ = base_class.__name__
            return CompatibilityWrapper

        # Create compatibility wrappers for all three ops
        FusedRopeKVCacheDecodeOp = _make_compatibility_wrapper(
            PybindFusedRopeKVCacheDecodeOp
        )
        FusedRopeKVCachePrefillOpQKVOut = _make_compatibility_wrapper(
            PybindFusedRopeKVCachePrefillOpQKVOut
        )
        FusedRopeKVCachePrefillOpQOut = _make_compatibility_wrapper(
            PybindFusedRopeKVCachePrefillOpQOut
        )
