import logging

from librtp_compute_ops import *
from librtp_compute_ops.rtp_llm_ops import *

try:
    import rtp_kernel

    # Override pybind classes
    from .fused_rope_kvcache_op import (
        FusedRopeKVCacheDecodeOp,
        FusedRopeKVCachePrefillOp,
        TRTAttn,
    )
    logging.info("rtp_kernel successfully imported.")

except ImportError:
    logging.info("rtp_kernel not imported, falling back to pybind.")

    try:
        logging.info("Attempting to override pybind classes.")
        from librtp_compute_ops.rtp_llm_ops import (
            FusedRopeKVCacheDecodeOp as PybindFusedRopeKVCacheDecodeOp,
        )
        from librtp_compute_ops.rtp_llm_ops import (
            FusedRopeKVCachePrefillOp as PybindFusedRopeKVCachePrefillOp,
        )

        class FusedRopeKVCachePrefillOp(PybindFusedRopeKVCachePrefillOp):
            def __init__(self, attn_configs, max_seq_len):
                super().__init__(attn_configs)

        class FusedRopeKVCacheDecodeOp(PybindFusedRopeKVCacheDecodeOp):
            def __init__(self, attn_configs, max_seq_len):
                super().__init__(attn_configs)

    except ImportError:
        logging.info(
            "librtp_compute_ops.rtp_llm_ops not imported, maybe you are using rocm."
        )
