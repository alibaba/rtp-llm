import logging

from librtp_compute_ops import *
from librtp_compute_ops.rtp_llm_ops import *

try:
    import rtp_kernel

    # Override pybind classes
    from .fused_rope_kvcache_op import (
        FusedRopeKVCacheDecodeOp,
        FusedRopeKVCachePrefillOpQKVOut,
        FusedRopeKVCachePrefillOpQOut,
        TRTAttn,
    )
    logging.info("rtp_kernel successfully imported.")
except ImportError:
    logging.info("rtp_kernel not imported, maybe you are using rocm.")
