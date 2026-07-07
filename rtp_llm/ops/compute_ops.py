import logging

from librtp_compute_ops import *
from librtp_compute_ops.rtp_llm_ops import *

# Import is_cuda from the low-level device module, NOT via
# rtp_llm.models_py.utils.arch — the latter re-exports it but triggers
# models_py/__init__ to eagerly import the whole model zoo (new_models ->
# GPU attention impls -> compute_ops), creating a circular import when
# compute_ops is itself mid-initialization. device_type is a leaf (os/enum/torch).
from rtp_llm.device.device_type import is_cuda

if is_cuda():
    logging.info("Use rtp_kernel FusedRopeKVCacheOp on CUDA device.")

    from .fused_rope_kvcache_op import (
        FusedRopeKVCacheDecodeOp,
        FusedRopeKVCachePrefillOpQKVOut,
        FusedRopeKVCachePrefillOpQOut,
    )
else:
    logging.info(
        "Fallback to default implementation of FusedRopeKVCacheOp on non-CUDA device."
    )

    # On non-CUDA platforms (e.g. PPU), use C++ FusedRopeKVCacheOp from bindings
    try:
        from librtp_compute_ops.rtp_llm_ops import (
            FusedRopeKVCacheDecodeOp,
            FusedRopeKVCachePrefillOpQKVOut,
            FusedRopeKVCachePrefillOpQOut,
        )

        logging.info("Loaded C++ FusedRopeKVCacheOp from librtp_compute_ops")
    except ImportError as e:
        logging.warning(f"Failed to load C++ FusedRopeKVCacheOp: {e}")
