import logging

from librtp_compute_ops import *
from librtp_compute_ops.rtp_llm_ops import *

from rtp_llm.device.device_type import is_cuda, is_hip

if is_cuda():
    logging.info("Use rtp_kernel FusedRopeKVCacheOp on CUDA device.")

    from .fused_rope_kvcache_op import (
        FusedRopeKVCacheDecodeOp,
        FusedRopeKVCachePrefillOpQKVOut,
        FusedRopeKVCachePrefillOpQOut,
    )
elif is_hip():
    # ROCm exports layout-specific operators, not the CUDA/PPU QOut/QKVOut
    # interface. Do not try the obsolete unsuffixed DecodeOp or hide an
    # actually missing ROCm binding behind a generic fallback warning.
    from librtp_compute_ops.rtp_llm_ops import (
        FusedRopeKVCacheDecodeOpAsm,
        FusedRopeKVCacheDecodeOpNonAsm,
        FusedRopeKVCachePrefillOpAsm,
        FusedRopeKVCachePrefillOpNonAsm,
    )

    logging.info("Loaded ROCm ASM/non-ASM FusedRopeKVCache operators")
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
