import logging

import torch
from librtp_compute_ops import *
from librtp_compute_ops.rtp_llm_ops import *


def _is_cuda_runtime() -> bool:
    return torch.cuda.is_available() and getattr(torch.version, "hip", None) is None


if _is_cuda_runtime():
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
