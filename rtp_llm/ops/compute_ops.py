import logging

from librtp_compute_ops import *
from librtp_compute_ops.rtp_llm_ops import *

use_rtp_kernel = False

try:
    import torch

    major, minor = map(int, torch.version.cuda.split(".")[:2])
    if (major, minor) >= (12, 8):
        try:
            import rtp_kernel

            # Override pybind classes
            from .fused_rope_kvcache_op import (
                FusedRopeKVCacheDecodeOp,
                FusedRopeKVCachePrefillOp,
                TRTAttn,
            )

            use_rtp_kernel = True
            logging.info("CUDA >= 12.8, using rtp_kernel")
        except (ImportError, AttributeError) as e:
            logging.info(
                f"CUDA >= 12.8 but rtp_kernel not available ({e}), falling back to pybind."
            )
    else:
        logging.info(f"CUDA version {major}.{minor} < 12.8, using pybind.")
except Exception as e:
    logging.warning(f"Failed to check CUDA version ({e}), using pybind.")

if not use_rtp_kernel:
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
        logging.info("librtp_compute_ops.rtp_llm_ops not imported, may be you are using rocm.")
