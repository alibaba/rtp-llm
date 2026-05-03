import logging

from librtp_compute_ops import *
from librtp_compute_ops.rtp_llm_ops import *

from rtp_llm.models_py.utils.arch import is_cuda

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

        # Define stub names so `from rtp_llm.ops.compute_ops import
        # FusedRopeKVCache*` succeeds at module import time (e.g. during
        # pytest collection on a no-driver runner). Test files that
        # actually use these classes are gated by pytest.mark.skipif on
        # CUDA availability and never instantiate the stub. Calling the
        # stub raises clearly to surface real misuse instead of failing
        # silently with AttributeError later.
        class _FusedRopeKVCacheUnavailable:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "FusedRopeKVCacheOp is unavailable: not on CUDA and "
                    "C++ binding (librtp_compute_ops.rtp_llm_ops) does "
                    "not export it. This stub exists for collection-time "
                    "import safety only."
                )

        FusedRopeKVCacheDecodeOp = _FusedRopeKVCacheUnavailable  # type: ignore[assignment,misc]
        FusedRopeKVCachePrefillOpQKVOut = _FusedRopeKVCacheUnavailable  # type: ignore[assignment,misc]
        FusedRopeKVCachePrefillOpQOut = _FusedRopeKVCacheUnavailable  # type: ignore[assignment,misc]
