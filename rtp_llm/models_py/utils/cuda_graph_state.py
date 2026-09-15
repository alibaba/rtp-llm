"""Process-wide CUDA graph lifetime state.

Some Python-owned device allocations are read directly by kernels launched from
a captured graph.  Once any graph-enabled model has been loaded in a process,
we conservatively keep those allocations resident for the process lifetime.
The state is intentionally sticky: a second non-graph model must not turn the
protection off while the first model's graph can still be replayed.

TODO(sleep): replace the conservative keep-resident policy with an explicit
VMM fixed-VA allocation/graph invalidation protocol once each allocator-backed
cache has a rank-symmetric recapture path.
"""

import os
import threading

_GRAPH_BAKED = False
_LOCK = threading.Lock()

# One operator-facing switch controls all optional Python-owned runtime caches
# released by sleep.  Keep the old Mega-only name as a compatibility alias for
# launch scripts created before the unified switch was introduced.
RUNTIME_CACHE_RELEASE_ENV = "RTP_LLM_SLEEP_FREE_RUNTIME_CACHES"
_LEGACY_RUNTIME_CACHE_RELEASE_ENV = "RTP_LLM_SLEEP_FREE_MEGA_SYMM"


def mark_cuda_graph_baked(enabled: bool) -> None:
    """Latch graph protection when a graph-capable model is configured."""
    if not enabled:
        return
    global _GRAPH_BAKED
    with _LOCK:
        _GRAPH_BAKED = True


def cuda_graph_baked() -> bool:
    """Return whether graph-safe sleep reclaim is required in this process."""
    with _LOCK:
        return _GRAPH_BAKED


def runtime_cache_release_enabled() -> bool:
    """Whether sleep may release optional runtime caches.

    The canonical switch is ``RTP_LLM_SLEEP_FREE_RUNTIME_CACHES=1``.  The
    legacy Mega switch is accepted as a global alias so existing launchers keep
    their behavior during migration; it is not needed in new deployments.
    """
    return (
        os.environ.get(RUNTIME_CACHE_RELEASE_ENV, "0") == "1"
        or os.environ.get(_LEGACY_RUNTIME_CACHE_RELEASE_ENV, "0") == "1"
    )
