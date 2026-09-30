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

import threading

from rtp_llm.config.sleep_mode import (
    LEGACY_RUNTIME_CACHES_ENV,
    RUNTIME_CACHES_ENV,
    resource_release_enabled,
)

_GRAPH_BAKED = False
_LOCK = threading.Lock()

# One operator-facing switch controls all optional Python-owned runtime caches
# released by sleep.  Keep the old Mega-only name as a compatibility alias for
# launch scripts created before the unified switch was introduced.
RUNTIME_CACHE_RELEASE_ENV = RUNTIME_CACHES_ENV


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

    Unset follows sleep activation. Explicit 0 is an escape hatch and wins over
    the old Mega-only alias. Graph protection is applied by resource owners.
    """
    return resource_release_enabled(
        RUNTIME_CACHE_RELEASE_ENV, legacy_alias=LEGACY_RUNTIME_CACHES_ENV
    )
