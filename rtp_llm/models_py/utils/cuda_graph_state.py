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

_GRAPH_BAKED = False
_LOCK = threading.Lock()


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
