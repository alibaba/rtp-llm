"""MiniMax MSA (sparse attention) Triton kernels.

The runtime prefill and decode operators read paged K/V. Prefill uses compact
HND working pages; decode reads the persistent cache-manager pages.

**Triton 3.6.0 / Python 3.10 note**: the kernels carry 3-deep
``@triton.heuristics → @triton.autotune → @triton.jit`` decorator stacks. When
the ``@triton.heuristics({...})`` dict spans multiple physical lines and holds
``lambda`` entries, ``inspect.getsourcelines(fn)`` truncates to that first
decorator block, and ``triton.jit`` then fails its ``^def funcname(`` regex
with ``AttributeError: 'NoneType' object has no attribute 'start'``. The fix
applied to every kernel here hoists those dicts to module-level
``_HEUR_<kernel>`` variables so the decorator line stays single-physical-line;
the kernels now import (and compile lazily on first call) safely.

Use :func:`get_sparse_ops` (or import :mod:`.minimax_sparse` directly) when the
MSA path is wired up.
"""

from typing import Callable, Tuple


def get_sparse_ops() -> Tuple[Callable, Callable]:
    """Return the paged prefill and decode operators, compiling lazily."""
    from .minimax_sparse import minimax_paged_sparse_decode, minimax_sparse_prefill

    return minimax_sparse_prefill, minimax_paged_sparse_decode


__all__ = ["get_sparse_ops"]
