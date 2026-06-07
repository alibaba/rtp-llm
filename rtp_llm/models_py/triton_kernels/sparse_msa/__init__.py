"""MiniMax MSA (sparse attention) Triton kernels.

Direct port of sglang's ``srt/layers/attention/minimax_sparse_ops``. The
three-step prefill / decode flow (index attention → topk reduce → sparse GQA)
is intentionally kept identical to the reference; only the consumer
(MSAAttention module) will need to adapt to rtp-llm's KV cache layout.

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
    """Return (minimax_sparse_prefill, minimax_sparse_decode), compiling lazily."""
    from .minimax_sparse import minimax_sparse_decode, minimax_sparse_prefill

    return minimax_sparse_prefill, minimax_sparse_decode


__all__ = ["get_sparse_ops"]
