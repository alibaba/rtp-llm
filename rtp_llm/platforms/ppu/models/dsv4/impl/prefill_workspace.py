"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.modules.dsv4.prefill_workspace import (
    _PREFILL_WS_ALIGN_BYTES,
    _dtype_size,
    PrefillWorkspace,
)
import torch

def tp_local_prefill_q_dim(n_heads: int, head_dim: int, tp_size: int) -> int:
    """Return the per-rank dense-Q width for a head-sharded attention layer."""
    n_heads = int(n_heads)
    head_dim = int(head_dim)
    tp_size = int(tp_size)
    assert tp_size > 0 and n_heads % tp_size == 0, (
        f"n_heads={n_heads} not divisible by tp_size={tp_size}"
    )
    return (n_heads // tp_size) * head_dim
