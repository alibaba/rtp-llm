"""Canonical attn_type ids for DSV4's 7 BlockPools.

Phase F: the ``PoolDescriptor`` + ``build_*`` helpers below are retained
ONLY for byte-equal regression tests (``test_phase_b_prefill_dual_write``,
``test_paged_kv_write``, etc.) which compare our paged write path against
a reference scatter implementation.  Production ``Attention`` code no
longer uses them — it resolves pool views inline via
``self._kv_cache.get_layer_cache(layer_id, attn_type).kv_cache_base``.

The attn_type id constants are used as dict keys and cross-module
identifiers (``Attention`` write/read paths, metadata slot mappings,
decode metadata builders).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

# Canonical attn_type ids (mirror C++ KVCacheAttnType enum).
SWA_KV = 7
CSA_KV = 1
HCA_KV = 2
INDEXER_KV = 3
INDEXER_STATE = 4
CSA_STATE = 5
HCA_STATE = 6


# ---------------------------------------------------------------------------
# Legacy PoolDescriptor — test-only (see module docstring).
# Production code is expected to use ``Attention._pool_view(attn_type)``
# instead, which resolves through the framework ``KVCache`` handle.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolDescriptor:
    """DEPRECATED (test-only, Phase F).  Geometry of one BlockPool.

    For KV pools: ``entries_per_block`` = K vectors per page; ``vec_dim``
    = K head_dim; ``vec_dtype`` = bf16.  Slot mapping = ``block_id * eb +
    offset``.  For STATE pools: ``vec_dim`` is the PACKED ``2 * coff *
    inner_dim`` width (kv_state ‖ score_state, fp32).
    """

    pool: torch.Tensor  # [num_blocks, stride_bytes] uint8
    attn_type: int
    entries_per_block: int
    vec_dim: int
    vec_dtype: torch.dtype
    bytes_per_entry: int  # vec_dim * dtype.itemsize

    @property
    def num_blocks(self) -> int:
        return int(self.pool.shape[0])

    @property
    def stride_bytes(self) -> int:
        return int(self.pool.shape[1])

    @property
    def total_slots(self) -> int:
        return self.num_blocks * self.entries_per_block

    def view(self) -> torch.Tensor:
        """Return a flat ``[total_slots, vec_dim]`` typed view of the pool."""
        useful = self.entries_per_block * self.bytes_per_entry
        return self.pool[:, :useful].view(self.vec_dtype).view(-1, self.vec_dim)


def build_pool_descriptor(
    pool: torch.Tensor,
    attn_type: int,
    head_dim: int,
    indexer_head_dim: int,
    coff: int,
) -> Optional[PoolDescriptor]:
    """DEPRECATED (test-only, Phase F).  Construct a ``PoolDescriptor``
    for one pool tensor.  Returns None if the pool is empty."""
    if pool is None or pool.numel() == 0:
        return None
    stride = int(pool.shape[1])

    if attn_type == SWA_KV:
        vec_dtype, vec_dim = torch.bfloat16, head_dim
    elif attn_type in (CSA_KV, HCA_KV):
        vec_dtype, vec_dim = torch.bfloat16, head_dim
    elif attn_type == INDEXER_KV:
        vec_dtype, vec_dim = torch.bfloat16, indexer_head_dim
    elif attn_type == CSA_STATE:
        vec_dtype, vec_dim = torch.float32, 2 * coff * head_dim
    elif attn_type == HCA_STATE:
        vec_dtype, vec_dim = torch.float32, 2 * head_dim
    elif attn_type == INDEXER_STATE:
        vec_dtype, vec_dim = torch.float32, 2 * coff * indexer_head_dim
    else:
        return None

    bytes_per_entry = vec_dim * vec_dtype.itemsize
    if bytes_per_entry == 0 or stride < bytes_per_entry:
        return None
    entries_per_block = stride // bytes_per_entry
    return PoolDescriptor(
        pool=pool,
        attn_type=attn_type,
        entries_per_block=entries_per_block,
        vec_dim=vec_dim,
        vec_dtype=vec_dtype,
        bytes_per_entry=bytes_per_entry,
    )


def build_layer_pool_descriptors(
    flat_pools: list,
    layer_idx: int,
    head_dim: int,
    indexer_head_dim: int,
    compress_ratio: int,
    attn_type_count: int = 8,
) -> Dict[int, PoolDescriptor]:
    """DEPRECATED (test-only, Phase F).  Build all present pools for one
    V4 layer."""
    coff = 2 if compress_ratio == 4 else 1
    out: Dict[int, PoolDescriptor] = {}
    for at in (SWA_KV, CSA_KV, HCA_KV, INDEXER_KV, INDEXER_STATE, CSA_STATE, HCA_STATE):
        idx = layer_idx * attn_type_count + at
        if idx >= len(flat_pools):
            continue
        pool = flat_pools[idx]
        desc = build_pool_descriptor(pool, at, head_dim, indexer_head_dim, coff)
        if desc is not None:
            out[at] = desc
    return out
