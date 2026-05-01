"""Runtime descriptor for the framework's 7 BlockPools.

The framework's HybridPoolKVCacheAllocator allocates each attn_type as a
flat ``[num_blocks, stride_bytes] uint8`` tensor; ``stride_bytes`` is set
internally and may NOT match the user-visible ``seq_size_per_block``
config (in practice DSV4 uses 256 tokens/block regardless of that knob).

This module derives the actual per-pool geometry from the pool tensor
itself, so all paged decode ops have a single source of truth.

See ``POOL_LAYOUT.md`` for the empirical layout table this matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

# Canonical attn_type ids (mirror C++ side; used as keys throughout decode).
SWA_KV = 7
CSA_KV = 1
HCA_KV = 2
INDEXER_KV = 3
INDEXER_STATE = 4
CSA_STATE = 5
HCA_STATE = 6


@dataclass(frozen=True)
class PoolDescriptor:
    """Geometry of one BlockPool, derived from its uint8 storage tensor.

    For KV pools (multi-entry per block): ``entries_per_block`` is how many
    K vectors fit in one page, ``vec_dim`` is the K head_dim, ``vec_dtype``
    is bf16. Slot mapping is ``slot = block_id * entries_per_block + offset``.

    For STATE pools: ``entries_per_block`` is how many compressor "slots"
    (rows of ``compressor.kv_state``) fit in one page. ``vec_dim`` is the
    PACKED ``2 * coff * inner_dim`` width (kv_state ‖ score_state, fp32).
    Each row of the view holds one slot's kv-half + score-half concatenated.
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
        """Flat slot capacity = num_blocks * entries_per_block."""
        return self.num_blocks * self.entries_per_block

    def view(self) -> torch.Tensor:
        """Return a flat ``[total_slots, vec_dim]`` typed view of the pool.

        Reuses the underlying storage; safe for ``index_copy_`` /
        ``index_put_`` writes and ``index_select`` reads. Trailing
        ``stride_bytes - entries_per_block * bytes_per_entry`` bytes
        per page are ignored (alignment slack).
        """
        useful = self.entries_per_block * self.bytes_per_entry
        return self.pool[:, :useful].view(self.vec_dtype).view(-1, self.vec_dim)


def build_pool_descriptor(
    pool: torch.Tensor,
    attn_type: int,
    head_dim: int,
    indexer_head_dim: int,
    coff: int,
) -> Optional[PoolDescriptor]:
    """Construct a ``PoolDescriptor`` for one pool tensor.

    Returns None if the pool is empty/missing (some attn_types are absent
    on layers whose ``compress_ratio`` doesn't use them — e.g. CSA pools
    on a SWA-only layer).

    Args:
        pool: ``[num_blocks, stride_bytes]`` uint8 framework tensor.
        attn_type: one of SWA_KV / CSA_KV / HCA_KV / INDEXER_KV /
            INDEXER_STATE / CSA_STATE / HCA_STATE.
        head_dim: attention head_dim (V4: 512).
        indexer_head_dim: indexer's head_dim (V4: 128).
        coff: compressor's overlap factor (1 for HCA, 2 for CSA/INDEXER).
    """
    if pool is None or pool.numel() == 0:
        return None
    stride = int(pool.shape[1])

    # Each attn_type's vec dtype + dim is FIXED by the model spec — entries
    # per block is the only thing that depends on pool stride.
    if attn_type == SWA_KV:
        vec_dtype, vec_dim = torch.bfloat16, head_dim
    elif attn_type in (CSA_KV, HCA_KV):
        vec_dtype, vec_dim = torch.bfloat16, head_dim
    elif attn_type == INDEXER_KV:
        vec_dtype, vec_dim = torch.bfloat16, indexer_head_dim
    elif attn_type == CSA_STATE:
        vec_dtype, vec_dim = torch.float32, 2 * coff * head_dim
    elif attn_type == HCA_STATE:
        # HCA compressor: coff=1 (overlap=False) → vec = 2 * 1 * head_dim.
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
    """Build all present pools for one V4 layer.

    Returns a dict ``{attn_type: PoolDescriptor}``; absent pools are
    omitted (NOT mapped to None, so callers can use ``in`` checks).

    Args:
        flat_pools: ``self.kv_cache.kv_cache_base_by_layer_attn_flat``,
            indexed as ``flat[layer*attn_type_count + attn_type]``.
        layer_idx: V4 layer index.
        head_dim: 512 for V4.
        indexer_head_dim: 128 for V4.
        compress_ratio: layer's compress_ratio (0/4/128).
    """
    # CSA layers carry overlap=True → coff=2; HCA layers carry overlap=False
    # → coff=1; SWA-only layers don't have a compressor and never read the
    # state-pool descriptor anyway, so coff is unused there.
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
