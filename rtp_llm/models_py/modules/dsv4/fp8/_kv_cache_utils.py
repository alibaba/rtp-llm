"""FP8-only DSV4 pool context helpers."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    DSV4_KERNEL_ROW_TAGS,
    DSV4_PHYSICAL_ROW_TAGS,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import group_tags as _group_tags

_PHYSICAL_ROW_TAGS = frozenset(DSV4_PHYSICAL_ROW_TAGS)
_KERNEL_ROW_TAGS = frozenset(DSV4_KERNEL_ROW_TAGS)


def _positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


def _seq_size_per_block(kv_cache: Any, tag: str) -> Optional[int]:
    getter = getattr(kv_cache, "get_seq_size_per_block", None)
    if getter is None:
        return None
    try:
        return _positive_int(getter(tag))
    except RuntimeError:
        return None


def _kernel_seq_size_per_block(kv_cache: Any, tag: str) -> Optional[int]:
    getter = getattr(kv_cache, "get_kernel_seq_size_per_block", None)
    if getter is None:
        return None
    try:
        return _positive_int(getter(tag))
    except RuntimeError:
        return None


def require_pool_tokens_per_block(kv_cache: Any, tag: str) -> int:
    """Return block-table row raw-token coverage for a cache group.

    The row size follows the group's identity: FULL paged pools (csa_kv /
    hca_kv / indexer_kv) index at kernel-block granularity and therefore use
    ``KVCache.get_kernel_seq_size_per_block(tag)``; the fixed/ring pools
    (swa_kv and the *_state groups) index at physical-block granularity and use
    ``KVCache.get_seq_size_per_block(tag)``.

    The semantic tag is the only accepted group selector: ``KVCache.group_tags``
    is a canonically ordered set of identities, so a position in it never
    identifies a group.
    """
    resolved_tag = str(tag)
    if resolved_tag in _PHYSICAL_ROW_TAGS:
        value = _seq_size_per_block(kv_cache, resolved_tag)
        if value is not None:
            return value
    if resolved_tag in _KERNEL_ROW_TAGS:
        value = _kernel_seq_size_per_block(kv_cache, resolved_tag)
        if value is not None:
            return value

    raise RuntimeError(
        "DSV4 KVCache pool tokens-per-block cannot be inferred. "
        "tag=%r, group_tags=%r" % (tag, _group_tags(kv_cache))
    )


def pool_physical_tokens_per_block(kv_cache: Any, tag: Optional[str]) -> int:
    """Physical ``seq_size_per_block`` of a cache group (0 when unavailable).

    This is the CP page-ownership unit: it is decoupled from the block-table row
    coverage returned by :func:`require_pool_tokens_per_block`, which for FULL
    paged pools reports the (possibly smaller) kernel block size and for
    CP-compacted fixed pools reports the compacted row size.
    """
    if kv_cache is None or tag is None:
        return 0
    return _seq_size_per_block(kv_cache, str(tag)) or 0


class PoolBackedModule(nn.Module):
    """Base class for modules backed by framework-managed paged pools.

    The KV pool view is expected to be the production block-major tensor.
    State pool views may arrive flat or block-major; this helper normalizes
    them to ``_state_pool_3d`` for compressor kernels.
    """

    def __init__(self) -> None:
        super().__init__()
        self._kv_pool_view: Optional[torch.Tensor] = None
        self._kv_block_table: Optional[torch.Tensor] = None
        self._kv_eb: int = 0
        self._kv_tokens_per_block: int = 0
        self._kv_owner_tokens_per_block: int = 0

        self._state_pool_3d: Optional[torch.Tensor] = None
        self._state_block_table: Optional[torch.Tensor] = None
        self._state_eb: int = 0
        self._state_tokens_per_block: int = 0

    def set_pool_context(
        self,
        kv_pool_view: Optional[torch.Tensor],
        kv_block_table: Optional[torch.Tensor],
        kv_eb: int,
        state_pool_view: Optional[torch.Tensor],
        state_block_table: Optional[torch.Tensor],
        state_eb: int,
        state_tokens_per_block: int,
        kv_tokens_per_block: int,
        kv_owner_tokens_per_block: int = 0,
    ) -> None:
        """Install framework pool views.

        ``kv_pool_view`` is normally block-major
        ``[num_blocks, kv_eb, entry_bytes]`` for FP8 pools. ``state_pool_view``
        is normally flat ``[num_blocks * state_eb, hidden]`` and is reshaped to
        ``_state_pool_3d`` for compressor kernels.

        ``kv_eb`` is the KV pool's flat entries-per-block multiplier.
        ``kv_tokens_per_block`` is the raw-token coverage of one KV block-table
        row.
        ``kv_owner_tokens_per_block`` is the raw-token coverage used for CP
        page ownership. It can differ from both KV kernel rows and fixed/SWA
        rows when fixed/SWA rows are compacted by cp_size.

        ``state_tokens_per_block`` is the raw-token coverage of one state-pool
        block-table row. It is decoupled from ``state_eb`` because state pools
        are ring buffers: the state pool is indexed with
        ``pos // state_tokens_per_block`` while the in-block offset uses
        ``pos % state_eb``.
        """
        if kv_pool_view is not None:
            assert kv_eb > 0 and kv_tokens_per_block > 0, (
                f"KV pool bound but kv_eb={kv_eb} / "
                f"kv_tokens_per_block={kv_tokens_per_block} non-positive; "
                "CacheConfig propagation broken (writer would index with zero stride)"
            )
        self._kv_pool_view = kv_pool_view
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb
        self._kv_tokens_per_block = kv_tokens_per_block
        self._kv_owner_tokens_per_block = (
            kv_owner_tokens_per_block
            if kv_owner_tokens_per_block > 0
            else kv_tokens_per_block
        )

        if state_pool_view is not None:
            assert state_eb > 0 and state_tokens_per_block > 0, (
                f"state pool bound but state_eb={state_eb} / "
                f"state_tokens_per_block={state_tokens_per_block} non-positive; "
                "CacheConfig propagation broken (writer would index with zero stride)"
            )
            if state_pool_view.dim() == 2:
                total_slots, hidden = state_pool_view.shape
                assert total_slots % state_eb == 0, (
                    f"state pool total_slots={total_slots} not divisible by "
                    f"state_eb={state_eb}"
                )
                num_blocks = total_slots // state_eb
                self._state_pool_3d = state_pool_view.view(num_blocks, state_eb, hidden)
            elif state_pool_view.dim() == 3:
                assert int(state_pool_view.shape[1]) == state_eb, (
                    f"state pool block entries={state_pool_view.shape[1]} "
                    f"does not match state_eb={state_eb}"
                )
                self._state_pool_3d = state_pool_view
            else:
                raise AssertionError(
                    f"expected 2D or 3D state pool view, got {state_pool_view.shape}"
                )
        else:
            self._state_pool_3d = None
        self._state_block_table = state_block_table
        self._state_eb = state_eb
        self._state_tokens_per_block = state_tokens_per_block

    def clear_pool_context(self) -> None:
        self._kv_pool_view = None
        self._kv_block_table = None
        self._kv_eb = 0
        self._kv_tokens_per_block = 0
        self._kv_owner_tokens_per_block = 0

        self._state_pool_3d = None
        self._state_block_table = None
        self._state_eb = 0
        self._state_tokens_per_block = 0
