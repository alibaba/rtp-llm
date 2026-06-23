"""FP8-only DSV4 pool context helpers."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.attn_type import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
    TAG_BY_ATTN_TYPE,
)

_PHYSICAL_ROW_REGIONS = {
    int(SWA_KV),
    int(CSA_STATE),
    int(HCA_STATE),
    int(INDEXER_STATE),
}
_KERNEL_ROW_REGIONS = {
    int(CSA_KV),
    int(HCA_KV),
    int(INDEXER_KV),
}


def _positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


_ATTN_TYPE_BY_TAG = {tag: attn_type for attn_type, tag in TAG_BY_ATTN_TYPE.items()}


def _attn_type_for_group_or_region(
    kv_cache: Any,
    group: Optional[int] = None,
    region: Optional[int] = None,
) -> Optional[int]:
    if region is not None:
        return int(region)
    if group is None:
        return None
    group_tags = getattr(kv_cache, "group_tags", None)
    if not group_tags or group < 0 or group >= len(group_tags):
        return None
    return _ATTN_TYPE_BY_TAG.get(str(group_tags[group]))


def _group_for_region(kv_cache: Any, region: int) -> Optional[int]:
    group_tags = getattr(kv_cache, "group_tags", None)
    if not group_tags:
        return None
    region = int(region)
    expected_tag = TAG_BY_ATTN_TYPE.get(region)
    for gid, group_tag in enumerate(group_tags):
        if str(group_tag) == expected_tag:
            return gid
    return None


def _group_tokens_per_block(kv_cache: Any, group: Optional[int]) -> Optional[int]:
    if group is None:
        return None
    group_sizes = getattr(kv_cache, "group_seq_size_per_block", None)
    if not group_sizes or group < 0 or group >= len(group_sizes):
        return None
    return _positive_int(group_sizes[group])


def require_pool_tokens_per_block(
    kv_cache: Any,
    group: Optional[int] = None,
    region: Optional[int] = None,
) -> int:
    """Return block-table row raw-token coverage for a pool.

    C++ exposes only scalar physical/kernel block sizes. The group-specific
    row size is inferred from the region identity: FULL paged pools use
    ``kernel_seq_size_per_block``; SWA_KV and state pools use
    ``seq_size_per_block``.
    """
    attn_type = _attn_type_for_group_or_region(kv_cache, group=group, region=region)
    if attn_type in _PHYSICAL_ROW_REGIONS:
        value = _group_tokens_per_block(
            kv_cache,
            group if group is not None else _group_for_region(kv_cache, int(attn_type)),
        )
        if value is not None:
            return value
        value = _positive_int(getattr(kv_cache, "seq_size_per_block", None))
        if value is not None:
            return value
    if attn_type in _KERNEL_ROW_REGIONS:
        value = _positive_int(getattr(kv_cache, "kernel_seq_size_per_block", None))
        if value is not None:
            return value

    raise RuntimeError(
        "DSV4 KVCache pool tokens-per-block cannot be inferred. "
        "group=%r, region=%r, group_tags=%r"
        % (group, region, getattr(kv_cache, "group_tags", None))
    )


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
            kv_owner_tokens_per_block if kv_owner_tokens_per_block > 0 else kv_tokens_per_block
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
