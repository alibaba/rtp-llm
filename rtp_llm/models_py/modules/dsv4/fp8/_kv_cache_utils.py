"""FP8-only DSV4 pool context helpers."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def require_kernel_tokens_per_block(kv_cache: Any) -> int:
    """Return ``kernel_seq_size_per_block`` promoted into KVCache by C++.

    No fallback: if the C++ side failed to populate this we want a loud
    crash with diagnostics instead of silently writing the state ring
    buffer with the wrong stride. Shared by Attention / model decode
    helpers so the contract is enforced once.
    """
    ksb = int(getattr(kv_cache, "kernel_seq_size_per_block", 0))
    if ksb <= 0:
        spb = int(getattr(kv_cache, "seq_size_per_block", 0))
        grp = getattr(kv_cache, "group_region_names", None)
        raise RuntimeError(
            "DSV4 KVCache.kernel_seq_size_per_block is %d (expected >0). "
            "seq_size_per_block=%d, group_region_names=%r. The C++ CacheConfig "
            "must propagate kernel_seq_size_per_block (256 for DSV4) into "
            "PyWrappedModel's KVCache before forward." % (ksb, spb, grp)
        )
    return ksb


class PoolBackedModule(nn.Module):
    """Base class for modules backed by framework-managed paged pools.

    The FP8 indexer reads the raw pool views via ``_kv_pool_view`` /
    ``_state_pool_view``. The FP8 compressor reads the same context via
    ``_kv_pool_3d`` / ``_state_pool_3d`` because its kernels expect block-major
    tensors. Keeping both aliases here lets the two modules share the lifecycle
    code without reintroducing the old generic bind/scatter helpers.
    """

    def __init__(self) -> None:
        super().__init__()
        self._kv_pool_view: Optional[torch.Tensor] = None
        self._kv_pool_3d: Optional[torch.Tensor] = None
        self._kv_block_table: Optional[torch.Tensor] = None
        self._kv_eb: int = 0

        self._state_pool_view: Optional[torch.Tensor] = None
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
    ) -> None:
        """Install framework pool views.

        ``kv_pool_view`` is normally block-major
        ``[num_blocks, kv_eb, entry_bytes]`` for FP8 pools. ``state_pool_view``
        is normally flat ``[num_blocks * state_eb, hidden]`` and is reshaped to
        ``_state_pool_3d`` for compressor kernels.

        ``state_tokens_per_block`` is the number of tokens per kernel block
        for state-pool block_table indexing (DSV4 = 256, sourced from
        CacheConfig.kernel_seq_size_per_block). It is decoupled from
        ``state_eb`` (= ring entries per block), so the block_table is
        indexed with ``pos // state_tokens_per_block`` while the in-block
        offset uses ``pos % state_eb``.
        """
        self._kv_pool_view = kv_pool_view
        self._kv_pool_3d = kv_pool_view
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb

        self._state_pool_view = state_pool_view
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
        self._kv_pool_3d = None
        self._kv_block_table = None
        self._kv_eb = 0

        self._state_pool_view = None
        self._state_pool_3d = None
        self._state_block_table = None
        self._state_eb = 0
        self._state_tokens_per_block = 0
