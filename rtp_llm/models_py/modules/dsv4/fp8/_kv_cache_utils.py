"""FP8-only DSV4 pool context helpers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


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

    def set_pool_context(
        self,
        kv_pool_view: Optional[torch.Tensor],
        kv_block_table: Optional[torch.Tensor],
        kv_eb: int,
        state_pool_view: Optional[torch.Tensor],
        state_block_table: Optional[torch.Tensor],
        state_eb: int,
    ) -> None:
        """Install framework pool views.

        ``kv_pool_view`` is normally block-major
        ``[num_blocks, kv_eb, entry_bytes]`` for FP8 pools. ``state_pool_view``
        is normally flat ``[num_blocks * state_eb, hidden]`` and is reshaped to
        ``_state_pool_3d`` for compressor kernels.
        """
        self._kv_pool_view = kv_pool_view
        self._kv_pool_3d = kv_pool_view
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb

        self._state_pool_view = state_pool_view
        if state_pool_view is not None and state_eb > 0:
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

    def clear_pool_context(self) -> None:
        self._kv_pool_view = None
        self._kv_pool_3d = None
        self._kv_block_table = None
        self._kv_eb = 0

        self._state_pool_view = None
        self._state_pool_3d = None
        self._state_block_table = None
        self._state_eb = 0
