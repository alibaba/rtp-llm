"""Small shape helpers shared by DSV4 HC implementations."""

from __future__ import annotations

import torch


def maybe_squeeze_hc_1d(t: torch.Tensor) -> torch.Tensor:
    """Undo the loader's generic 1D scale -> [N, 1] promotion for HC scales."""
    if t.dim() == 2 and t.shape[-1] == 1:
        return t.squeeze(-1)
    return t


def wrap_hc_batch(
    t: torch.Tensor,
    batched_dims: int,
    *,
    name: str,
) -> tuple[torch.Tensor, bool]:
    """Convert flat HC tensors to the batched shape required by TileLang.

    TileLang kernels consume batched tensors. The public HC interface accepts
    either flat prefill ``[T, ...]`` or batched decode ``[B, S, ...]``. Before
    adding a synthetic batch dimension, assert the tensor is exactly the flat
    rank expected by the public contract; otherwise assert it is already the
    batched rank.
    """
    assert batched_dims in (3, 4), f"{name}: unsupported batched_dims={batched_dims}"
    if t.dim() == batched_dims - 1:
        # Flat prefill uses the public HC layout [T, ...], with request
        # boundaries carried separately by cu_seqlens outside HC. TileLang
        # kernels consume only the batched layout [B, S, ...], so present the
        # same token stream as B=1, S=T. unsqueeze(0) is a metadata-only view:
        # it does not copy and keeps a contiguous input contiguous.
        return t.unsqueeze(0), True
    assert (
        t.dim() == batched_dims
    ), f"{name}: expected rank {batched_dims - 1} or {batched_dims}, got shape={tuple(t.shape)}"
    return t, False


def squeeze_hc_batch(t: torch.Tensor, wrapped: bool, *, name: str) -> torch.Tensor:
    if wrapped:
        # Undo only the synthetic batch dimension from wrap_hc_batch. Enforce
        # size 1 so a real batch axis can never be removed accidentally.
        assert t.dim() >= 1 and int(t.shape[0]) == 1, (
            f"{name}: expected synthetic leading batch of size 1 before squeeze, "
            f"got shape={tuple(t.shape)}"
        )
        return t.squeeze(0)
    return t
