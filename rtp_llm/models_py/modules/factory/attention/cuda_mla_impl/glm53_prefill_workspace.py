"""Workspace manager for GLM5.3 context-parallel SparseMLA prefill.

The input projection produces ``q_transformed[T, H, K+R]``. Once SparseMLA
has consumed it, the same storage can hold the CP scatter result
``attention[T, H, K]`` because ``K <= K+R``. Keeping these mutually exclusive
tensors in one slot avoids a second multi-GiB allocation at the attention peak.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch


class Glm53PrefillWorkspace:
    """Union q-transform and CP-scatter storage within each attention layer.

    The manager belongs to one prefill forward, while its large storage is
    returned before that layer's MoE starts.
    """

    def __init__(self) -> None:
        self._storage: Optional[torch.Tensor] = None
        self._rows = 0
        self._heads = 0
        self._width = 0
        self._allocation_logged = False

    def q_transformed(
        self,
        rows: int,
        heads: int,
        width: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the owner view, allocating it on first use."""
        shape = (int(rows), int(heads), int(width))
        if self._storage is None:
            self._storage = torch.empty(shape, dtype=dtype, device=device)
            self._rows, self._heads, self._width = shape
            if not self._allocation_logged:
                # A new workspace is created for every prefill forward.
                logging.debug(
                    "GLM53_PREFILL_WORKSPACE_ALLOCATED: shape=%s dtype=%s bytes=%d",
                    shape,
                    dtype,
                    self._storage.numel() * self._storage.element_size(),
                )
                self._allocation_logged = True
        elif (
            self._storage.dtype != dtype
            or self._storage.device != device
            or shape != (self._rows, self._heads, self._width)
        ):
            raise RuntimeError(
                "GLM5.3 prefill workspace shape/device/dtype changed within one "
                f"forward: allocated={tuple(self._storage.shape)}/"
                f"{self._storage.device}/{self._storage.dtype}, requested="
                f"{shape}/{device}/{dtype}"
            )
        return self._storage

    def scatter_output(self, rows: int, heads: int, width: int) -> torch.Tensor:
        """Return a compact prefix view that aliases the q-transform slot."""
        if self._storage is None:
            raise RuntimeError("q_transformed must be requested before scatter_output")
        rows, heads, width = int(rows), int(heads), int(width)
        required = rows * heads * width
        if (
            rows != self._rows
            or heads != self._heads
            or required > self._storage.numel()
        ):
            raise RuntimeError(
                "GLM5.3 prefill scatter output does not fit workspace: "
                f"allocated={tuple(self._storage.shape)}, requested="
                f"{(rows, heads, width)}"
            )
        return self._storage.view(-1)[:required].view(rows, heads, width)

    def release(self) -> None:
        """Return the attention slot before the following MoE starts."""
        self._storage = None
        self._rows = self._heads = self._width = 0
