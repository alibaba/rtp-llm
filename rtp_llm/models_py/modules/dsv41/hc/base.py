"""Common interfaces for DeepSeek-V4 mHC modules.

All implementations must use the same public shape contract:

  * Flat prefill layout:
    ``residual`` is ``[T, hc_mult, dim]`` where ``T = sum(sequence_lengths)``.
    Batch/sequence boundaries are represented outside HC by ``cu_seqlens``.

  * Batched decode/standalone layout:
    ``residual`` is ``[B, S, hc_mult, dim]``. Decode usually has ``S == 1``.

For both layouts, HC pre/post/head preserve the leading token layout exactly:

  * ``HCUnitBase.pre(residual)`` returns
    ``x_pre [..., dim]``, ``post_mix [..., hc_mult, 1]``,
    ``comb_mix [..., hc_mult, hc_mult]``.
  * ``HCUnitBase.post(x, residual, post_mix, comb_mix)`` accepts
    ``x [..., dim]`` and returns ``[..., hc_mult, dim]``.
  * ``HCHeadBase.head(residual)`` returns ``[..., dim]``.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range


class HCMode(str, Enum):
    TILELANG = "tilelang"
    FALLBACK = "fallback"


class HCUnitBase(nn.Module):
    def __init__(
        self,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        *,
        dim: int,
        hc_mult: int,
        hc_sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
        layer_id: int = -1,
        name: str = "",
    ) -> None:
        super().__init__()
        self.fn = fn
        self.base = base
        self.scale = scale
        self.dim = dim
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.layer_id = layer_id
        self.name = name

    def pre(
        self, x: torch.Tensor, dbg_tag: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply HC readout.

        Args:
            x: ``[T, hc_mult, dim]`` or ``[B, S, hc_mult, dim]``.

        Returns:
            ``x_pre`` with shape ``[T, dim]`` or ``[B, S, dim]``;
            ``post_mix`` with shape ``[T, hc_mult, 1]`` or
            ``[B, S, hc_mult, 1]``;
            ``comb_mix`` with shape ``[T, hc_mult, hc_mult]`` or
            ``[B, S, hc_mult, hc_mult]``.
        """
        layer = f"L{self.layer_id:02d}" if self.layer_id >= 0 else "Lxx"
        name = self.name or "unit"
        with record_function_range(f"dsv4.hc.{layer}.{name}.pre"):
            y, post, comb = self._pre_impl(x, dbg_tag=dbg_tag)
        return y, post, comb

    def _pre_impl(
        self, x: torch.Tensor, dbg_tag: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Apply HC writeback.

        Args:
            x: sublayer output, ``[T, dim]`` or ``[B, S, dim]``.
            residual: original HC residual, ``[T, hc_mult, dim]`` or
                ``[B, S, hc_mult, dim]``.
            post: post mixer from ``pre``, ``[..., hc_mult, 1]``.
            comb: comb mixer from ``pre``, ``[..., hc_mult, hc_mult]``.

        Returns:
            Updated residual with the same shape as ``residual``.
        """
        layer = f"L{self.layer_id:02d}" if self.layer_id >= 0 else "Lxx"
        name = self.name or "unit"
        with record_function_range(f"dsv4.hc.{layer}.{name}.post"):
            out = self._post_impl(x, residual, post, comb)
        return out

    def _post_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class HCHeadBase(nn.Module):
    def __init__(
        self,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        *,
        dim: int,
        hc_mult: int,
        norm_eps: float,
        hc_eps: float,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.base = base
        self.scale = scale
        self.dim = dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps

    def head(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce HC streams.

        Args:
            x: ``[T, hc_mult, dim]`` or ``[B, S, hc_mult, dim]``.

        Returns:
            ``[T, dim]`` or ``[B, S, dim]`` with the same leading token layout.
        """
        with record_function_range("dsv4.hc.head"):
            out = self._head_impl(x)
        return out

    def _head_impl(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
