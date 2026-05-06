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

from rtp_llm.models_py.modules.dsv4._nvtx import nvtx_range


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

    def _check_residual_shape(self, x: torch.Tensor, arg_name: str) -> Tuple[int, ...]:
        if x.dim() not in (3, 4):
            raise ValueError(
                f"{self.__class__.__name__}.{arg_name} expected [T, hc, dim] "
                f"or [B, S, hc, dim], got shape={tuple(x.shape)}"
            )
        if int(x.shape[-2]) != self.hc_mult or int(x.shape[-1]) != self.dim:
            raise ValueError(
                f"{self.__class__.__name__}.{arg_name} expected trailing "
                f"[hc, dim]=[{self.hc_mult}, {self.dim}], got {tuple(x.shape[-2:])}"
            )
        return tuple(int(v) for v in x.shape[:-2])

    def _check_pre_output(
        self,
        leading: Tuple[int, ...],
        y: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> None:
        expected_y = leading + (self.dim,)
        expected_post = leading + (self.hc_mult, 1)
        expected_comb = leading + (self.hc_mult, self.hc_mult)
        if tuple(y.shape) != expected_y:
            raise ValueError(
                f"{self.__class__.__name__}.pre returned y shape={tuple(y.shape)}, "
                f"expected {expected_y}"
            )
        if tuple(post.shape) != expected_post:
            raise ValueError(
                f"{self.__class__.__name__}.pre returned post shape={tuple(post.shape)}, "
                f"expected {expected_post}"
            )
        if tuple(comb.shape) != expected_comb:
            raise ValueError(
                f"{self.__class__.__name__}.pre returned comb shape={tuple(comb.shape)}, "
                f"expected {expected_comb}"
            )

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
        leading = self._check_residual_shape(x, "pre input")
        layer = f"L{self.layer_id:02d}" if self.layer_id >= 0 else "Lxx"
        name = self.name or "unit"
        with nvtx_range(f"dsv4.hc.{layer}.{name}.pre"):
            y, post, comb = self._pre_impl(x, dbg_tag=dbg_tag)
        self._check_pre_output(leading, y, post, comb)
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
        leading = self._check_residual_shape(residual, "post residual")
        expected_x = leading + (self.dim,)
        expected_post = leading + (self.hc_mult, 1)
        expected_comb = leading + (self.hc_mult, self.hc_mult)
        if tuple(x.shape) != expected_x:
            raise ValueError(
                f"{self.__class__.__name__}.post expected x shape={expected_x}, "
                f"got {tuple(x.shape)}"
            )
        if tuple(post.shape) != expected_post:
            raise ValueError(
                f"{self.__class__.__name__}.post expected post shape={expected_post}, "
                f"got {tuple(post.shape)}"
            )
        if tuple(comb.shape) != expected_comb:
            raise ValueError(
                f"{self.__class__.__name__}.post expected comb shape={expected_comb}, "
                f"got {tuple(comb.shape)}"
            )
        layer = f"L{self.layer_id:02d}" if self.layer_id >= 0 else "Lxx"
        name = self.name or "unit"
        with nvtx_range(f"dsv4.hc.{layer}.{name}.post"):
            out = self._post_impl(x, residual, post, comb)
        if tuple(out.shape) != tuple(residual.shape):
            raise ValueError(
                f"{self.__class__.__name__}.post returned shape={tuple(out.shape)}, "
                f"expected {tuple(residual.shape)}"
            )
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

    def _check_residual_shape(self, x: torch.Tensor, arg_name: str) -> Tuple[int, ...]:
        if x.dim() not in (3, 4):
            raise ValueError(
                f"{self.__class__.__name__}.{arg_name} expected [T, hc, dim] "
                f"or [B, S, hc, dim], got shape={tuple(x.shape)}"
            )
        if int(x.shape[-2]) != self.hc_mult or int(x.shape[-1]) != self.dim:
            raise ValueError(
                f"{self.__class__.__name__}.{arg_name} expected trailing "
                f"[hc, dim]=[{self.hc_mult}, {self.dim}], got {tuple(x.shape[-2:])}"
            )
        return tuple(int(v) for v in x.shape[:-2])

    def head(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce HC streams.

        Args:
            x: ``[T, hc_mult, dim]`` or ``[B, S, hc_mult, dim]``.

        Returns:
            ``[T, dim]`` or ``[B, S, dim]`` with the same leading token layout.
        """
        leading = self._check_residual_shape(x, "head input")
        with nvtx_range("dsv4.hc.head"):
            out = self._head_impl(x)
        expected = leading + (self.dim,)
        if tuple(out.shape) != expected:
            raise ValueError(
                f"{self.__class__.__name__}.head returned shape={tuple(out.shape)}, "
                f"expected {expected}"
            )
        return out

    def _head_impl(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
