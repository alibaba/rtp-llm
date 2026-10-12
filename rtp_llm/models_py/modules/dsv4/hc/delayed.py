"""DeepSeek V4.1 delayed hyper-connections.

The coefficients projected at a sublayer are used to collapse the *next*
sublayer's residual. Post/residual mixing still belongs to the current
sublayer. The last FFN pre-mix is also the final head readout; V4.1 has no
learned ``hc_head`` weights.

Ported from RTP-LLM e4a9162f6f; standalone explicit-coefficient fixtures live
in delayed_reference.py.
"""

from __future__ import annotations

import importlib
import weakref
from functools import lru_cache
from typing import Optional, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv4.hc.base import HCUnitBase
from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import _hc_split_sinkhorn
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit


@lru_cache(maxsize=1)
def _tile_ops():
    # Set up libz3/TVM before importing the vendored TileLang kernels.
    from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

    return importlib.import_module(
        "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional"
    )


def collapse_delayed(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Read residual streams using FP32 coefficients and accumulation."""
    if tuple(pre_mix.shape) != tuple(x.shape[:-1]):
        raise ValueError(
            f"Delayed mHC pre-mix {tuple(pre_mix.shape)} does not match "
            f"residual {tuple(x.shape)}"
        )
    if x.is_cuda and x.dtype == torch.bfloat16:
        ops = _tile_ops()
        shape = x.shape
        return ops.mhc_pre_apply_mix(
            x.reshape(1, -1, shape[-2], shape[-1]),
            pre_mix.reshape(1, -1, shape[-2], 1).contiguous(),
        ).reshape(*shape[:-2], shape[-1])
    return (x.float() * pre_mix.unsqueeze(-1)).sum(dim=-2).to(x.dtype)


def _validate_ordered_collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> None:
    if x.ndim not in (3, 4) or min(x.shape[-2:]) <= 0:
        raise ValueError("Ordered delayed mHC requires [..., streams, dim] residuals")
    if pre_mix.shape != x.shape[:-1]:
        raise ValueError("Ordered delayed mHC coefficients must match residual streams")
    if x.dtype != torch.bfloat16 or pre_mix.dtype != torch.float32:
        raise TypeError(
            "Ordered delayed mHC requires BF16 residuals and FP32 coefficients"
        )
    if pre_mix.device != x.device:
        raise ValueError("Ordered delayed mHC operands must share a device")


def collapse_ordered_fp32(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Preserve separate FP32 multiplication and stream reduction before BF16."""
    _validate_ordered_collapse(x, pre_mix)
    return (pre_mix.unsqueeze(-1) * x.float()).sum(dim=-2).to(x.dtype)


def post_ordered_fp32(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Reduce source streams in FP32 before adding and writing the BF16 result."""
    if residual.ndim not in (3, 4) or min(residual.shape[-2:]) <= 0:
        raise ValueError("Ordered delayed mHC post requires [..., streams, dim]")
    leading = residual.shape[:-2]
    streams, dim = residual.shape[-2:]
    if (
        x.shape != leading + (dim,)
        or post.shape != leading + (streams, 1)
        or comb.shape != leading + (streams, streams)
    ):
        raise ValueError("Ordered delayed mHC post operands must match residuals")
    if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise TypeError("Ordered delayed mHC post requires BF16 activations")
    if post.dtype != torch.float32 or comb.dtype != torch.float32:
        raise TypeError("Ordered delayed mHC post requires FP32 coefficients")
    if any(value.device != residual.device for value in (x, post, comb)):
        raise ValueError("Ordered delayed mHC post operands must share a device")
    mixed = (comb.unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(dim=-3)
    result = post * x.float().unsqueeze(-2) + mixed
    residual.copy_(result.to(residual.dtype))
    return residual


def post_delayed(owner, x, residual, post, comb):
    if residual.is_cuda and residual.dtype == torch.bfloat16:
        return TileLangHCUnit._post_impl(owner, x, residual, post, comb)
    return (
        post.float() * x.float().unsqueeze(-2)
        + torch.matmul(comb.float().transpose(-1, -2), residual.float())
    ).to(residual.dtype)


class DelayedHCPredecessor(Protocol):
    @property
    def pre_mix_out(self) -> Optional[torch.Tensor]:
        """Coefficients produced locally or imported at a pipeline boundary."""
        ...


class DelayedHCUnit(HCUnitBase):
    def __init__(
        self,
        *args,
        collapse_mode: Optional[str] = None,
        post_mode: Optional[str] = None,
        prenorm_mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        if collapse_mode is not None and (
            type(collapse_mode) is not str or collapse_mode != "ordered_fp32"
        ):
            raise ValueError("Delayed mHC collapse_mode must be ordered_fp32 or None")
        if post_mode is not None and (
            type(post_mode) is not str or post_mode != "ordered_fp32"
        ):
            raise ValueError("Delayed mHC post_mode must be ordered_fp32 or None")
        if prenorm_mode is not None and (
            type(prenorm_mode) is not str or prenorm_mode != "fp32"
        ):
            raise ValueError("Delayed mHC prenorm_mode must be fp32 or None")
        if prenorm_mode is not None:
            from rtp_llm.models_py.modules.dsv4.hc.v41_prenorm import (
                require_fp32_prenorm_precision,
            )

            require_fp32_prenorm_precision()
        super().__init__(*args, **kwargs)
        self._collapse_mode = collapse_mode
        self._post_mode = post_mode
        self._prenorm_mode = prenorm_mode
        mix_width = self.hc_mult * (self.hc_mult + 2)
        expected_fn_shape = (mix_width, self.hc_mult * self.dim)
        if tuple(self.fn.shape) != expected_fn_shape:
            raise ValueError(
                f"Delayed mHC fn shape {tuple(self.fn.shape)} != {expected_fn_shape}"
            )
        if self.base.numel() != mix_width or self.scale.numel() != 3:
            raise ValueError("Delayed mHC requires mix-width base and three scales")
        if any(t.dtype != torch.float32 for t in (self.fn, self.base, self.scale)):
            raise TypeError("Delayed mHC fn, base and scale must be FP32")
        # AtomicWeight gives named scale tensors a trailing singleton axis.
        # mHC scales are three scalar coefficients, unlike quantization scales;
        # restore the kernel's 1-D contract once, before graph capture.
        self.base = self.base.reshape(mix_width).contiguous()
        self.scale = self.scale.reshape(3).contiguous()
        self._previous_ref: Optional[weakref.ReferenceType[DelayedHCPredecessor]] = None
        self.pre_mix_out: Optional[torch.Tensor] = None

    @property
    def collapse_mode(self) -> Optional[str]:
        return self._collapse_mode

    @property
    def post_mode(self) -> Optional[str]:
        return self._post_mode

    @property
    def prenorm_mode(self) -> Optional[str]:
        return self._prenorm_mode

    def collapse(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        if self._collapse_mode == "ordered_fp32":
            return collapse_ordered_fp32(x, pre_mix)
        return collapse_delayed(x, pre_mix)

    def set_previous(self, previous: Optional[DelayedHCPredecessor]) -> None:
        # Avoid registering predecessors as nested modules or creating cycles.
        self._previous_ref = weakref.ref(previous) if previous is not None else None

    def _pre_impl(self, x: torch.Tensor, dbg_tag=None):
        shape = x.shape
        previous_mix = None
        if self._prenorm_mode == "fp32":
            from rtp_llm.models_py.modules.dsv4.hc.v41_prenorm import (
                prenorm_fp32,
                require_fp32_prenorm_precision,
            )

            require_fp32_prenorm_precision()
            if x.dtype != torch.bfloat16:
                raise TypeError("FP32 delayed mHC prenorm requires BF16 residuals")
            if any(t.device != x.device for t in (self.fn, self.base, self.scale)):
                raise ValueError("FP32 delayed mHC prenorm weights must share a device")
        if self._collapse_mode == "ordered_fp32":
            if x.dtype != torch.bfloat16:
                raise TypeError("Ordered delayed mHC requires BF16 residuals")
            if any(t.device != x.device for t in (self.fn, self.base, self.scale)):
                raise ValueError(
                    "Ordered delayed mHC weights must share the residual device"
                )
            if self._previous_ref is not None:
                previous = self._previous_ref()
                if previous is None or previous.pre_mix_out is None:
                    raise RuntimeError("Delayed mHC predecessor has not run")
                previous_mix = previous.pre_mix_out
                _validate_ordered_collapse(x, previous_mix)
        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and (
                (self._collapse_mode is None and self._prenorm_mode is None)
                or x.numel() != 0
            )
        ):
            from rtp_llm.models_py.modules.dsv4.hc.v41_prenorm import prenorm

            ops = _tile_ops()
            residual = x.reshape(1, -1, self.hc_mult, self.dim)
            if self._collapse_mode == "ordered_fp32":
                residual = residual.contiguous()
            mixes = (
                prenorm_fp32(residual, self.fn, self.norm_eps)
                if self._prenorm_mode == "fp32"
                else prenorm(residual, self.fn, self.norm_eps)
            )
            if mixes is None:
                mixes = ops.mhc_pre_norm_fn(
                    residual, self.fn, None, self.norm_eps, n_splits=1
                )
            pre, post, comb = ops.mhc_pre_split_mixes(
                mixes, self.scale, self.base, self.hc_mult, 2.0, self.hc_eps
            )
            comb = ops.sinkhorn_normalize(
                comb, repeat=self.hc_sinkhorn_iters, eps=self.hc_eps
            )
            self.pre_mix_out = pre.reshape(*shape[:-1]).contiguous()
            post = post.reshape(*shape[:-1], 1)
            comb = comb.reshape(*shape[:-2], self.hc_mult, self.hc_mult)
        else:
            if self._prenorm_mode == "fp32":
                mixes = prenorm_fp32(x, self.fn, self.norm_eps)
            else:
                flattened = x.flatten(-2).float()
                mixes = F.linear(flattened, self.fn.float()) * torch.rsqrt(
                    flattened.square().mean(dim=-1, keepdim=True) + self.norm_eps
                )
            pre, post, comb = _hc_split_sinkhorn(
                mixes,
                self.scale,
                self.base,
                self.hc_mult,
                self.hc_sinkhorn_iters,
                self.hc_eps,
            )
            self.pre_mix_out = pre
            post = post.unsqueeze(-1)

        if self._previous_ref is None:
            # Identity readout at entry selects lane zero, not the new mix.
            # A single row is already contiguous; copy so in-place norm in
            # Block's fast path cannot overwrite the residual lane.
            y = x[..., 0, :].clone(memory_format=torch.contiguous_format)
        else:
            if previous_mix is None:
                previous = self._previous_ref()
                if previous is None or previous.pre_mix_out is None:
                    raise RuntimeError("Delayed mHC predecessor has not run")
                previous_mix = previous.pre_mix_out
            y = self.collapse(x, previous_mix)
        return y, post, comb

    def _post_impl(self, x, residual, post, comb):
        if self._post_mode == "ordered_fp32":
            return post_ordered_fp32(x, residual, post, comb)
        return post_delayed(self, x, residual, post, comb)


class DelayedHCHead(nn.Module):
    def __init__(self, last_ffn: DelayedHCUnit) -> None:
        super().__init__()
        self._last_ffn_ref = weakref.ref(last_ffn)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        unit = self._last_ffn_ref()
        if unit is None or unit.pre_mix_out is None:
            raise RuntimeError("V4.1 final FFN pre-mix is unavailable")
        return unit.collapse(hidden, unit.pre_mix_out)


__all__ = [
    "DelayedHCUnit",
    "DelayedHCHead",
    "collapse_delayed",
    "collapse_ordered_fp32",
    "post_ordered_fp32",
]
