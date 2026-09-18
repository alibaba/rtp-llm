"""DeepSeek V4.1 delayed hyper-connections.

The coefficients projected at a sublayer are used to collapse the *next*
sublayer's residual. Post/residual mixing still belongs to the current
sublayer. The last FFN pre-mix is also the final head readout; V4.1 has no
learned ``hc_head`` weights.
"""

from __future__ import annotations

import importlib
import weakref
from functools import lru_cache
from typing import Optional

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


class DelayedHCUnit(HCUnitBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        self._previous_ref = None
        self.pre_mix_out: Optional[torch.Tensor] = None

    def set_previous(self, previous: Optional["DelayedHCUnit"]) -> None:
        # Avoid registering predecessors as nested modules or creating cycles.
        self._previous_ref = weakref.ref(previous) if previous is not None else None

    def _pre_impl(self, x: torch.Tensor, dbg_tag=None):
        shape = x.shape
        if x.is_cuda and x.dtype == torch.bfloat16:
            from rtp_llm.models_py.modules.dsv4.hc.v41_prenorm import prenorm

            ops = _tile_ops()
            residual = x.reshape(1, -1, self.hc_mult, self.dim)
            mixes = prenorm(residual, self.fn, self.norm_eps)
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
            y = x[..., 0, :].contiguous()
        else:
            previous = self._previous_ref()
            if previous is None or previous.pre_mix_out is None:
                raise RuntimeError("Delayed mHC predecessor has not run")
            y = collapse_delayed(x, previous.pre_mix_out)
        return y, post, comb

    def _post_impl(self, x, residual, post, comb):
        if residual.is_cuda and residual.dtype == torch.bfloat16:
            return TileLangHCUnit._post_impl(self, x, residual, post, comb)
        return (
            post.float() * x.float().unsqueeze(-2)
            + torch.matmul(comb.float().transpose(-1, -2), residual.float())
        ).to(residual.dtype)


class DelayedHCHead(nn.Module):
    def __init__(self, last_ffn: DelayedHCUnit) -> None:
        super().__init__()
        self._last_ffn_ref = weakref.ref(last_ffn)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        unit = self._last_ffn_ref()
        if unit is None or unit.pre_mix_out is None:
            raise RuntimeError("V4.1 final FFN pre-mix is unavailable")
        return collapse_delayed(hidden, unit.pre_mix_out)


__all__ = ["DelayedHCUnit", "DelayedHCHead", "collapse_delayed"]
