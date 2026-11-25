"""DeepSeek V4.1 delayed hyper-connections.

The coefficients projected at a sublayer are used to collapse the *next*
sublayer's residual. Post/residual mixing still belongs to the current
sublayer. The last FFN pre-mix is also the final head readout; V4.1 has no
learned ``hc_head`` weights.
"""

from __future__ import annotations

import importlib
import os
import weakref
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv41.hc.base import HCUnitBase
from rtp_llm.models_py.modules.dsv41.hc.fallback_impl import _hc_split_sinkhorn
from rtp_llm.models_py.modules.dsv41.hc.tilelang_impl import TileLangHCUnit


@lru_cache(maxsize=1)
def _tile_ops():
    # Set up libz3/TVM before importing the vendored TileLang kernels.
    from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

    return importlib.import_module(
        "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional"
    )


def collapse_delayed(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Read residual streams using FP32 coefficients and accumulation."""
    if x.is_cuda and x.dtype == torch.bfloat16:
        ops = _tile_ops()
        shape = x.shape
        return ops.mhc_pre_apply_mix(
            x.reshape(1, -1, shape[-2], shape[-1]),
            pre_mix.reshape(1, -1, shape[-2], 1).contiguous(),
        ).reshape(*shape[:-2], shape[-1])
    return (x.float() * pre_mix.unsqueeze(-1)).sum(dim=-2).to(x.dtype)


_EXPLICIT_TILELANG_BACKENDS = frozenset(
    ("tilelang_single", "tilelang", "single", "tilelang_splitk", "splitk")
)


def can_fuse_big_pre() -> bool:
    backend = os.environ.get("DSV4_MHC_PRE_GEMM_BACKEND", "").strip().lower()
    if backend in _EXPLICIT_TILELANG_BACKENDS:
        return True
    if backend in ("", "auto", "deepgemm", "dg"):
        from rtp_llm.models_py.modules.dsv41.hc.v41_prenorm import _has_prenorm_gemm

        return _has_prenorm_gemm()
    return False


# SM90 decode fuses the whole pre chain into two kernels. Measured on L20Z,
# DeepGEMM's combined projection/square-sum primitive is at least as fast as
# the TileLang split-K planner across the decode window (17.4 us vs 18.8 us at
# 128 rows, 23.3 us vs 25.2 us at 264, 39.6 us vs 42.2 us at 512; only 256 rows
# is a ~1 us split-K win), so the fused chain uses it whenever it is installed
# and split-K otherwise. Large prefill keeps the standalone chain so a fused
# launch never becomes the long-sequence bottleneck.
_SM90_BIG_PRE_MAX_ROWS = 512


def _sm90_big_pre_rows(rows: int, capability: tuple[int, int]) -> bool:
    return capability[0] == 9 and 64 < rows <= _SM90_BIG_PRE_MAX_ROWS


def _big_pre_backend(capability: tuple[int, int], deepgemm_ok: bool):
    """Fused-pre GEMM backend, or ``None`` to let the op resolve its default.

    SM100 keeps its established DeepGEMM-first contract. On SM90 an explicit
    TileLang selection keeps the single-split kernel; otherwise DeepGEMM is
    used when its fused row budget accepts the batch and split-K covers the
    rest (no DeepGEMM install, or an unsupported geometry).
    """
    if capability[0] == 9:
        if deepgemm_ok:
            return "deepgemm"
        requested = os.environ.get("DSV4_MHC_PRE_GEMM_BACKEND", "").strip().lower()
        if requested in _EXPLICIT_TILELANG_BACKENDS:
            return None
        return "tilelang_splitk"
    return "deepgemm" if deepgemm_ok else None


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
        self._mega_mhc_supported: Optional[bool] = None
        self.pre_mix_out: Optional[torch.Tensor] = None

    def set_previous(self, previous: Optional["DelayedHCUnit"]) -> None:
        # Avoid registering predecessors as nested modules or creating cycles.
        self._mega_mhc_supported = None
        self._previous_ref = weakref.ref(previous) if previous is not None else None

    def _pre_impl(self, x: torch.Tensor, dbg_tag=None):
        shape = x.shape
        if x.is_cuda and x.dtype == torch.bfloat16:
            from rtp_llm.models_py.modules.dsv41.hc.v41_prenorm import (
                is_supported,
                prenorm,
            )

            ops = _tile_ops()
            residual = x.reshape(1, -1, self.hc_mult, self.dim)
            previous = self._previous_ref() if self._previous_ref is not None else None
            capability = torch.cuda.get_device_capability(x.device)
            rows = residual.shape[1]
            # The fused chain has no standalone reduction, so on SM90 it may
            # spend DeepGEMM across the whole decode window.
            deepgemm_ok = is_supported(
                residual,
                self.fn,
                max_tokens=(
                    _SM90_BIG_PRE_MAX_ROWS if capability[0] == 9 else None
                ),
            )
            if (
                previous is not None
                and not torch.is_grad_enabled()
                and can_fuse_big_pre()
                and residual.shape[-2:] == (4, 5120)
                and residual.is_contiguous()
                and self.fn.is_contiguous()
                and (
                    (capability[0] == 10 and (rows > 64 or deepgemm_ok))
                    or _sm90_big_pre_rows(rows, capability)
                )
            ):
                pre, post, comb, y = ops.mhc_pre_big_fuse(
                    residual,
                    self.fn,
                    self.scale,
                    self.base,
                    rms_eps=self.norm_eps,
                    mhc_pre_eps=self.hc_eps,
                    mhc_sinkhorn_eps=self.hc_eps,
                    mhc_post_mult_value=2.0,
                    sinkhorn_repeat=self.hc_sinkhorn_iters,
                    previous_pre_mix=previous.pre_mix_out,
                    gemm_backend=_big_pre_backend(capability, deepgemm_ok),
                )
                self.pre_mix_out = pre.reshape(*shape[:-1])
                return (
                    y.reshape(*shape[:-2], self.dim),
                    post.reshape(*shape[:-1], 1),
                    comb.reshape(*shape[:-2], self.hc_mult, self.hc_mult),
                )
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
        return collapse_delayed(hidden, unit.pre_mix_out)


__all__ = ["DelayedHCUnit", "DelayedHCHead", "collapse_delayed"]
