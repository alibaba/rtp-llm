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

# Row-slice the fp32-upcast delayed-HC post when >0: bounds its [rows, hc,
# dim] fp32 temporaries on tight-HBM ranks (row-independent, bitwise
# identical). Default 0 keeps the historical single-expression path.
_HC_POST_ROW_CHUNK = int(os.environ.get("DSV41_HC_POST_ROW_CHUNK", "0"))


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


def can_fuse_big_pre() -> bool:
    backend = os.environ.get("DSV4_MHC_PRE_GEMM_BACKEND", "").strip().lower()
    if backend in ("tilelang_single", "tilelang", "single"):
        return True
    if backend in ("", "auto", "deepgemm", "dg"):
        from rtp_llm.models_py.modules.dsv41.hc.v41_prenorm import _has_prenorm_gemm

        return _has_prenorm_gemm()
    return False


@lru_cache(maxsize=None)
def _tilelang_mhc_post_supported(index) -> bool:
    """TileLang's mhc_post kernel rejects pre-SM100 devices at runtime.

    Observed on L20/sm89: "TileLang mhc_post failed" for a small bf16
    residual while the torch reference below produces identical math.
    SM100+ keeps the TileLang path unchanged.
    """
    return torch.cuda.get_device_capability(index)[0] >= 10


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
            from rtp_llm.models_py.modules.dsv41.hc.v41_prenorm import (
                is_supported,
                prenorm,
            )

            ops = _tile_ops()
            residual = x.reshape(1, -1, self.hc_mult, self.dim)
            previous = self._previous_ref() if self._previous_ref is not None else None
            if (
                previous is not None
                and not torch.is_grad_enabled()
                and can_fuse_big_pre()
                and residual.shape[-2:] == (4, 5120)
                and residual.is_contiguous()
                and self.fn.is_contiguous()
                and torch.cuda.get_device_capability(x.device)[0] == 10
                and (residual.shape[1] > 64 or is_supported(residual, self.fn))
            ):
                use_deepgemm = is_supported(residual, self.fn)
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
                    gemm_backend="deepgemm" if use_deepgemm else None,
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
        if (
            residual.is_cuda
            and residual.dtype == torch.bfloat16
            and _tilelang_mhc_post_supported(residual.device)
        ):
            return TileLangHCUnit._post_impl(self, x, residual, post, comb)
        if _HC_POST_ROW_CHUNK > 0 and x.shape[0] > _HC_POST_ROW_CHUNK:
            # The fp32-upcast post materializes several [rows, hc, dim] fp32
            # temporaries (42MB each at 512 rows; load52probe rank0 died on
            # one of them with 39.88MiB free). Rows are independent (per-row
            # scale plus a per-row [hc, hc] matmul), so row slicing is
            # bitwise identical while bounding the transient for ranks whose
            # post-weight headroom is tens of MiB. Default 0 keeps the
            # single-expression path.
            outs = []
            for r0 in range(0, x.shape[0], _HC_POST_ROW_CHUNK):
                r1 = min(r0 + _HC_POST_ROW_CHUNK, x.shape[0])
                outs.append(
                    (
                        post[r0:r1].float() * x[r0:r1].float().unsqueeze(-2)
                        + torch.matmul(
                            comb[r0:r1].float().transpose(-1, -2),
                            residual[r0:r1].float(),
                        )
                    ).to(residual.dtype)
                )
            return torch.cat(outs, 0)
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
