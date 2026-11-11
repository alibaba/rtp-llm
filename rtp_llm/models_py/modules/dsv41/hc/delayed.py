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
from rtp_llm.models_py.modules.dsv41.hc.fallback_impl import (
    _hc_fallback_chunk_tokens,
    _hc_split_sinkhorn,
)
from rtp_llm.models_py.modules.dsv41.hc.tilelang_impl import TileLangHCUnit


@lru_cache(maxsize=1)
def _tile_ops():
    # Set up libz3/TVM before importing the vendored TileLang kernels.
    from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

    return importlib.import_module(
        "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional"
    )


def _tilelang_supported(x: torch.Tensor) -> bool:
    """TileLang mHC kernels target Hopper/Blackwell; SM80 uses torch math."""
    if not x.is_cuda or x.dtype != torch.bfloat16:
        return False
    if torch.version.hip is not None:
        return False
    try:
        return torch.cuda.get_device_capability(x.device)[0] >= 9
    except Exception:
        return False


def collapse_delayed(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Read residual streams using FP32 coefficients and accumulation."""
    if _tilelang_supported(x):
        ops = _tile_ops()
        shape = x.shape
        return ops.mhc_pre_apply_mix(
            x.reshape(1, -1, shape[-2], shape[-1]),
            pre_mix.reshape(1, -1, shape[-2], 1).contiguous(),
        ).reshape(*shape[:-2], shape[-1])
    shape = x.shape
    flat = x.reshape(-1, shape[-2], shape[-1])
    mix = pre_mix.reshape(-1, shape[-2])
    out = torch.empty(
        (flat.shape[0], shape[-1]), dtype=x.dtype, device=x.device
    )
    chunk = _hc_fallback_chunk_tokens()
    for start in range(0, flat.shape[0], chunk):
        end = min(start + chunk, flat.shape[0])
        out[start:end] = (
            flat[start:end].float() * mix[start:end].float().unsqueeze(-1)
        ).sum(dim=-2).to(x.dtype)
    return out.view(*shape[:-2], shape[-1])


def can_fuse_big_pre() -> bool:
    backend = os.environ.get("DSV4_MHC_PRE_GEMM_BACKEND", "").strip().lower()
    if backend in ("tilelang_single", "tilelang", "single"):
        return True
    if backend in ("", "auto", "deepgemm", "dg"):
        from rtp_llm.models_py.modules.dsv41.hc.v41_prenorm import _has_prenorm_gemm

        return _has_prenorm_gemm()
    return False


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
        if _tilelang_supported(x):
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
            previous = self._previous_ref() if self._previous_ref is not None else None
            y, post, comb = self._pre_torch_chunked(x, previous)
            self.pre_mix_out = self._last_pre_mix
            return y, post, comb

        if self._previous_ref is None:
            # Identity readout at entry selects lane zero, not the new mix.
            y = x[..., 0, :].contiguous()
        else:
            previous = self._previous_ref()
            y = collapse_delayed(x, previous.pre_mix_out)
        return y, post, comb

    def _pre_torch_chunked(self, x, previous):
        """Chunked FP32 reference pre-mix for SM80 (no TileLang, no full
        ``[T, hc*dim]`` fp32 materialization)."""
        shape = x.shape
        hc = self.hc_mult
        dim = self.dim
        flat = x.reshape(-1, hc, dim)
        tokens = flat.shape[0]
        chunk = _hc_fallback_chunk_tokens()
        out = torch.empty((tokens, dim), dtype=x.dtype, device=x.device)
        pre_all = torch.empty((tokens, hc), dtype=torch.float32, device=x.device)
        post_all = torch.empty((tokens, hc), dtype=torch.float32, device=x.device)
        comb_all = torch.empty((tokens, hc, hc), dtype=torch.float32, device=x.device)
        prev = None
        if previous is not None:
            prev = previous.pre_mix_out.reshape(-1, hc)
        for start in range(0, tokens, chunk):
            end = min(start + chunk, tokens)
            xc = flat[start:end]
            flattened = xc.flatten(-2).float()
            mixes = F.linear(flattened, self.fn.float()) * torch.rsqrt(
                flattened.square().mean(dim=-1, keepdim=True) + self.norm_eps
            )
            pre, post, comb = _hc_split_sinkhorn(
                mixes,
                self.scale,
                self.base,
                hc,
                self.hc_sinkhorn_iters,
                self.hc_eps,
            )
            pre_all[start:end] = pre
            post_all[start:end] = post
            comb_all[start:end] = comb
            if prev is None:
                out[start:end] = xc[..., 0, :]
            else:
                out[start:end] = (
                    xc.float() * prev[start:end].unsqueeze(-1)
                ).sum(dim=-2).to(x.dtype)
        self._last_pre_mix = pre_all.view(*shape[:-1])
        return (
            out.view(*shape[:-2], dim),
            post_all.view(*shape[:-1], 1),
            comb_all.view(*shape[:-2], hc, hc),
        )

    def _post_impl(self, x, residual, post, comb):
        if _tilelang_supported(residual):
            return TileLangHCUnit._post_impl(self, x, residual, post, comb)
        # In-place chunked reference: the caller discards ``residual`` after
        # this sublayer, matching the TileLang ``out=residual`` contract and
        # avoiding a second 40 GiB allocation at 1M tokens.
        shape = residual.shape
        hc, dim = shape[-2], shape[-1]
        flat_x = x.reshape(-1, dim)
        flat_res = residual.reshape(-1, hc, dim)
        flat_post = post.reshape(-1, hc)
        flat_comb = comb.reshape(-1, hc, hc)
        chunk = _hc_fallback_chunk_tokens()
        for start in range(0, flat_x.shape[0], chunk):
            end = min(start + chunk, flat_x.shape[0])
            first = flat_post[start:end].float().unsqueeze(-1) * flat_x[
                start:end
            ].float().unsqueeze(-2)
            second = torch.matmul(
                flat_comb[start:end].float().transpose(-1, -2),
                flat_res[start:end].float(),
            )
            flat_res[start:end].copy_((first + second).to(residual.dtype))
        return residual


class DelayedHCHead(nn.Module):
    def __init__(self, last_ffn: DelayedHCUnit) -> None:
        super().__init__()
        self._last_ffn_ref = weakref.ref(last_ffn)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        unit = self._last_ffn_ref()
        return collapse_delayed(hidden, unit.pre_mix_out)


__all__ = ["DelayedHCUnit", "DelayedHCHead", "collapse_delayed"]
