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


# Row budget for token-chunked HC fallback math on huge prefills. The
# pure-torch paths below materialize [T, hc_mult*dim] FP32 temporaries:
# at 1M tokens that is [1048320, 20480] fp32 = 79.98 GiB per temporary
# (n14, 2026-11-12: fatal GPU OOM with 54.95 GiB free, traceback frame
# delayed.py:143 `x.flatten(-2).float()`; n13 died identically). Every HC
# op here is token-local, so beyond this budget the same math runs over
# row slices. At or below it (decode graph captures, 8K smokes) the
# original single-shot expressions run verbatim - captured graphs stay
# byte-identical.
_HC_FALLBACK_CHUNK_ROWS = 16384


def _hc_fallback_chunk_rows() -> int:
    try:
        rows = int(
            os.environ.get("DSV4_HC_CHUNK_ROWS", "") or _HC_FALLBACK_CHUNK_ROWS
        )
    except ValueError:
        rows = _HC_FALLBACK_CHUNK_ROWS
    return max(1, rows)


# Cache for huge-prefill collapse readouts, keyed by exact output
# shape/device/dtype. See collapse_delayed's chunked branch for why the
# buffer must not be freshly allocated per call (n24: 10.00 GiB request
# failed with 9.91 GiB free mid-prefill).
_COLLAPSE_OUT_CACHE: dict = {}


def _collapse_out_cached(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, tuple((*x.shape[:-2], x.shape[-1])), x.dtype)
    buf = _COLLAPSE_OUT_CACHE.get(key)
    if buf is None:
        buf = torch.empty(key[1], dtype=x.dtype, device=x.device)
        _COLLAPSE_OUT_CACHE[key] = buf
    return buf


def collapse_delayed(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Read residual streams using FP32 coefficients and accumulation."""
    # TileLang/TileKernels are CUDA-only; ROCm uses the pure-torch path.
    if x.is_cuda and x.dtype == torch.bfloat16 and torch.version.hip is None:
        ops = _tile_ops()
        shape = x.shape
        return ops.mhc_pre_apply_mix(
            x.reshape(1, -1, shape[-2], shape[-1]),
            pre_mix.reshape(1, -1, shape[-2], 1).contiguous(),
        ).reshape(*shape[:-2], shape[-1])
    rows = _hc_fallback_chunk_rows()
    t = x.shape[0]
    if t <= rows:
        return (x.float() * pre_mix.unsqueeze(-1)).sum(dim=-2).to(x.dtype)
    # Token-chunked identical math (see _HC_FALLBACK_CHUNK_ROWS): the fp32
    # product temporary shrinks from [T, hc, dim] to [rows, hc, dim].
    # The readout itself comes from a per-forward-shape cached buffer:
    # n24 (2026-11-12) died requesting exactly this fresh [1048320, 5120]
    # bf16 10.00 GiB tensor with 9.91 GiB free - even after every prior
    # reduction, a NEW 10 GiB allocation per HC pre cannot be assembled
    # from the fragmented headroom mid-prefill. One cached buffer
    # (created at layer 0 while ~68 GiB is free, rewritten in place by
    # every later collapse, all same-stream ordered, consumers done
    # before the next rewrite) removes the allocation entirely.
    y = _collapse_out_cached(x)
    for begin in range(0, t, rows):
        end = min(begin + rows, t)
        y[begin:end] = (
            x[begin:end].float() * pre_mix[begin:end].unsqueeze(-1)
        ).sum(dim=-2).to(x.dtype)
    return y


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
        # TileLang mHC kernels are CUDA-only; the else branch below is the
        # complete pure-torch pre (F.linear + rsqrt + sinkhorn fallback).
        if x.is_cuda and x.dtype == torch.bfloat16 and torch.version.hip is None:
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
            w = self.fn.float()
            t = x.shape[0]
            rows = _hc_fallback_chunk_rows()
            if t <= rows:
                flattened = x.flatten(-2).float()
                mixes = F.linear(flattened, w) * torch.rsqrt(
                    flattened.square().mean(dim=-1, keepdim=True) + self.norm_eps
                )
            else:
                # Token-chunked identical math (see _HC_FALLBACK_CHUNK_ROWS):
                # bounded [rows, hc*dim] fp32 temporaries instead of one
                # [T, 20480] fp32 (79.98 GiB at 1M - the n13/n14 OOM frame).
                # mixes is [T, mix_width] fp32 (~100 MB at 1M), and the
                # sinkhorn split below is row-local on it.
                mixes = torch.empty(
                    (t, w.shape[0]), dtype=torch.float32, device=x.device
                )
                for begin in range(0, t, rows):
                    end = min(begin + rows, t)
                    flattened = x[begin:end].flatten(-2).float()
                    mixes[begin:end] = F.linear(flattened, w) * torch.rsqrt(
                        flattened.square().mean(dim=-1, keepdim=True)
                        + self.norm_eps
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
        # Diagnostic checkpoint: this frame is where the sticky capture error
        # surfaces (first error-checked torch op); probe at entry to tell
        # whether the illegal recording predates the HC post itself.
        from rtp_llm.models_py.modules.dsv41._awi_capture_probe import (
            probe as _awi_probe,
        )

        _awi_probe("hc_post")
        # TileLang mHC post kernel is CUDA-only; the torch path below is the
        # reference equivalent.
        if (
            residual.is_cuda
            and residual.dtype == torch.bfloat16
            and torch.version.hip is None
        ):
            return TileLangHCUnit._post_impl(self, x, residual, post, comb)
        if residual.is_cuda and torch.version.hip is not None:
            # The degenerate tiny batched GEMM ([T,4,4]^T @ [T,4,dim])
            # raises hipErrorInvalidConfiguration under HIP graph capture
            # (load23/24, exact frame under AMD_SERIALIZE_KERNEL=3) while
            # working eagerly. Unroll the hc_mult=4 combination as
            # elementwise multiply + reduce: identical math, capture-safe,
            # and bounded temporary memory (one [T,j,dim] product per lane).
            t = residual.shape[0]
            rows = _hc_fallback_chunk_rows()
            if t <= rows:
                residual_f = residual.float()
                comb_f = comb.float()
                # out[..., i, d] = sum_j comb[..., j, i] * residual[..., j, d]
                lanes = [
                    (residual_f * comb_f[..., lane : lane + 1]).sum(-2)
                    for lane in range(comb_f.shape[-1])
                ]
                mixed = torch.stack(lanes, dim=-2)
                return (post.float() * x.float().unsqueeze(-2) + mixed).to(
                    residual.dtype
                )
            # Token-chunked identical math (see _HC_FALLBACK_CHUNK_ROWS).
            # Written IN PLACE into the residual stream: a fresh [T,hc,dim]
            # bf16 output would transiently double the 39.99 GiB stream at
            # 1M and blow the ~20 GiB headroom left after the pre pass.
            # Every caller immediately rebinds its stream reference to this
            # return value (block.py fast paths: `x = attn_hc_post(...)`,
            # `residual = x`, `return ffn_hc_post(...)`); the deferred
            # decode-transition post only ever sees decode-sized t and keeps
            # the fresh-tensor path above. Chunk-local: reads of a row slice
            # complete (fp32 copy) before that slice is overwritten.
            comb_shape = comb.shape[-1]
            for begin in range(0, t, rows):
                end = min(begin + rows, t)
                residual_f = residual[begin:end].float()
                comb_f = comb[begin:end].float()
                lanes = [
                    (residual_f * comb_f[..., lane : lane + 1]).sum(-2)
                    for lane in range(comb_shape)
                ]
                mixed = torch.stack(lanes, dim=-2)
                residual[begin:end] = (
                    post[begin:end].float() * x[begin:end].float().unsqueeze(-2)
                    + mixed
                ).to(residual.dtype)
            return residual
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
