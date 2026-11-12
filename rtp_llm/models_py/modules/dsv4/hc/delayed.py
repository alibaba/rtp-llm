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
import os
import weakref
from functools import lru_cache
from typing import Optional, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv4._nvtx_phases import nvtx_phase
from rtp_llm.models_py.modules.dsv4._v41_graph_probe import (
    premix_pin_enabled as _premix_pin_enabled,
)
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


def _mhc_pre_big_fuse_enabled() -> bool:
    """R54: precision-class fused delayed mHC pre (DSV41_MHC_PRE_BIG_FUSE).

    Default OFF. This fuses the prenorm GEMM + split + sinkhorn + the delayed
    readout into one mhc_pre_big_fuse launch per boundary. PRECISION-CLASS: the
    fused epilogue keeps FP32 partials in registers (upstream test rtol 0.02), so
    it is NOT byte-exact with the unfused path — only engage under an accuracy bar.
    """
    return os.environ.get("DSV41_MHC_PRE_BIG_FUSE", "0") == "1"


_V54_ENGAGED = [False]


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
        # R25/O1: pending side-stream stats join (event recorded after the
        # stats kernels on the side stream; consumers join via join_stats()).
        self._stats_event: Optional[torch.cuda.Event] = None
        self._stats_ev_in: Optional[torch.cuda.Event] = None

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

    def join_stats(self) -> None:
        """R25/O1: event-join pending side-stream stats into the CURRENT stream.

        Must run before any consumer of this boundary's post/comb/pre_mix_out:
        _post_impl, DelayedHCHead.head and the PP frame export. No-op (one
        attribute read) when the side stream is off or already joined.
        """
        event = self._stats_event
        if event is None:
            return
        ev_in = self._stats_ev_in
        self._stats_event = None
        self._stats_ev_in = None
        already = None
        if ev_in is not None:
            from rtp_llm.models_py.modules.dsv4.hc import _v41_mhc_side_stream

            if _v41_mhc_side_stream.mhc_side_stats_enabled():
                already = event.query()
        torch.cuda.current_stream().wait_event(event)
        if already is not None:
            from rtp_llm.models_py.modules.dsv4.hc import _v41_mhc_side_stream

            # Diagnostics-only host wait: elapsed_time requires BOTH events
            # complete; wait_event only enqueued the GPU-side join. Timed
            # arms never take this branch (stats flag off => already None).
            event.synchronize()
            _v41_mhc_side_stream.mhc_side_note_join(already, ev_in.elapsed_time(event))

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

            # R54: fused delayed mHC pre (prenorm GEMM + split + sinkhorn +
            # the delayed readout) in one mhc_pre_big_fuse launch. Engages for
            # boundaries with a linked predecessor (layers 2+). Precision-class.
            if (
                _mhc_pre_big_fuse_enabled()
                and not torch.is_grad_enabled()
                and not torch.cuda.is_current_stream_capturing()
                and self._previous_ref is not None
            ):
                previous = self._previous_ref()
                if previous is not None and previous.pre_mix_out is not None:
                    from rtp_llm.models_py.modules.dsv4.hc.v41_prenorm import (
                        _has_prenorm_gemm,
                        prepare_tf32_weight,
                    )

                    small_deepgemm = residual.shape[1] <= 64 and _has_prenorm_gemm()
                    pre, post, comb, y = ops.mhc_pre_big_fuse(
                        residual,
                        prepare_tf32_weight(self.fn) if small_deepgemm else self.fn,
                        self.scale,
                        self.base,
                        rms_eps=self.norm_eps,
                        mhc_pre_eps=self.hc_eps,
                        mhc_sinkhorn_eps=self.hc_eps,
                        mhc_post_mult_value=2.0,
                        sinkhorn_repeat=self.hc_sinkhorn_iters,
                        previous_pre_mix=previous.pre_mix_out,
                        gemm_backend="deepgemm" if small_deepgemm else None,
                    )
                    self.pre_mix_out = pre.reshape(*shape[:-1])
                    if not _V54_ENGAGED[0]:
                        _V54_ENGAGED[0] = True
                        print(
                            "[V54MHCBIGFUSE] engaged rows=%d dim=%d hc=%d backend=%s"
                            % (
                                residual.shape[1],
                                self.dim,
                                self.hc_mult,
                                "deepgemm" if small_deepgemm else "tilelang_single",
                            ),
                            flush=True,
                        )
                    return (
                        y.reshape(*shape[:-2], self.dim),
                        post.reshape(*shape[:-1], 1),
                        comb.reshape(*shape[:-2], self.hc_mult, self.hc_mult),
                    )

            def _stats():
                # Kernels 3-6 of the boundary (fn_mul/norm, split_mixes,
                # sinkhorn) + the pre_mix_out materialization. They read only
                # the new residual and weights; consumers are this sublayer's
                # post and the NEXT boundary's collapse/PP export.
                from rtp_llm.models_py.modules.dsv4._v41_mhc_mega_fused import (
                    mhc_delayed_stats_fused,
                )
                from rtp_llm.models_py.modules.dsv4._v41_mhc_mega_fused import (
                    mhc_mega_fused_enabled as _mega_on,
                )

                if _mega_on():
                    # R30/A: one fused kernel for the stats tail (shared GEMM
                    # producer; vendored big_fuse epilogue verbatim + a pre_mix
                    # global write; the collapse stays eager below).
                    pre, post_out, comb_out = mhc_delayed_stats_fused(
                        residual,
                        self.fn,
                        self.scale,
                        self.base,
                        norm_eps=self.norm_eps,
                        pre_eps=self.hc_eps,
                        sinkhorn_eps=self.hc_eps,
                        post_mult_value=2.0,
                        sinkhorn_repeat=self.hc_sinkhorn_iters,
                    )
                    return (
                        pre.reshape(*shape[:-1]).contiguous(),
                        post_out.reshape(*shape[:-1], 1),
                        comb_out.reshape(*shape[:-2], self.hc_mult, self.hc_mult),
                    )
                mixes = (
                    prenorm_fp32(residual, self.fn, self.norm_eps)
                    if self._prenorm_mode == "fp32"
                    else prenorm(residual, self.fn, self.norm_eps)
                )
                if mixes is None:
                    mixes = ops.mhc_pre_norm_fn(
                        residual, self.fn, None, self.norm_eps, n_splits=1
                    )
                pre, post_out, comb_out = ops.mhc_pre_split_mixes(
                    mixes, self.scale, self.base, self.hc_mult, 2.0, self.hc_eps
                )
                comb_out = ops.sinkhorn_normalize(
                    comb_out, repeat=self.hc_sinkhorn_iters, eps=self.hc_eps
                )
                return (
                    pre.reshape(*shape[:-1]).contiguous(),
                    post_out.reshape(*shape[:-1], 1),
                    comb_out.reshape(*shape[:-2], self.hc_mult, self.hc_mult),
                )

            from rtp_llm.models_py.modules.dsv4.hc import _v41_mhc_side_stream

            if (
                _v41_mhc_side_stream.mhc_side_stream_enabled()
                and not torch.cuda.is_current_stream_capturing()
            ):
                # R25/O1: issue the stats on the shared side stream so they
                # overlap the sublayer's main-stream work (attention / MoE
                # incl. the EP return wait); every consumer joins via
                # join_stats() (post/head/PP export). Same kernels, inputs
                # and order => byte-exact; only scheduling changes.
                _v41_mhc_side_stream.mhc_side_announce()
                main = torch.cuda.current_stream()
                side = _v41_mhc_side_stream.mhc_side_stream(x.device)
                ev_in = torch.cuda.Event(
                    enable_timing=_v41_mhc_side_stream.mhc_side_stats_enabled()
                )
                ev_in.record(main)
                side.wait_event(ev_in)
                with torch.cuda.stream(side):
                    with nvtx_phase("v41.hc.stats"):
                        pre_out, post, comb = _stats()
                # Both events need timing enabled for the diagnostics'
                # elapsed_time; harmless for the pure-ordering use otherwise.
                ev_out = torch.cuda.Event(
                    enable_timing=_v41_mhc_side_stream.mhc_side_stats_enabled()
                )
                ev_out.record(side)
                # Allocator edges: outputs are consumed on the main stream;
                # the residual is read on the side stream. record_stream
                # defers block reuse until the recorded stream's work done.
                pre_out.record_stream(main)
                post.record_stream(main)
                comb.record_stream(main)
                x.record_stream(side)
                self.pre_mix_out = pre_out
                self._stats_event = ev_out
                self._stats_ev_in = ev_in
            else:
                with nvtx_phase("v41.hc.stats"):
                    self.pre_mix_out, post, comb = _stats()
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
        # R25/O1: post consumes this boundary's side-stream stats outputs.
        self.join_stats()
        with nvtx_phase("v41.hc.post"):
            if self._post_mode == "ordered_fp32":
                out = post_ordered_fp32(x, residual, post, comb)
            else:
                out = post_delayed(self, x, residual, post, comb)
        # R29/A: pin pre_mix_out to a per-unit static buffer so any graph
        # capturing a SUCCESSOR boundary reads a permanently stable address
        # with always-current content (the observed stale-read fix).
        # R59: keep one static buffer PER SHAPE (was: a single buffer
        # reallocated on shape change) — the decoder-trim oscillates units
        # between full (1024-row) and compact (36-row) premix shapes, and the
        # reallocation churned the address the boundary probes key on. With
        # per-shape pins both addresses stay stable -> probes hit on both.
        if self.pre_mix_out is not None and _premix_pin_enabled():
            bufs = getattr(self, "_premix_static_by_shape", None)
            if bufs is None:
                bufs = self._premix_static_by_shape = {}
            skey = (
                tuple(self.pre_mix_out.shape),
                self.pre_mix_out.dtype,
                self.pre_mix_out.device,
            )
            buf = bufs.get(skey)
            if buf is None:
                buf = torch.empty_like(self.pre_mix_out)
                bufs[skey] = buf
            buf.copy_(self.pre_mix_out)
            self.pre_mix_out = buf
        return out


class DelayedHCHead(nn.Module):
    def __init__(self, last_ffn: DelayedHCUnit) -> None:
        super().__init__()
        self._last_ffn_ref = weakref.ref(last_ffn)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        unit = self._last_ffn_ref()
        if unit is None or unit.pre_mix_out is None:
            raise RuntimeError("V4.1 final FFN pre-mix is unavailable")
        unit.join_stats()  # R25/O1: head consumes the last boundary's pre_mix
        return unit.collapse(hidden, unit.pre_mix_out)


__all__ = [
    "DelayedHCUnit",
    "DelayedHCHead",
    "collapse_delayed",
    "collapse_ordered_fp32",
    "post_ordered_fp32",
]
